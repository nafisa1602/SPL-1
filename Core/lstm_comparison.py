import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
import random

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors configure.h in C++ project)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_PATH = "/home/nafisa/Documents/SPL-1 Project Draft/Datasets/merged/train.csv"
TEST_PATH  = "/home/nafisa/Documents/SPL-1 Project Draft/Datasets/merged/test.csv"

MAX_LEN     = 100       # kMaxLength
HIDDEN_SIZE = 32        # kHiddenSize
EPOCHS      = 40
LR          = 0.001     # initialLR
LR_DECAY    = 0.95
DROPOUT     = 0.3
BATCH_SIZE  = 64
PATIENCE    = 8

# Vocabulary: same as C++ charToIndex()
# a-z → 1-26, 0-9 → 27-36, '.' → 37, '-' → 38, '_' → 39
VOCAB_SIZE  = 40        # 0 = padding

CLASS_NAMES = ["benign", "dga", "phishing", "tunneling", "c2"]
NUM_CLASSES = len(CLASS_NAMES)
SEED = 123456

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING  (mirrors PreProcessing pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def char_to_index(c: str) -> int:
    if 'a' <= c <= 'z': return ord(c) - ord('a') + 1
    if '0' <= c <= '9': return ord(c) - ord('0') + 27
    if c == '.': return 37
    if c == '-': return 38
    if c == '_': return 39
    return 0  # unknown → padding

def clean_dns(domain: str) -> str:
    cleaned = []
    for ch in str(domain):
        if 'A' <= ch <= 'Z':
            ch = chr(ord(ch) + (ord('a') - ord('A')))
        if ('a' <= ch <= 'z') or ('0' <= ch <= '9') or ch in {'.', '-', '_'}:
            cleaned.append(ch)

    if cleaned and cleaned[-1] == '.':
        cleaned.pop()
    return ''.join(cleaned)

def encode_domain(domain: str) -> list[int]:
    domain = clean_dns(domain)
    indices = [char_to_index(c) for c in domain]
    # Truncate or pad to MAX_LEN
    indices = indices[:MAX_LEN]
    indices += [0] * (MAX_LEN - len(indices))
    return indices

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class DNSDataset(Dataset):
    def __init__(self, path: str):
        print(f"[*] Loading {path} ...")
        df = pd.read_csv(
            path,
            header=None,
            names=["domain", "label"],
            quoting=3,
            on_bad_lines="skip",
            dtype=str,
        ).fillna("")
        df["domain"] = df["domain"].map(clean_dns)
        df["label"] = df["label"].str.lower().str.strip()
        df = df[df["label"].isin(CLASS_NAMES)].reset_index(drop=True)

        self.X = torch.tensor(
            [encode_domain(d) for d in df["domain"]], dtype=torch.long
        )
        self.y = torch.tensor(
            [CLASS_NAMES.index(l) for l in df["label"]], dtype=torch.long
        )
        print(f"    Loaded {len(self.y):,} samples")
        counts = {CLASS_NAMES[i]: (self.y == i).sum().item() for i in range(NUM_CLASSES)}
        for cls, cnt in counts.items():
            print(f"    {cls:12s}: {cnt:,}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (mirrors C++ architecture)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE, padding_idx=0)
        # One-hot style: embed each index as a VOCAB_SIZE-dim vector
        # (mirrors C++ one-hot input encoding)
        nn.init.eye_(self.embedding.weight)
        self.embedding.weight.requires_grad = False  # fixed one-hot

        self.lstm = nn.LSTM(
            input_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.reset_parameters()

    def reset_parameters(self):
        stddev = np.sqrt(2.0 / float(VOCAB_SIZE + HIDDEN_SIZE))
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[1:, 1:].copy_(torch.eye(VOCAB_SIZE - 1))

            nn.init.uniform_(self.lstm.weight_ih_l0, -stddev, stddev)
            nn.init.uniform_(self.lstm.weight_hh_l0, -stddev, stddev)
            nn.init.zeros_(self.lstm.bias_ih_l0)
            nn.init.zeros_(self.lstm.bias_hh_l0)
            self.lstm.bias_ih_l0[HIDDEN_SIZE:2 * HIDDEN_SIZE].fill_(1.0)

            nn.init.uniform_(self.classifier.weight, -stddev, stddev)
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        emb = self.embedding(x).float()          # (B, T, VOCAB_SIZE)
        nonzero = x.ne(0)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        lengths = torch.where(nonzero, positions + 1, torch.zeros_like(positions)).max(dim=1).values
        lengths = lengths.clamp(min=1)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)           # hn: (1, B, H)
        out = self.dropout(hn.squeeze(0))        # (B, H)
        return self.classifier(out)              # (B, NUM_CLASSES)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train():
    train_ds = DNSDataset(TRAIN_PATH)
    test_ds  = DNSDataset(TEST_PATH)

    train_by_class = {
        cls: torch.where(train_ds.y == cls)[0].tolist()
        for cls in range(NUM_CLASSES)
    }

    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = LSTMClassifier().to(DEVICE)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY)

    counts = torch.bincount(train_ds.y, minlength=NUM_CLASSES).float()
    weights = counts.sum() / (NUM_CLASSES * counts)
    weights[2] *= 1.8  # phishing boost
    weights = weights.to(DEVICE)
    print(f"[*] Class weights: { {CLASS_NAMES[i]: f'{weights[i].item():.2f}' for i in range(NUM_CLASSES)} }")
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_acc   = 0.0
    best_state = None
    patience_count = 0

    print(f"\n[*] Training for up to {EPOCHS} epochs (patience={PATIENCE}) ...")
    print(f"    Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for _ in range(len(train_ds)):
            cls = random.randrange(NUM_CLASSES)
            indices = train_by_class[cls]
            if not indices:
                continue
            idx = random.choice(indices)
            X_batch = train_ds.X[idx].unsqueeze(0).to(DEVICE)
            y_batch = train_ds.y[idx].unsqueeze(0).to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # matches gradClip=1.0
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += len(y_batch)

        train_acc = correct / total
        avg_loss  = total_loss / total

        # Evaluate
        test_acc = evaluate(model, test_loader, verbose=False)

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={avg_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f} | lr={current_lr:.6f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\n[*] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

        scheduler.step()

    elapsed = time.time() - start_time
    print(f"\n[*] Training complete in {elapsed:.1f}s")
    print(f"[*] Best test accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # Load best weights and do final detailed evaluation
    model.load_state_dict(best_state)
    torch.save(best_state, "pytorch_best_model.pt")
    print("[*] Saved best model to pytorch_best_model.pt")

    print("\n" + "=" * 60)
    print("FINAL EVALUATION (best model)")
    print("=" * 60)
    evaluate(model, test_loader, verbose=True)

    return best_acc

def evaluate(model, loader, verbose=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()

    if verbose:
        print(f"\nOverall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        print("Per-class Performance:")
        print(classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        ))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(all_labels, all_preds)
        header = f"{'':12s}" + "".join(f"{n:>12s}" for n in CLASS_NAMES)
        print(header)
        for i, row in enumerate(cm):
            print(f"{CLASS_NAMES[i]:12s}" + "".join(f"{v:12d}" for v in row))

        # ── Comparison table ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("COMPARISON: Custom C++ LSTM  vs  PyTorch LSTM")
        print("=" * 70)
        cpp_results = {
            "benign":    91.5,
            "dga":       90.0,
            "phishing":  78.1,
            "tunneling": 95.3,
            "c2":        95.3,
            "overall":   87.9,
        }
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
        print(f"{'Class':12s} {'C++ Accuracy':>14s} {'PyTorch Accuracy':>16s} {'Diff':>8s}")
        print("-" * 54)
        for idx, cls in enumerate(CLASS_NAMES):
            cpp_acc = cpp_results.get(cls, 0)
            total = cm[idx].sum()
            py_acc = (cm[idx][idx] / total * 100.0) if total else 0.0
            diff    = py_acc - cpp_acc
            sign    = "+" if diff >= 0 else ""
            print(f"{cls:12s} {cpp_acc:>13.1f}% {py_acc:>15.1f}% {sign}{diff:>6.1f}%")
        print("-" * 54)
        cpp_ov = cpp_results["overall"]
        py_ov  = acc * 100
        diff   = py_ov - cpp_ov
        sign   = "+" if diff >= 0 else ""
        print(f"{'OVERALL':12s} {cpp_ov:>13.1f}% {py_ov:>15.1f}% {sign}{diff:>6.1f}%")
        print("=" * 70)

    return acc

if __name__ == "__main__":
    train()