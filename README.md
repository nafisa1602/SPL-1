# SPL-1 Project Draft

Character-level DNS threat detection using a custom C++ LSTM classifier, with a PyTorch comparison script.

## Overview

This project classifies DNS domains into 5 classes:

- `benign`
- `dga`
- `phishing`
- `tunneling`
- `c2`

Core model pipeline:

1. Clean and encode domain strings into fixed-length integer sequences.
2. Run a custom LSTM forward pass in C++.
3. Predict class logits with a dense output layer.
4. Train with cross-entropy loss and backpropagation through time.

## Repository Structure

- `Core/`: C++ model training, inference, and CLI tools.
- `PreProcessing/`: header-based DNS cleaning, encoding, label mapping, and dataset loader utilities.
- `Datasets/`: raw and merged datasets.

Important files:

- `Core/train.cpp`: trains and saves `best_model.bin`.
- `Core/percentage.cpp`: evaluates saved model and prints confusion matrix.
- `Core/cli_classifier.cpp`: interactive and batch DNS classification CLI.
- `Core/lstm_comparison.py`: PyTorch baseline/comparison training script.
- `PreProcessing/csv_loader.h`: resolves dataset paths and loads CSV data.

## Requirements

### C++

- Linux (tested)
- `g++` with C++17 support

### Python (optional, for comparison script)

- Python 3.10+
- Packages:
  - `torch`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install Python packages (inside your virtual environment):

```bash
pip install torch pandas numpy scikit-learn
```

## Dataset Expectations

Training/evaluation code expects these files:

- `Datasets/merged/train.csv`
- `Datasets/merged/test.csv`

Each CSV should be in `domain,label` format (with header).

The loader checks these locations in order:

1. `$DATASET_PATH/<filename>` (if `DATASET_PATH` is set)
2. `../Datasets/merged/<filename>`
3. `Datasets/merged/<filename>`
4. `./Datasets/merged/<filename>`

## Quick Start (C++)

From project root:

```bash
cd Core
```

### 1) Train the custom C++ model

```bash
g++ -std=c++17 -O2 -I../PreProcessing \
  train.cpp lstm.cpp lstm_backward.cpp dense.cpp cross_entropy.cpp \
  matrix_math.cpp vector_math.cpp advanced_math.cpp basic_math.cpp rng.cpp \
  -o train

./train
```

Output model:

- `Core/best_model.bin`

### 2) Evaluate with confusion matrix

```bash
g++ -std=c++17 -O2 -I../PreProcessing \
  percentage.cpp lstm.cpp dense.cpp matrix_math.cpp vector_math.cpp \
  advanced_math.cpp basic_math.cpp rng.cpp \
  -o percentage

./percentage
```

### 3) Run interactive or batch classification

```bash
g++ -std=c++17 -O2 -I../PreProcessing \
  cli_classifier.cpp lstm.cpp dense.cpp matrix_math.cpp vector_math.cpp \
  advanced_math.cpp basic_math.cpp rng.cpp \
  -o cli_classifier
```

Interactive mode:

```bash
./cli_classifier
```

Single-domain query:

```bash
./cli_classifier -q google.com
```

Batch from CSV:

```bash
./cli_classifier -f ../Datasets/merged/test.csv
```

Use a specific model file:

```bash
./cli_classifier -m best_model.bin -q example.com
```

Show help:

```bash
./cli_classifier --help
```

## PyTorch Comparison

Run from `Core/`:

```bash
python lstm_comparison.py
```

This script trains a PyTorch LSTM baseline and saves:

- `Core/pytorch_best_model.pt`

Note: the script currently uses absolute dataset paths. Update `TRAIN_PATH` and `TEST_PATH` in `Core/lstm_comparison.py` if your project path differs.

## Notes

- Preprocessing utilities are mostly header-only and shared by C++ binaries via `-I../PreProcessing`.
- Class index mapping used across the project:
  - `0: benign`
  - `1: dga`
  - `2: phishing`
  - `3: tunneling`
  - `4: c2`

## Troubleshooting

- If you see `configure.h`/`csv_loader.h` not found, ensure you compile with:
  - `-I../PreProcessing`
- If datasets are not found, either:
  - run binaries from `Core/`, or
  - set `DATASET_PATH` to the directory containing `train.csv` and `test.csv`.

Example:

```bash
export DATASET_PATH="/absolute/path/to/Datasets/merged"
./train
```
