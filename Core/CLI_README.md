# DGA Classifier CLI Tool

A command-line interface for training and evaluating an LSTM-based DNS domain classifier that distinguishes between benign domains, DGA (Domain Generation Algorithm) domains, and C2 (Command & Control) traffic.

## Overview

The `dga_classifier` tool implements an LSTM neural network trained from scratch in C++ (no external ML libraries). It encodes DNS query sequences and classifies them into three categories:
- **Class 0**: Benign domains
- **Class 1**: DGA (Domain Generation Algorithm) domains  
- **Class 2**: C2 (Command & Control) traffic

## Building

### Prerequisites
- g++ (C++17 or later)
- Standard C++ libraries

### Compile

From the `Core` directory:

```bash
make
# or manually:
g++ -std=c++17 -O2 -I../PreProcessing dga_classifier.cpp advanced_math.cpp \
    basic_math.cpp cross_entropy.cpp dense.cpp lstm.cpp lstm_backward.cpp \
    matrix_math.cpp rng.cpp vector_math.cpp -o dga_classifier
```

## Usage

```bash
dga_classifier <command> [options]
```

### Commands

#### 1. Train
Trains the LSTM model on a dataset.

```bash
dga_classifier train [options]
```

**Options:**
- `--train <file>` - Training CSV file (default: `train.csv`)
- `--test <file>` - Test CSV file (default: `test.csv`)
- `--model <file>` - Output model binary path (default: `best_model.bin`)
- `--epochs <n>` - Number of training epochs (default: 40)

**Example:**
```bash
cd Datasets
../Core/dga_classifier train --train train.csv --test test.csv --epochs 40
```

**Output:**
- Saves best model to specified path
- Displays per-epoch training loss, training accuracy, and test accuracy
- Early stopping when validation accuracy plateaus (patience: 8 epochs)

#### 2. Evaluate
Evaluates a trained model on test data with confusion matrix.

```bash
dga_classifier evaluate [options]
```

**Options:**
- `--model <file>` - Model binary file (default: `best_model.bin`)
- `--test <file>` - Test CSV file (default: `test.csv`)

**Example:**
```bash
cd Core
./dga_classifier evaluate --model best_model.bin --test ../Datasets/test.csv
```

**Output:**
- Confusion matrix showing true positives, false positives, etc.
- Overall accuracy and error rate
- Per-class breakdown

#### 3. Help
Shows usage information.

```bash
dga_classifier help
# or
dga_classifier --help
dga_classifier -h
```

## Dataset Format

CSV files with comma-separated columns: `domain,class`

**Example:**
```csv
domain,class
example.com,0
abcbot.net,1
c2server.org,2
```

**Class Labels:**
- `0` = Benign
- `1` = DGA
- `2` = C2

## Model Architecture

- **LSTM Layer**: 64 hidden units with 4 gates (forget, input, output, candidate)
- **Embedding**: One-hot encoded DNS character sequences (vocabulary size: 256)
- **Classification**: Fully connected layer (64 → 3 classes)
- **Regularization**: Dropout (rate: 0.3), gradient clipping (1.0)
- **Optimization**: SGD with learning rate decay (0.95x per epoch)
- **Loss**: Categorical cross-entropy

## Implementation Details

### Core Components

1. **LSTM Forward Pass** (`lstm.cpp/h`)
   - Computes cell state and hidden state across sequence
   - Implements forget, input, output gates and candidate activation

2. **LSTM Backward Pass** (`lstm_backward.cpp/h`)
   - BPTT (Backprop Through Time) with truncation (k=15)
   - Gradient computation for all LSTM parameters

3. **Dense Layer** (`dense.cpp/h`)
   - Fully connected classification layer
   - Forward and backward propagation

4. **Utilities**
   - Cross-entropy loss and softmax (`cross_entropy.cpp/h`)
   - Vector operations (`vector_math.cpp/h`)
   - Matrix operations (`matrix_math.cpp/h`)
   - Advanced math (`advanced_math.cpp/h`)
   - RNG (`rng.cpp/h`)

### Training Features

- **Stratified Sampling**: Balanced class representation during training
- **Early Stopping**: Stops when validation accuracy plateaus
- **Weight clipping**: Gradient clipping prevents exploding gradients
- **State Reset**: LSTM state reset between samples prevents stale activations
- **Dropout**: Applied to hidden layer during training, disabled during evaluation

## Performance

Reference results on the combined multiclass dataset:
- **Training Samples**: ~2.5M
- **Test Samples**: ~620k
- **Classes**: 3 (Benign, DGA, C2)
- **Test Accuracy**: ~41-42% (depends on class distribution and training epochs)

Note: Class 2 (C2) is severely underrepresented (~0.4% of data), making it challenging to classify accurately.

## Tips for Better Results

1. **Increase epochs**: Set `--epochs 100` or higher for better convergence
2. **Class weighting**: Consider implementing weighted loss for imbalanced classes
3. **Data augmentation**: Add DNS query variations during preprocessing
4. **Model tuning**: Adjust hidden size and dropout rate in `configure.h`

## Files

- `dga_classifier.cpp` - CLI tool with train/evaluate commands
- `lstm.cpp/h` - LSTM forward pass
- `lstm_backward.cpp/h` - LSTM backward pass and BPTT
- `dense.cpp/h` - Classification layer
- `cross_entropy.cpp/h` - Loss functions
- `Makefile` - Build script

## See Also

- `../PreProcessing/` - Dataset encoding and loading utilities
- `configure.h` - Model hyperparameters
- `../Datasets/` - Sample data (requires preprocessing)

## License

Educational use only.
