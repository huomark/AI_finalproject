# AI Final Project: Predicting Algorithm Tags for Competitive Programming Problems

## Project Overview

This project implements a multi-label classification system to predict algorithm tags for competitive programming problems using CodeBERT.

## Project Structure

```
AI_finalproject/
├── README.md
├── Makefile
├── tag.py  # crawler
├── test.py # crawler
├── dataset/
│   ├── alljianglycode.zip            # C++ source code files
│   └── codeforces_jiangly_tags.zip   # Corresponding tag files
├── codebert/
│   └── main.py            # CodeBERT implementation
└── textCNN/
  ├── textCNN.py         # TextCNN model implementation
  └── dataprepare.py     # Data preparation for TextCNN
```

## Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- pytorch-lightning: For model training framework
- transformers: For CodeBERT model and tokenization
- scikit-learn: For evaluation metrics
- pandas, numpy: For data manipulation
- matplotlib, seaborn: For visualization
- loguru: For logging
- jupytext: For Jupyter notebook text conversion
- flake8: For code linting
- black: For code formatting

## Using the Makefile

### Commands

- `make notebook`: Converts `main.py` to a Jupyter notebook (`main.ipynb`) using jupytext
- `make py`: Converts `main.ipynb` back to a Python script (`main.py`) using jupytext
- `make flake`: Runs flake8 linter on `main.py` to check for code quality issues
- `make black`: Formats `main.py` using black with a line length of 80 characters

## Model Architecture

### CodeBERT Classifier
- **Base Model**: `microsoft/codebert-base`
- **Architecture**: Pre-trained CodeBERT with classification head for multi-label prediction
- **Problem Type**: Multi-label classification
- **Fine-tuning Strategy**: Layer-wise learning rates with optional layer freezing

### Key Features
- **Multi-label Classification**: Predicts multiple algorithm tags simultaneously
- **Code Preprocessing**: Removes comments and empty lines from C++ code
- **Tokenization Strategy**: Supports both head and tail token prioritization
- **Class Imbalance Handling**: Multiple loss functions for imbalanced data

## Hyperparameters

### Model Configuration
| Parameter | Value | Description |
|-----------|--------|-------------|
| `CODEBERT_MODEL_NAME` | `microsoft/codebert-base` | Pre-trained model |
| `MAX_SEQUENCE_LENGTH` | 512 | Maximum input token length |
| `BATCH_SIZE` | 8 | Training batch size |
| `LEARNING_RATE` | 1e-4 | Base learning rate |
| `NUM_EPOCHS` | 10 | Maximum training epochs |

### Training Configuration
| Parameter | Value | Description |
|-----------|--------|-------------|
| `freeze_layers` | 0-9 | Number of layers to freeze |
| `train_split` | 0.7 | Training data ratio |
| `val_split` | 0.15 | Validation data ratio |
| `test_split` | 0.15 | Test data ratio |

### Loss Functions
- **BCE**: Binary Cross Entropy with optional class weights
- **Weighted BCE**: BCE with positive class weighting
- **Focal Loss**: Focal loss with α=0.25, γ=2.5 for severe imbalance

### Optimization
- **Optimizer**: AdamW with layer-wise learning rates
- **Scheduler**: Cosine Annealing LR (T_max=NUM_EPOCHS, eta_min=1e-6)
- **Weight Decay**: 0.01 for most parameters, 0.0 for bias and LayerNorm
- **Learning Rate Strategy**:
  - Base model: `learning_rate * 0.1`
  - Classifier head: `learning_rate`

## Data Processing

### Code Preprocessing
1. **Comment Removal**: Strips single-line (`//`) and multi-line (`/* */`) comments
2. **Whitespace Normalization**: Removes empty lines and normalizes spacing
3. **Tag Filtering**: Only includes tags with frequency ≥ 3 occurrences

### Tokenization Strategy
- **Default**: Standard BERT tokenization with truncation
- **Tail Priority**: Preserves end of code when truncating (experimental)

## Experiments Results

### **1. Architecture Comparison**

| Model    | Precision | Recall | F1 Score | Hamming Loss |
| -------- | --------- | ------ | -------- | ------------ |
| TextCNN  | 0.61      | 0.23   | 0.33     | 0.18         |
| CodeBERT | 0.46      | 0.58   | 0.51     | 0.10         |

### **2. Sequence Length Comparison (BCE)**

| MAX\_SEQ\_LEN | Precision | Recall | F1 Score | Hamming Loss |
| ------------- | --------- | ------ | -------- | ------------ |
| 256           | 0.44      | 0.56   | 0.49     | 0.11         |
| 512           | 0.46      | 0.58   | 0.51     | 0.10         |

### **3. Loss Function Comparison (CodeBERT, MAX\_SEQ\_LEN = 512)**

| Loss Function | Precision | Recall | F1 Score | Hamming Loss |
| ------------- | --------- | ------ | -------- | ------------ |
| BCE           | 0.46      | 0.58   | 0.51     | 0.10         |
| Focal         | 0.35      | 0.73   | 0.47     | 0.15         |

### **4. Token Position Comparison**

| Token Source | Precision | Recall | F1 Score | Hamming Loss |
| ------------ | --------- | ------ | -------- | ------------ |
| Head         | 0.46      | 0.58   | 0.51     | 0.10         |
| Tail         | 0.48      | 0.60   | 0.53     | 0.10         |

## Usage

### Model Configuration
Modify hyperparameters in the configuration section of `main.py`:
```python
# Model configuration
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
```
## Evaluation Metrics

The model is evaluated using:
- **Precision/Recall/F1**: Both micro and macro averaged
- **Hamming Loss**: Fraction of incorrect label predictions
- **Per-label Metrics**: Individual performance for each tag
- **Confusion Matrices**: Both per-label and unified visualizations