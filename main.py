# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] _cell_guid="d4a6ff98-c7fa-48f8-ae08-7c190d31d0d9" _uuid="924e97cb-dffc-4128-b7a9-42082f0d1585" jupyter={"outputs_hidden": false}
# ## Preliminary

# %%
import os
import numpy as np
import pandas as pd
import re
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

def seed_everything(seed=42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if hasattr(pl, 'seed_everything'):
        pl.seed_everything(seed)

seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# ### Configuration

# %%
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")
TAG_DIR = os.path.join(DATA_DIR, "tags")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Model configuration
CODEBERT_MODEL_NAME = "microsoft/codebert-base"
MAX_SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10


# %% [markdown]
# ### Data Loading and Cleaning

# %%
def clean_cpp_code(code) -> str:
    """Clean C++ code by removing comments and empty lines."""
    if not isinstance(code, str):
        raise ValueError("Input code must be a string")

    # Remove multi-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL | re.MULTILINE)
    # Remove single-line comments
    code = re.sub(r"//.*", "", code)
    # Remove leading and trailing whitespace from each line
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    
    return "\n".join(lines)

def load_dataset():
    """Load and preprocess the dataset."""
    data_records = []

    for submission_filename in os.listdir(SUBMISSIONS_DIR):
        if not submission_filename.endswith(".txt"):
            continue

        problem_id = submission_filename.replace(".txt", "")
        code_filepath = os.path.join(SUBMISSIONS_DIR, submission_filename)
        tag_filepath = os.path.join(TAG_DIR, submission_filename)

        # Read source code
        try:
            with open(code_filepath, "r", encoding="utf-8") as code_file:
                code_content = code_file.read()
        except Exception as e:
            logger.error(f"Could not read code file {code_filepath}: {e}")
            continue

        # Read tags
        try:
            with open(tag_filepath, "r", encoding="utf-8") as tag_file:
                tag_content = tag_file.read()
                tag_list = re.split(r"\s*,\s*", tag_content.strip())
                tag_list = [tag.strip() for tag in tag_list if tag.strip()]
        except Exception as e:
            logger.error(f"Could not read tag file {tag_filepath}: {e}")
            continue

        if tag_list:
            data_records.append({
                "problem_id": problem_id,
                "raw_code": code_content,
                "tags": tag_list,
            })

    df = pd.DataFrame(data_records)
    
    if not df.empty:
        df["cleaned_code"] = df["raw_code"].apply(clean_cpp_code)
        logger.info(f"Successfully loaded {len(df)} records.")
        
        all_tags = []
        for tags in df["tags"]:
            all_tags.extend(tags)
        from collections import Counter
        tag_counts = Counter(all_tags)
        logger.info(f"Top 10 most common tags: {tag_counts.most_common(10)}")
        logger.info(f"Total unique tags: {len(tag_counts)}")
    else:
        logger.warning("No data records were loaded.")
    
    return df

# Load the dataset
df = load_dataset()


# %%
class CodeforceDataset(Dataset):
    """Custom Dataset for Codeforces problems with vocabulary limitation."""
    
    def __init__(self, df, tokenizer, max_length=512, use_vocab_filtered=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_vocab_filtered = use_vocab_filtered
        
        # Choose which code column to use
        if use_vocab_filtered and 'vocab_filtered_code' in df.columns:
            self.code_column = 'vocab_filtered_code'
            logger.info("Using vocabulary-filtered code")
        else:
            self.code_column = 'cleaned_code'
            logger.info("Using original cleaned code")
        
        all_tags = []
        for tags in df["tags"]:
            all_tags.extend(tags)
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        min_count = 3
        frequent_tags = [tag for tag, count in tag_counts.items() if count >= min_count]
        
        self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(frequent_tags))}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.num_labels = len(frequent_tags)
        
        logger.info(f"Found {len(tag_counts)} total tags, using {self.num_labels} frequent tags (>= {min_count} occurrences)")
        logger.info(f"Filtered tags: {list(self.tag2idx.keys())[:20]}...")
        
        # Convert tags to multi-hot encoding
        self.labels = []
        valid_samples = []
        
        for idx, tags in enumerate(df["tags"]):
            label_vec = [0] * self.num_labels
            has_valid_tag = False
            
            for tag in tags:
                if tag.strip() in self.tag2idx:
                    label_vec[self.tag2idx[tag.strip()]] = 1
                    has_valid_tag = True
            
            if has_valid_tag:
                self.labels.append(label_vec)
                valid_samples.append(idx)
        
        self.df = df.iloc[valid_samples].reset_index(drop=True)
        
        logger.info(f"Kept {len(self.labels)} samples with valid tags out of {len(df)} total samples")
        
        label_sums = np.sum(self.labels, axis=0)
        logger.info(f"Label distribution: min={np.min(label_sums)}, max={np.max(label_sums)}, mean={np.mean(label_sums):.2f}")
    
    def _tokenize_with_tail_priority(self, code):
        """
        Tokenize code with priority to keep the tail (end) of the code.
        If code is too long, truncate from the head (beginning).
        """
        # First tokenize without truncation to get full tokens
        full_tokens = self.tokenizer.tokenize(code)
        
        # If within limit, use normal tokenization
        if len(full_tokens) <= self.max_length - 2:  # -2 for [CLS] and [SEP] tokens
            encoding = self.tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            return encoding
        
        # If too long, keep the tail part
        max_code_tokens = self.max_length - 2  # Reserve space for special tokens
        tail_tokens = full_tokens[-max_code_tokens:]  # Take last N tokens
        
        # Convert tokens back to string
        tail_code = self.tokenizer.convert_tokens_to_string(tail_tokens)
        
        # Tokenize the truncated code
        encoding = self.tokenizer(
            tail_code,
            truncation=True,
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoding
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        code = self.df.iloc[idx][self.code_column]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # Use custom tokenization that prioritizes tail
        encoding = self._tokenize_with_tail_priority(code)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }


# %%
class CodeforceDataset(Dataset):
    """Custom Dataset for Codeforces problems."""
    
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        all_tags = []
        for tags in df["tags"]:
            all_tags.extend(tags)
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        min_count = 3
        frequent_tags = [tag for tag, count in tag_counts.items() if count >= min_count]
        
        self.tag2idx = {tag: idx for idx, tag in enumerate(sorted(frequent_tags))}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.num_labels = len(frequent_tags)
        
        logger.info(f"Found {len(tag_counts)} total tags, using {self.num_labels} frequent tags (>= {min_count} occurrences)")
        logger.info(f"Filtered tags: {list(self.tag2idx.keys())[:20]}...")
        
        # Convert tags to multi-hot encoding
        self.labels = []
        valid_samples = []
        
        for idx, tags in enumerate(df["tags"]):
            label_vec = [0] * self.num_labels
            has_valid_tag = False
            
            for tag in tags:
                if tag.strip() in self.tag2idx:
                    label_vec[self.tag2idx[tag.strip()]] = 1
                    has_valid_tag = True
            
            if has_valid_tag:
                self.labels.append(label_vec)
                valid_samples.append(idx)
        
        self.df = df.iloc[valid_samples].reset_index(drop=True)
        
        logger.info(f"Kept {len(self.labels)} samples with valid tags out of {len(df)} total samples")
        
        label_sums = np.sum(self.labels, axis=0)
        logger.info(f"Label distribution: min={np.min(label_sums)}, max={np.max(label_sums)}, mean={np.mean(label_sums):.2f}")

    def _tokenize_with_tail_priority(self, code):
        """
        Tokenize code with priority to keep the tail (end) of the code.
        If code is too long, truncate from the head (beginning).
        """
        # First tokenize without truncation to get full tokens
        full_tokens = self.tokenizer.tokenize(code)
        
        # If within limit, use normal tokenization
        if len(full_tokens) <= self.max_length - 2:  # -2 for [CLS] and [SEP] tokens
            encoding = self.tokenizer(
                code,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            return encoding
        
        # If too long, keep the tail part
        max_code_tokens = self.max_length - 2  # Reserve space for special tokens
        tail_tokens = full_tokens[-max_code_tokens:]  # Take last N tokens
        
        # Convert tokens back to string
        tail_code = self.tokenizer.convert_tokens_to_string(tail_tokens)
        
        # Tokenize the truncated code
        encoding = self.tokenizer(
            tail_code,
            truncation=True,
            padding="max_length", 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoding
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        code = self.df.iloc[idx]["cleaned_code"]
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # Tokenize
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize from code tail
        # encoding = self._tokenize_with_tail_priority(code)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }


# %%
def init_weights(m):
    """Initialize weights for linear layers."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float))
        else:
            self.pos_weight = None
    
    def forward(self, inputs, targets):
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for handling class imbalance."""
    
    def __init__(self, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float))
        else:
            self.pos_weight = None
    
    def forward(self, inputs, targets):
        if inputs.device != targets.device:
            targets = targets.to(inputs.device)
            
        return nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight
        )

class CodeBERTClassifier(pl.LightningModule):
    """CodeBERT classifier using AutoModelForSequenceClassification."""
    
    def __init__(self, model_name=CODEBERT_MODEL_NAME, num_labels=None, 
                 learning_rate=LEARNING_RATE, pos_weights=None, 
                 freeze_layers=9, loss_type='focal', hidden_dims=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.loss_type = loss_type
        
        # Load pre-trained model with classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Freeze layers (first freeze_layers out of 12)
        if freeze_layers > 0:
            # For AutoModelForSequenceClassification, the base model is usually called 'roberta' or 'bert'
            base_model = getattr(self.model, 'roberta', getattr(self.model, 'bert', None))
            if base_model is not None:
                layers_to_freeze = [base_model.embeddings, *base_model.encoder.layer[:freeze_layers]]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Initialize loss function
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.5, pos_weight=pos_weights)
        elif loss_type == 'bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weights)
        elif loss_type == 'plain_bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        logger.info(f"Using loss function: {loss_type}")
        logger.info(f"Frozen layers: {freeze_layers}")
        logger.info(f"Model loaded with {num_labels} labels")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def get_preds(self, y):
        """Get binary predictions from probabilities."""
        return (y >= 0.5).long()
    
    def forward(self, input_ids, attention_mask):
        # Use the model's forward pass which returns logits
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = self.get_preds(probs)
        
        # Convert to numpy for sklearn metrics
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
        hamming = hamming_loss(labels_np, preds_np)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_micro", f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_hamming_loss", hamming, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = self.get_preds(probs)
        
        # Convert to numpy for sklearn metrics
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        precision_micro = precision_score(labels_np, preds_np, average='micro', zero_division=0)
        precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        recall_micro = recall_score(labels_np, preds_np, average='micro', zero_division=0)
        recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        hamming = hamming_loss(labels_np, preds_np)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1_micro", f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1_macro", f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision_micro", precision_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_precision_macro", precision_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall_micro", recall_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_recall_macro", recall_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_hamming_loss", hamming, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        # Use different learning rates for different parts
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad and 'classifier' not in n],
                "weight_decay": 0.01,
                "lr": self.learning_rate * 0.1,  # Lower LR for base model
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad and 'classifier' not in n],
                "weight_decay": 0.0,
                "lr": self.learning_rate * 0.1,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if 'classifier' in n and p.requires_grad],
                "weight_decay": 0.01,
                "lr": self.learning_rate,  # Higher LR for classifier head
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # Use cosine annealing for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }


# %%
# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)

# Create dataset
dataset = CodeforceDataset(df, tokenizer, MAX_SEQUENCE_LENGTH)

if len(dataset) < 10:
    logger.error("Dataset too small! Need at least 10 samples for training.")
    raise ValueError("Insufficient data for training")

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=0
)

logger.info(f"Train size: {len(train_dataset)}")
logger.info(f"Val size: {len(val_dataset)}")
logger.info(f"Test size: {len(test_dataset)}")
logger.info(f"Number of labels: {dataset.num_labels}")


# %%
# Calculate class weights for imbalanced data
def calculate_pos_weights(dataset):
    """Calculate positive weights for each class."""
    all_labels = []
    
    if hasattr(dataset, 'dataset'):
        original_dataset = dataset.dataset
        indices = dataset.indices
        for idx in indices:
            all_labels.append(original_dataset.labels[idx])
    else:
        for i in range(len(dataset)):
            all_labels.append(dataset.labels[i])
    
    all_labels = np.array(all_labels)
    pos_counts = np.sum(all_labels, axis=0)
    neg_counts = len(all_labels) - pos_counts
    
    # Calculate pos_weight for each class
    pos_weights = []
    for i in range(all_labels.shape[1]):
        if pos_counts[i] > 0:
            weight = neg_counts[i] / pos_counts[i]
            pos_weights.append(min(weight, 7.5))
        else:
            pos_weights.append(1.0)
    
    logger.info(f"Positive counts per class: {pos_counts}")
    logger.info(f"Calculated pos_weights range: {min(pos_weights):.2f} - {max(pos_weights):.2f}")
    
    return pos_weights

pos_weights = calculate_pos_weights(train_dataset)
logger.info(f"Calculated pos_weights: {pos_weights[:10]}...")  # Show first 10

# %%
# Initialize model with different loss options
loss_type = 'bce' 

model = CodeBERTClassifier(
    model_name=CODEBERT_MODEL_NAME,
    num_labels=dataset.num_labels,
    learning_rate=LEARNING_RATE,
    pos_weights=pos_weights,
    freeze_layers=0,
    loss_type=loss_type
)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    devices=1 if torch.cuda.is_available() else 0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision='16-mixed' if torch.cuda.is_available() else 32,`
    log_every_n_steps=5,
    enable_checkpointing=True,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            monitor='val_f1_micro',
            mode='max',
            save_top_k=1,
            filename='best-model-{epoch:02d}-{val_f1_micro:.3f}',
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_f1_micro',
            mode='max',
            patience=3,
            verbose=True,
            min_delta=0.001
        )
    ]
)

# Train the model
trainer.fit(model, train_loader, val_loader)


# %% [markdown]
# ## Model Evaluation

# %%
def plot_confusion_matrices(y_true, y_pred, idx2tag, save_dir=None):
    """Plot confusion matrices for multi-label classification."""
    
    # Create multi-label confusion matrices
    cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)
    
    # Calculate number of subplots needed
    n_labels = len(idx2tag)
    n_cols = 4
    n_rows = (n_labels + n_cols - 1) // n_cols
    
    # Create figure for individual class confusion matrices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Confusion Matrices for Each Label', fontsize=16, y=0.98)
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_labels):
        row = i // n_cols
        col = i % n_cols
        
        cm = cm_multilabel[i]
        ax = axes[row, col]
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not ' + idx2tag[i], idx2tag[i]],
                   yticklabels=['Not ' + idx2tag[i], idx2tag[i]])
        ax.set_title(f'Label: {idx2tag[i]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_labels, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrices_per_label.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Per-label confusion matrices saved to {save_dir}")
    
    plt.show()
    
    # Create unified confusion matrix
    plot_unified_confusion_matrix(y_true, y_pred, save_dir)
    
    # Create integrated confusion matrix
    plot_integrated_confusion_matrix(y_true, y_pred, idx2tag, save_dir)
    
    # Create overall metrics visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Multi-Label Classification Metrics Overview', fontsize=16)
    
    # 1. Label frequency in predictions vs true labels
    pred_freq = np.sum(y_pred, axis=0)
    true_freq = np.sum(y_true, axis=0)
    
    x_pos = np.arange(len(idx2tag))
    width = 0.35
    
    ax1.bar(x_pos - width/2, true_freq, width, label='True', alpha=0.8)
    ax1.bar(x_pos + width/2, pred_freq, width, label='Predicted', alpha=0.8)
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Label Frequency: True vs Predicted')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([idx2tag[i] for i in range(len(idx2tag))], rotation=45)
    ax1.legend()
    
    # 2. Per-label F1 scores
    f1_per_label = []
    for i in range(n_labels):
        f1 = f1_score(y_true[:, i], y_pred[:, i], average='micro', zero_division=0)
        f1_per_label.append(f1)
    
    ax2.bar(x_pos, f1_per_label)
    ax2.set_xlabel('Labels')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score per Label')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([idx2tag[i] for i in range(len(idx2tag))], rotation=45)
    ax2.set_ylim(0, 1)
    
    # 3. Precision per label
    precision_per_label = []
    for i in range(n_labels):
        prec = precision_score(y_true[:, i], y_pred[:, i], average='micro', zero_division=0)
        precision_per_label.append(prec)
    
    ax3.bar(x_pos, precision_per_label, color='orange')
    ax3.set_xlabel('Labels')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision per Label')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([idx2tag[i] for i in range(len(idx2tag))], rotation=45)
    ax3.set_ylim(0, 1)
    
    # 4. Recall per label
    recall_per_label = []
    for i in range(n_labels):
        rec = recall_score(y_true[:, i], y_pred[:, i], average='micro', zero_division=0)
        recall_per_label.append(rec)
    
    ax4.bar(x_pos, recall_per_label, color='green')
    ax4.set_xlabel('Labels')
    ax4.set_ylabel('Recall')
    ax4.set_title('Recall per Label')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([idx2tag[i] for i in range(len(idx2tag))], rotation=45)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_overview.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Metrics overview saved to {save_dir}")
    
    plt.show()
    
    return f1_per_label, precision_per_label, recall_per_label

def plot_unified_confusion_matrix(y_true, y_pred, save_dir=None):
    """Plot unified 2x2 confusion matrix treating all labels as single classification problem."""
    
    # Flatten all predictions and true labels
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Calculate confusion matrix elements
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    
    # Create 2x2 confusion matrix
    unified_cm = np.array([[tn, fp], [fn, tp]])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Unified Confusion Matrix Analysis', fontsize=16)
    
    # Plot confusion matrix
    sns.heatmap(unified_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Predicted: 0', 'Predicted: 1'],
               yticklabels=['Actual: 0', 'Actual: 1'])
    ax1.set_title('Unified Confusion Matrix\n(All Labels Combined)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Add percentage annotations
    total = np.sum(unified_cm)
    for i in range(2):
        for j in range(2):
            percentage = unified_cm[i, j] / total * 100
            ax1.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    # Calculate and display metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Create metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    values = [accuracy, precision, recall, f1, specificity]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_title('Unified Classification Metrics')
    ax2.set_ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'unified_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Unified confusion matrix saved to {save_dir}")
    
    plt.show()
    
    # Print detailed statistics
    logger.info("\n=== Unified Confusion Matrix Statistics ===")
    logger.info(f"True Negatives (TN): {tn:,}")
    logger.info(f"False Positives (FP): {fp:,}")
    logger.info(f"False Negatives (FN): {fn:,}")
    logger.info(f"True Positives (TP): {tp:,}")
    logger.info(f"Total Predictions: {total:,}")
    logger.info(f"")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall (Sensitivity): {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Specificity: {specificity:.4f}")
    logger.info(f"")
    logger.info(f"Positive Class Rate: {(tp + fn) / total:.4f}")
    logger.info(f"Negative Class Rate: {(tn + fp) / total:.4f}")
    
    return unified_cm, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

# ...existing code...

# Test the model
test_results = trainer.test(model, test_loader)

# Get predictions for confusion matrix
logger.info("Getting test predictions for confusion matrix...")
y_true, y_pred = get_test_predictions(model, test_loader, device)

# Plot confusion matrices
logger.info("Plotting confusion matrices...")
f1_per_label, precision_per_label, recall_per_label = plot_confusion_matrices(
    y_true, y_pred, dataset.idx2tag, save_dir=PROCESSED_DATA_DIR
)

# Print detailed per-label metrics
logger.info("\nDetailed per-label metrics:")
for i, tag in dataset.idx2tag.items():
    logger.info(f"{tag}: F1={f1_per_label[i]:.3f}, Precision={precision_per_label[i]:.3f}, Recall={recall_per_label[i]:.3f}")

# Save the model
model_save_path = os.path.join(PROCESSED_DATA_DIR, "codebert_classifier.pth")
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")

# Save the dataset info for later use
dataset_info = {
    'tag2idx': dataset.tag2idx,
    'idx2tag': dataset.idx2tag,
    'num_labels': dataset.num_labels
}

import pickle
with open(os.path.join(PROCESSED_DATA_DIR, "dataset_info.pkl"), 'wb') as f:
    pickle.dump(dataset_info, f)

# Save detailed metrics
metrics_data = {
    'f1_per_label': f1_per_label,
    'precision_per_label': precision_per_label,
    'recall_per_label': recall_per_label,
    'y_true': y_true,
    'y_pred': y_pred,
    'test_results': test_results
}

with open(os.path.join(PROCESSED_DATA_DIR, "test_metrics.pkl"), 'wb') as f:
    pickle.dump(metrics_data, f)

logger.info("Training completed!")
logger.info(f"Test results: {test_results}")
logger.info(f"Confusion matrices and metrics saved to {PROCESSED_DATA_DIR}")
