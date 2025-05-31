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
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from tqdm import tqdm
import random

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
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
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
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels
        }


# %%
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
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
    """CodeBERT-based multi-label classifier using PyTorch Lightning."""
    
    def __init__(self, model_name=CODEBERT_MODEL_NAME, num_labels=None, 
                 learning_rate=LEARNING_RATE, pos_weights=None, 
                 freeze_layers=8, loss_type='bce'):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.num_labels = num_labels
        self.loss_type = loss_type
        
        self.optimal_threshold = 0.5
        self.optimal_thresholds_per_class = None
        
        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        if freeze_layers > 0:
            for i, layer in enumerate(self.model.roberta.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Initialize loss function based on loss_type
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weights)
        elif loss_type == 'bce':
            self.criterion = WeightedBCELoss(pos_weight=pos_weights)
        elif loss_type == 'plain_bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'focal', 'bce', 'plain_bce'")
        
        # Metrics tracking
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
        
        logger.info(f"Using loss function: {loss_type}")
        logger.info(f"Frozen layers: {freeze_layers} out of {len(self.model.roberta.encoder.layer)}")
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store outputs for epoch-end metrics
        self.train_outputs.append({
            'loss': loss.detach().cpu(),
            'preds': torch.sigmoid(logits).detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.val_outputs.append({
            'loss': loss.detach().cpu(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits)
        
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.test_outputs.append({
            'loss': loss.detach().cpu(),
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return loss
    
    def on_train_epoch_end(self):
        if self.train_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean()
            all_preds = torch.cat([x['preds'] for x in self.train_outputs])
            all_labels = torch.cat([x['labels'] for x in self.train_outputs])
            
            # Calculate metrics
            preds_binary = (all_preds > 0.5).float()
            
            preds_np = preds_binary.numpy()
            labels_np = all_labels.numpy()
            
            f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
            f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
            precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
            hamming = hamming_loss(labels_np, preds_np)
            
            self.log("train_f1_micro", f1_micro, on_epoch=True)
            self.log("train_f1_macro", f1_macro, on_epoch=True)
            self.log("train_precision", precision, on_epoch=True)
            self.log("train_recall", recall, on_epoch=True)
            self.log("train_hamming_loss", hamming, on_epoch=True)
            
            self.train_outputs.clear()
    
    def on_validation_epoch_end(self):
        if self.val_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
            all_preds = torch.cat([x['preds'] for x in self.val_outputs])
            all_labels = torch.cat([x['labels'] for x in self.val_outputs])
            
            y_true_np = all_labels.numpy()
            y_probs_np = all_preds.numpy()
            
            try:
                best_threshold, best_score, _ = find_optimal_threshold(
                    y_true_np, y_probs_np, method='f1_micro'
                )
                self.optimal_threshold = best_threshold
                
                preds_binary_optimal = (all_preds >= best_threshold).float()
                preds_np_optimal = preds_binary_optimal.numpy()
                
                f1_micro = f1_score(y_true_np, preds_np_optimal, average='micro', zero_division=0)
                f1_macro = f1_score(y_true_np, preds_np_optimal, average='macro', zero_division=0)
                precision = precision_score(y_true_np, preds_np_optimal, average='micro', zero_division=0)
                recall = recall_score(y_true_np, preds_np_optimal, average='micro', zero_division=0)
                hamming = hamming_loss(y_true_np, preds_np_optimal)
                
                self.log("val_f1_micro", f1_micro, on_epoch=True, prog_bar=True)
                self.log("val_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
                self.log("val_precision", precision, on_epoch=True, prog_bar=True)
                self.log("val_recall", recall, on_epoch=True, prog_bar=True)
                self.log("val_hamming_loss", hamming, on_epoch=True)
                self.log("optimal_threshold", best_threshold, on_epoch=True)
                
                preds_binary_fixed = (all_preds > 0.5).float()
                f1_micro_fixed = f1_score(y_true_np, preds_binary_fixed.numpy(), average='micro', zero_division=0)
                improvement = f1_micro - f1_micro_fixed
                self.log("threshold_improvement", improvement, on_epoch=True)
                
            except Exception as e:
                logger.warning(f"Failed to optimize threshold: {e}. Using fixed threshold 0.5")
                # Fallback to fixed threshold
                preds_binary = (all_preds > 0.5).float()
                preds_np = preds_binary.numpy()
                
                f1_micro = f1_score(y_true_np, preds_np, average='micro', zero_division=0)
                f1_macro = f1_score(y_true_np, preds_np, average='macro', zero_division=0)
                precision = precision_score(y_true_np, preds_np, average='micro', zero_division=0)
                recall = recall_score(y_true_np, preds_np, average='micro', zero_division=0)
                hamming = hamming_loss(y_true_np, preds_np)
                
                self.log("val_f1_micro", f1_micro, on_epoch=True, prog_bar=True)
                self.log("val_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
                self.log("val_precision", precision, on_epoch=True, prog_bar=True)
                self.log("val_recall", recall, on_epoch=True, prog_bar=True)
                self.log("val_hamming_loss", hamming, on_epoch=True)
            
            self.val_outputs.clear()
    
    def on_test_epoch_end(self):
        if self.test_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.test_outputs]).mean()
            all_preds = torch.cat([x['preds'] for x in self.test_outputs])
            all_labels = torch.cat([x['labels'] for x in self.test_outputs])
            
            preds_binary = (all_preds >= self.optimal_threshold).float()
            
            preds_np = preds_binary.numpy()
            labels_np = all_labels.numpy()
            
            f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
            f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
            precision = precision_score(labels_np, preds_np, average='micro', zero_division=0)
            recall = recall_score(labels_np, preds_np, average='micro', zero_division=0)
            hamming = hamming_loss(labels_np, preds_np)
            
            self.log("test_f1_micro", f1_micro, on_epoch=True, prog_bar=True)
            self.log("test_f1_macro", f1_macro, on_epoch=True, prog_bar=True)
            self.log("test_precision", precision, on_epoch=True, prog_bar=True)
            self.log("test_recall", recall, on_epoch=True, prog_bar=True)
            self.log("test_hamming_loss", hamming, on_epoch=True)
            
            self.test_outputs.clear()
    
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.roberta.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.01,
                "lr": self.learning_rate * 0.1,
            },
            {
                "params": [p for n, p in self.model.roberta.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": self.learning_rate * 0.1,
            },
            {
                "params": [p for n, p in self.model.classifier.named_parameters()],
                "weight_decay": 0.01,
                "lr": self.learning_rate,
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1_micro",
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
            pos_weights.append(min(max(weight, 1.0), 5.0))
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
    freeze_layers=8,
    loss_type=loss_type
)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=NUM_EPOCHS,
    devices=1 if torch.cuda.is_available() else 0,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision='16-mixed' if torch.cuda.is_available() else 32,
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
            patience=5,
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
# Test the model
test_results = trainer.test(model, test_loader)

# Save the model and threshold information
model_save_path = os.path.join(PROCESSED_DATA_DIR, "codebert_classifier.pth")
torch.save(model.state_dict(), model_save_path)
logger.info(f"Model saved to {model_save_path}")

# Save the dataset info and optimal threshold for later use
dataset_info = {
    'tag2idx': dataset.tag2idx,
    'idx2tag': dataset.idx2tag,
    'num_labels': dataset.num_labels,
    'optimal_threshold': model.optimal_threshold,  # 保存最佳threshold
}

import pickle
with open(os.path.join(PROCESSED_DATA_DIR, "dataset_info.pkl"), 'wb') as f:
    pickle.dump(dataset_info, f)

logger.info("Training completed!")
logger.info(f"Test results: {test_results}")
logger.info(f"Optimal threshold: {model.optimal_threshold:.4f}")

# %% [markdown]
# ### Threshold Optimization

# %%
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

def find_optimal_threshold(y_true, y_probs, method='f1_micro', search_range=(0.1, 0.9), step=0.05):
    """
    尋找最佳threshold來最大化指定的評估指標
    
    Args:
        y_true: 真實標籤 (numpy array)
        y_probs: 預測機率 (numpy array)
        method: 優化目標 ('f1_micro', 'f1_macro', 'precision', 'recall')
        search_range: threshold搜尋範圍
        step: 搜尋步長
    
    Returns:
        best_threshold: 最佳threshold
        best_score: 最佳分數
        threshold_scores: 所有threshold的分數歷史
    """
    thresholds = np.arange(search_range[0], search_range[1] + step, step)
    threshold_scores = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if method == 'f1_micro':
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        elif method == 'f1_macro':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif method == 'precision':
            score = precision_score(y_true, y_pred, average='micro', zero_division=0)
        elif method == 'recall':
            score = recall_score(y_true, y_pred, average='micro', zero_division=0)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        threshold_scores.append(score)
    
    best_idx = np.argmax(threshold_scores)
    best_threshold = thresholds[best_idx]
    best_score = threshold_scores[best_idx]
    
    return best_threshold, best_score, list(zip(thresholds, threshold_scores))

def find_per_class_thresholds(y_true, y_probs, method='f1_score'):
    """
    為每個類別分別尋找最佳threshold
    
    Args:
        y_true: 真實標籤 (numpy array)
        y_probs: 預測機率 (numpy array) 
        method: 優化目標
    
    Returns:
        optimal_thresholds: 每個類別的最佳threshold
        class_scores: 每個類別的最佳分數
    """
    num_classes = y_true.shape[1]
    optimal_thresholds = []
    class_scores = []
    
    for class_idx in range(num_classes):
        y_true_class = y_true[:, class_idx]
        y_probs_class = y_probs[:, class_idx]
        
        best_threshold = 0.5
        best_score = 0.0
        
        if np.sum(y_true_class) > 0:
            thresholds = np.arange(0.1, 0.9, 0.05)
            
            for threshold in thresholds:
                y_pred_class = (y_probs_class >= threshold).astype(int)
                
                if method == 'f1_score':
                    score = f1_score(y_true_class, y_pred_class, zero_division=0)
                elif method == 'precision':
                    score = precision_score(y_true_class, y_pred_class, zero_division=0)
                elif method == 'recall':
                    score = recall_score(y_true_class, y_pred_class, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
        
        optimal_thresholds.append(best_threshold)
        class_scores.append(best_score)
    
    return optimal_thresholds, class_scores

def plot_threshold_curves(threshold_scores, title="Threshold Optimization"):
    """繪製threshold優化曲線"""
    thresholds, scores = zip(*threshold_scores)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 標記最佳點
    best_idx = np.argmax(scores)
    plt.plot(thresholds[best_idx], scores[best_idx], 'ro', markersize=8, 
             label=f'Best: {thresholds[best_idx]:.2f} (Score: {scores[best_idx]:.4f})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_with_optimal_threshold(y_true, y_probs, threshold_method='global', optimization_target='f1_micro'):
    """
    使用最佳threshold進行評估
    
    Args:
        y_true: 真實標籤
        y_probs: 預測機率
        threshold_method: 'global' 或 'per_class'
        optimization_target: 優化目標
    
    Returns:
        評估結果字典
    """
    results = {}
    
    if threshold_method == 'global':
        best_threshold, best_score, threshold_scores = find_optimal_threshold(
            y_true, y_probs, method=optimization_target
        )
        
        y_pred_optimal = (y_probs >= best_threshold).astype(int)
        
        results['method'] = 'Global Threshold'
        results['threshold'] = best_threshold
        results['threshold_scores'] = threshold_scores
        
        plot_threshold_curves(threshold_scores, f"Global Threshold Optimization ({optimization_target})")
        
    elif threshold_method == 'per_class':
        optimal_thresholds, class_scores = find_per_class_thresholds(
            y_true, y_probs, method='f1_score'
        )
        
        y_pred_optimal = np.zeros_like(y_probs)
        for class_idx, threshold in enumerate(optimal_thresholds):
            y_pred_optimal[:, class_idx] = (y_probs[:, class_idx] >= threshold).astype(int)
        
        results['method'] = 'Per-class Thresholds'
        results['thresholds'] = optimal_thresholds
        results['class_scores'] = class_scores
    
    f1_micro = f1_score(y_true, y_pred_optimal, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred_optimal, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred_optimal, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_optimal, average='micro', zero_division=0)
    hamming = hamming_loss(y_true, y_pred_optimal)
    
    y_pred_fixed = (y_probs >= 0.5).astype(int)
    f1_micro_fixed = f1_score(y_true, y_pred_fixed, average='micro', zero_division=0)
    
    results.update({
        'optimal_f1_micro': f1_micro,
        'optimal_f1_macro': f1_macro,
        'optimal_precision': precision,
        'optimal_recall': recall,
        'optimal_hamming_loss': hamming,
        'fixed_threshold_f1_micro': f1_micro_fixed,
        'improvement': f1_micro - f1_micro_fixed,
        'predictions': y_pred_optimal
    })
    
    return results


# %%
# Enhanced evaluation with threshold optimization
def evaluate_model_with_threshold_optimization(model, test_loader, dataset):
    """使用threshold優化進行詳細評估"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"]
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu()
            
            all_preds.append(probs)
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    print("\n" + "="*80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\n1. Global Threshold Optimization:")
    global_results = evaluate_with_optimal_threshold(
        all_labels, all_probs, 
        threshold_method='global', 
        optimization_target='f1_micro'
    )
    
    print(f"   Best threshold: {global_results['threshold']:.3f}")
    print(f"   F1 Micro (optimal): {global_results['optimal_f1_micro']:.4f}")
    print(f"   F1 Micro (fixed 0.5): {global_results['fixed_threshold_f1_micro']:.4f}")
    print(f"   Improvement: +{global_results['improvement']:.4f}")
    
    print("\n2. Per-class Threshold Optimization:")
    perclass_results = evaluate_with_optimal_threshold(
        all_labels, all_probs, 
        threshold_method='per_class'
    )
    
    print(f"   F1 Micro (per-class): {perclass_results['optimal_f1_micro']:.4f}")
    print(f"   F1 Macro (per-class): {perclass_results['optimal_f1_macro']:.4f}")
    print(f"   Improvement over fixed: +{perclass_results['improvement']:.4f}")
    
    # 顯示每類別的threshold
    print(f"\n   Per-class thresholds:")
    for i, (tag, threshold, score) in enumerate(zip(
        dataset.idx2tag.values(), 
        perclass_results['thresholds'], 
        perclass_results['class_scores']
    )):
        print(f"   {i:2d}. {tag:20s}: {threshold:.3f} (F1: {score:.3f})")
    
    print("\n3. Different Optimization Targets:")
    targets = ['f1_micro', 'f1_macro', 'precision', 'recall']
    
    for target in targets:
        result = evaluate_with_optimal_threshold(
            all_labels, all_probs, 
            threshold_method='global', 
            optimization_target=target
        )
        print(f"   {target:10s}: threshold={result['threshold']:.3f}, "
              f"F1_micro={result['optimal_f1_micro']:.4f}")
    
    print(f"\n4. Detailed Metrics (using best F1-micro threshold):")
    best_preds = global_results['predictions']
    
    f1_per_class = f1_score(all_labels, best_preds, average=None, zero_division=0)
    
    print(f"   Precision (Micro): {global_results['optimal_precision']:.4f}")
    print(f"   Recall (Micro): {global_results['optimal_recall']:.4f}")
    print(f"   Hamming Loss: {global_results['optimal_hamming_loss']:.4f}")
    
    print(f"\n   Top 10 classes by F1 score:")
    top_classes = np.argsort(f1_per_class)[::-1][:10]
    for i, class_idx in enumerate(top_classes):
        class_name = dataset.idx2tag[class_idx]
        print(f"   {i+1:2d}. {class_name:20s}: {f1_per_class[class_idx]:.4f}")
    
    return {
        'global_results': global_results,
        'perclass_results': perclass_results,
        'all_labels': all_labels,
        'all_probs': all_probs
    }

enhanced_results = evaluate_model_with_threshold_optimization(model, test_loader, dataset)


# %% [markdown]
# ### Threshold Analysis and Visualization

# %%
def analyze_threshold_sensitivity(y_true, y_probs, dataset, save_plots=True):
    """分析threshold敏感性並可視化"""
    
    thresholds = np.arange(0.05, 0.95, 0.01)
    metrics = {
        'f1_micro': [],
        'f1_macro': [], 
        'precision': [],
        'recall': [],
        'hamming_loss': []
    }
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        metrics['f1_micro'].append(f1_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision'].append(precision_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['hamming_loss'].append(hamming_loss(y_true, y_pred))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (metric_name, values) in enumerate(metrics.items()):
        if i < len(axes):
            axes[i].plot(thresholds, values, linewidth=2)
            axes[i].set_xlabel('Threshold')
            axes[i].set_ylabel(metric_name.replace('_', ' ').title())
            axes[i].set_title(f'{metric_name.replace("_", " ").title()} vs Threshold')
            axes[i].grid(True, alpha=0.3)
            
            if metric_name != 'hamming_loss':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            axes[i].plot(thresholds[best_idx], values[best_idx], 'ro', markersize=8,
                        label=f'Best: {thresholds[best_idx]:.2f}')
            axes[i].legend()
    
    if len(metrics) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    num_classes = min(10, y_true.shape[1])
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    class_counts = np.sum(y_true, axis=0)
    top_classes = np.argsort(class_counts)[::-1][:num_classes]
    
    for i, class_idx in enumerate(top_classes):
        if i < len(axes):
            y_true_class = y_true[:, class_idx]
            y_probs_class = y_probs[:, class_idx]
            
            if np.sum(y_true_class) > 0:
                class_f1_scores = []
                for threshold in thresholds:
                    y_pred_class = (y_probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
                    class_f1_scores.append(f1)
                
                axes[i].plot(thresholds, class_f1_scores, linewidth=2)
                axes[i].set_xlabel('Threshold')
                axes[i].set_ylabel('F1 Score')
                
                class_name = dataset.idx2tag.get(class_idx, f'Class_{class_idx}')
                axes[i].set_title(f'{class_name}\n(n={int(class_counts[class_idx])})')
                axes[i].grid(True, alpha=0.3)
                
                best_idx = np.argmax(class_f1_scores)
                axes[i].plot(thresholds[best_idx], class_f1_scores[best_idx], 'ro', markersize=6)
    
    for i in range(num_classes, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('per_class_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'thresholds': thresholds,
        'global_metrics': metrics,
        'best_thresholds': {
            metric: thresholds[np.argmax(values) if metric != 'hamming_loss' else np.argmin(values)]
            for metric, values in metrics.items()
        }
    }

if 'enhanced_results' in locals():
    sensitivity_analysis = analyze_threshold_sensitivity(
        enhanced_results['all_labels'], 
        enhanced_results['all_probs'], 
        dataset
    )
    
    print("\nBest thresholds for different metrics:")
    for metric, threshold in sensitivity_analysis['best_thresholds'].items():
        print(f"  {metric:15s}: {threshold:.3f}")
