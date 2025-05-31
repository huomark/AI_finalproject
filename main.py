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

# %% _cell_guid="b7b0a96e-f5b3-4b3e-b15a-69727c7d55bc" _uuid="fc2c214c-5541-463a-b9a1-dcc19de78245" jupyter={"outputs_hidden": false}
import os
import numpy as np
import pandas as pd
import re

from loguru import logger
import sys

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO",
)

# %% [markdown] _cell_guid="b28b9e68-764e-46d5-a102-ed540025d54b" _uuid="e1f475aa-1ac8-4272-8416-34c3acab9d4a" jupyter={"outputs_hidden": false}
# ## Data Preprocessing

# %% [markdown] _cell_guid="d1880ded-85f1-4625-85ce-df28cc7ec601" _uuid="f09aa410-e289-4688-9e26-5ceaa834a1eb" jupyter={"outputs_hidden": false}
# ### Configuration

# %% _cell_guid="dd9b4772-0451-4be1-b1ee-e8035640f09d" _uuid="bf9e2472-7c06-4a0c-a6f5-d1866a5e8c4e" jupyter={"outputs_hidden": false}
PROJECT_ROOT = os.getcwd()
DATA_DIR = "/kaggle/input/user-submissions-dataset/dataset"
SUBMISSIONS_DIR = os.path.join(DATA_DIR, "submissions")
TAG_DIR = os.path.join(DATA_DIR, "tags")
METADATA_FILE = os.path.join(PROJECT_ROOT, "metadata.csv")

if not os.path.exists(SUBMISSIONS_DIR):
    logger.error(f"Submissions directory not found at {SUBMISSIONS_DIR}")
if not os.path.exists(TAG_DIR):
    logger.error(f"Tags directory not found at {TAG_DIR}")

# %% [markdown] _cell_guid="2b5efd3c-11a2-4b92-b643-908a7606555d" _uuid="415cd136-cac6-4658-9b37-f336a063017f" jupyter={"outputs_hidden": false}
# ### Load Code and Tags

# %% _cell_guid="ef7cb614-3d38-4719-a7bb-9895341aef91" _uuid="fc94801a-4ac8-4568-afd1-f3822eefa9b3" jupyter={"outputs_hidden": false}
data_records = []

for submission_filename in os.listdir(SUBMISSIONS_DIR):
    if not submission_filename.endswith(".txt"):
        continue

    problem_id = submission_filename.replace(".txt", "")
    tag_filename = submission_filename
    code_filepath = os.path.join(SUBMISSIONS_DIR, submission_filename)
    tag_filepath = os.path.join(TAG_DIR, tag_filename)

    code_content = None
    tag_list = []

    # Read source code
    try:
        with open(code_filepath, "r", encoding="utf-8") as code_file:
            code_content = code_file.read()
    except FileNotFoundError:
        logger.error(f"Code file {code_filepath} not found, but was listed")
        continue
    except Exception as e:
        logger.error(f"Could not read code file {code_filepath}: {e}")
        continue

    # Read tags
    try:
        with open(tag_filepath, "r", encoding="utf-8") as tag_file:
            tag_content = tag_file.read()
            tag_list = re.split(
                r"\s*,\s*", tag_content.strip()
            )  # Split by commas and strip whitespace
    except FileNotFoundError:
        logger.error(f"Tag file {tag_filepath} not found, but was listed")
        continue
    except Exception as e:
        logger.error(f"Could not read tag file {tag_filepath}: {e}")
        continue

    if code_content is not None:
        data_records.append(
            {
                "problem_id": problem_id,
                "raw_code": code_content,
                "tags": tag_list,
            }
        )

df = pd.DataFrame(data_records)

if not df.empty:
    logger.info(f"Successfully loaded {len(df)} records.")
    logger.info("Sample of loaded data:")
    print(df.head())
else:
    logger.warning(
        "No data records were loaded. Please check your `SUBMISSIONS_DIR` and `TAGS_DIR` paths and content."
    )


# %% [markdown] _cell_guid="9426b79a-2139-4730-a6fe-45b1a3150523" _uuid="2a270222-0e8b-4332-a38d-3a50fa16903a" jupyter={"outputs_hidden": false}
# ### Data Cleaning


# %% _cell_guid="5eb9709b-8a47-4b17-b567-c15d193bf8dc" _uuid="6b64f3a4-e735-41c2-af6e-05abe9ef5bd2" jupyter={"outputs_hidden": false}
def clean_cpp_code(code) -> str:
    """
    - Removes single-line comments
    - Removes multi-line comments
    - Removes empty lines and leading/trailing whitespace from lines.
    """
    if not isinstance(code, str):
        raise ValueError("Input code must be a string")

    # Remove single-line comments
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL | re.MULTILINE)

    # Remove multi-line comments
    code = re.sub(r"//.*", "", code)

    # Remove leading and trailing whitespace from each line
    lines = [line.strip() for line in code.splitlines() if line.strip()]

    cleaned_code = "\n".join(lines)
    return cleaned_code


if not df.empty and "raw_code" in df.columns:
    df["cleaned_code"] = df["raw_code"].apply(clean_cpp_code)

    if len(df) > 0:
        logger.info("--- Cleaning Example ---")
        idx_to_check = 0  # or a random index
        logger.info(f"Problem ID: {df['problem_id'].iloc[idx_to_check]}")
        logger.info("Original Code (first 500 chars):")
        print(df["raw_code"].iloc[idx_to_check][:500])
        logger.info("Cleaned Code (first 500 chars):")
        print(df["cleaned_code"].iloc[idx_to_check][:500])
else:
    logger.warning(
        "DataFrame is empty or 'raw_code_content' column is missing. Skipping cleaning demonstration."
    )

# %% [markdown] _cell_guid="ef62a66c-df95-4f59-9ddc-de43082f8c27" _uuid="d76c5bec-1585-45d5-8107-a1e9acd1a40d" jupyter={"outputs_hidden": false}
# ### Save Metadata file

# %% _cell_guid="3d5319a3-dea4-4f97-9fea-9cd6d7a1a112" _uuid="57fb0371-2184-4158-b446-c1ad6255f74c" jupyter={"outputs_hidden": false}
if not df.empty:
    entries = ["problem_id", "cleaned_code", "tags"]

    existing_columns = [col for col in entries if col in df.columns]

    if "cleaned_code" not in df.columns and "raw_code" in df.columns:
        logger.warning(
            "'cleaned_code' not found, metadata might not be fully processed for next steps."
        )

    df_metadata = df[existing_columns]

    try:
        df_metadata.to_csv(METADATA_FILE, index=False)
        logger.success(
            f"Successfully saved metadata for {len(df_metadata)} records to {METADATA_FILE}"
        )
        logger.info("Sample of metadata.csv content:")
        print(df_metadata.head())
    except Exception as e:
        logger.error(f"Error saving metadata file: {e}")

else:
    logger.warning("DataFrame is empty. No metadata file created.")

# %% [markdown]
# ## Code Tokenization

# %%
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


def tokenize_code(code: str):
    encoded = tokenizer(
        code,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    return input_ids, attention_mask


# %%
input_ids = []
attention_masks = []

for code_str in df["cleaned_code"]:
    ids, mask = tokenize_code(code_str)
    input_ids.append(ids.tolist())
    attention_masks.append(mask.tolist())

df["input_ids"] = input_ids
df["attention_mask"] = attention_masks

df.to_parquet("tokenized_metadata.parquet", index=False)

# %% [markdown]
# ## Processing Dataset

# %%
import ast
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# %%
all_tags = sorted(set(tag for tags in df["tags"] for tag in tags if tag != ""))
tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}


def to_multihot(tag_list):
    vec = [0] * len(all_tags)
    for tag in tag_list:
        if tag in tag2idx:
            vec[tag2idx[tag]] = 1
    return vec


df["labels"] = df["tags"].apply(to_multihot)


# %%
class CodeforcesDataset(Dataset):
    def __init__(self, df):
        self.input_ids = df["input_ids"].tolist()
        self.attention_masks = df["attention_mask"].tolist()
        self.labels = df["labels"].tolist()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_masks[idx], dtype=torch.long
            ),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


df = CodeforcesDataset(df)

# %%
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, shuffle=True
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True
)

# %%
train_loader = DataLoader(train_df, batch_size=16, shuffle=True)
val_loader = DataLoader(val_df, batch_size=16, shuffle=False)

# %% [markdown]
# ## Build Model

# %%
from transformers import AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim

# %%
NUM_TAGS = len(all_tags)
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=NUM_TAGS,
    problem_type="multi_label_classification",
)
model.to(device)

# %%
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# %% [markdown]
# ## Training Model

# %%
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# %%
for epoch in range(NUM_EPOCHS):
    model.train()
    train_losses = []
    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    model.eval()
    val_losses = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            val_losses.append(loss.item())

            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = (np.vstack(all_preds) >= 0.5).astype(int)
    all_labels = np.vstack(all_labels)
    precision = precision_score(
        all_labels, all_preds, average="micro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds, average="micro", zero_division=0
    )
    f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    print(
        f"Epoch {epoch+1} | Train Loss: {np.mean(train_losses):.4f} | Val Loss: {np.mean(val_losses):.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )
