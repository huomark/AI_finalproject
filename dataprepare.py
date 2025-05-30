# %% [markdown]
# # TextCNN Multi-label Data Preparation (for Codeforces Tags)

# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from collections import Counter
import re
import os

# %% [markdown]
# ## Load Data

# %%
CSV_PATH = "dataset/metadata.csv"
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["cleaned_code", "tags"], inplace=True)

# %% [markdown]
# ## Tokenizer (simple whitespace split)

# %%
def tokenize(code):
    tokens = re.findall(r"\w+|[^\s\w]", code)  # includes punctuations
    return tokens

df["tokens"] = df["cleaned_code"].apply(tokenize)

# %% [markdown]
# ## Build Vocabulary

# %%
MAX_VOCAB_SIZE = 5000
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

all_tokens = [token for tokens in df["tokens"] for token in tokens]
token_freq = Counter(all_tokens)
most_common = token_freq.most_common(MAX_VOCAB_SIZE - 2)
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab.update({token: idx + 2 for idx, (token, _) in enumerate(most_common)})
# for idx, (token, _) in enumerate(most_common):
#     print(f'{token}: {_}')

# %% [markdown]
# ## Encode Tokens

# %%
MAX_SEQ_LEN = 300  # truncate/pad to fixed length

def encode_tokens(tokens):
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    if len(ids) < MAX_SEQ_LEN:
        ids += [vocab[PAD_TOKEN]] * (MAX_SEQ_LEN - len(ids))
    else:
        ids = ids[:MAX_SEQ_LEN]
    return ids

df["input_ids"] = df["tokens"].apply(encode_tokens)

# %% [markdown]
# ## Build Tag-to-Index Map

# %%
all_tags = sorted(set(tag for tags in df["tags"] for tag in tags))
tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}

# %%
NUM_TAGS = len(tag_to_idx)

def encode_tags(tag_list):
    label = [0] * NUM_TAGS
    for tag in tag_list:
        if tag in tag_to_idx:
            label[tag_to_idx[tag]] = 1
    return label

df["label"] = df["tags"].apply(encode_tags)

# %% [markdown]
# ## Define Dataset Class

# %%
class CodeTagDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

# %% [markdown]
# ## Create Train/Valid Split

# %%
full_dataset = CodeTagDataset(df["input_ids"].tolist(), df["label"].tolist())

train_size = 2600
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")
print(f"Number of unique tags: {NUM_TAGS}")

# %% [markdown]
# You can now pass `train_dataset` and `valid_dataset` into PyTorch DataLoader to begin training.

# %%
def get_data():
    return train_dataset, valid_dataset, vocab, tag_to_idx
# %%
