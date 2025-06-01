# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from collections import Counter
import re
import os
import ast

# %% [markdown]
# ## Load Data

# %%
CSV_PATH = "dataset/metadata.csv"
df = pd.read_csv(CSV_PATH)
df.dropna(subset=["cleaned_code", "tags"], inplace=True)  # 清垃圾

# %% [markdown]
# 把每個單字或符號切出來

# %%
def tokenize(code):
    tokens = re.findall(r"\w+|[^\s\w]", code)  # includes punctuations
    return tokens

df["tokens"] = df["cleaned_code"].apply(tokenize)

# %% [markdown]
# 數每個 token 的數量，取前 MAX_VOCAB_SIZE 多個的，其他都當作 <UNK> 

# %%
MAX_VOCAB_SIZE = 2000
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def is_valid_token(token):
    # 只允許長度 2~20 且為英數或簡單符號
    return 2 <= len(token) <= 20 and re.match(r'^[\w\+\-\*/=<>!\[\]{}().,;:]+$', token)

all_tokens = [token for tokens in df["tokens"] for token in tokens if is_valid_token(token)]
token_freq = Counter(all_tokens)
most_common = token_freq.most_common(MAX_VOCAB_SIZE - 2)
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
vocab.update({token: idx + 2 for idx, (token, _) in enumerate(most_common)})

# %% [markdown]
# 把每筆程式碼轉成固定長度是 MAX_SEQ_LEN 的數字陣列，不夠就補 <PAD>

# %%
MAX_SEQ_LEN = 750  # truncate/pad to fixed length

def encode_tokens(tokens):
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    if len(ids) < MAX_SEQ_LEN:
        ids += [vocab[PAD_TOKEN]] * (MAX_SEQ_LEN - len(ids))
    else:
        ids = ids[:MAX_SEQ_LEN]
    return ids

df["input_ids"] = df["tokens"].apply(encode_tokens)

# %% [markdown]
# ## 篩選前 10 個最常見的 tag

# %%
# 先把原本每筆的 tags 列攤平成一維計數
df["tags_list"] = df["tags"].apply(lambda s: ast.literal_eval(s))
all_label_lists = df["tags_list"].tolist()


tag_counter = Counter([tag for sublist in all_label_lists for tag in sublist])

# 取出前 10 名最常出現的標籤
top_tags = [tag for tag, _ in tag_counter.most_common(5)]
print("Top 10 tags:", top_tags)

# 建立只針對這 10 個標籤的映射
tag_to_idx = {tag: i for i, tag in enumerate(top_tags)}
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
NUM_TAGS = len(tag_to_idx)  # 10

# %% [markdown]
# ## 只保留包含這 10 個 tag 其中之一的樣本，並產生 10 維的 one-hot 向量

# %%
def encode_top10_tags(tag_list):
    """
    只對 top_tags 裡的標籤標為 1，其他標籤忽略。
    如果這筆資料沒有任何屬於 top_tags 的元素，將回傳全 0 的向量。
    """
    label = [0] * NUM_TAGS  # 全 10 維都設為 0
    for tag in tag_list:
        # print(f'tag: {tag}')
        if tag in tag_to_idx:
            label[tag_to_idx[tag]] = 1
    return label

# 先新增一欄 label_all10，之後再把「全 0」(沒有任何 top10 標籤) 的行捨棄
df["label_all10"] = df["tags_list"].apply(encode_top10_tags)

# 計算哪些行是全 0（二進位向量全為 0）
################################################################################!!!!!!!
mask_any_top10 = df["label_all10"].apply(lambda vec: sum(vec) >= 0)

# 只保留至少有一個 top10 標籤的樣本
df_filtered = df[mask_any_top10].reset_index(drop=True)
print(f"原始總樣本: {len(df)}，篩選後只含 top10 標籤的樣本: {len(df_filtered)}")

# 把 input_ids / labels 放到新的 DataFrame
inputs = df_filtered["input_ids"].tolist()
labels = df_filtered["label_all10"].tolist()

# %% [markdown]
# ## 定義 Dataset Class

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
# 切資料: 分成 training / validate / experiment

# %%
full_dataset = CodeTagDataset(inputs, labels)

# train/val/experiment 的大小
train_size = 1000
valid_size = 1000
experiment_size = len(full_dataset) - train_size - valid_size

train_dataset, valid_dataset, experiment_dataset = random_split(
    full_dataset,
    [train_size, valid_size, experiment_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Valid dataset size: {len(valid_dataset)}")
print(f"Experiment dataset size: {len(experiment_dataset)}")
print(f"Number of unique tags (top10): {NUM_TAGS}")

# %% [markdown]
# 在外部 call get_data() 可以直接拿到 train/val/experiment 與 vocab、tag_to_idx

# %%
def get_data():
    return train_dataset, valid_dataset, experiment_dataset, vocab, tag_to_idx
# %%
