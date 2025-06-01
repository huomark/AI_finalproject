import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from dataprepare import get_data
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

logger.add("train.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

# ==== Load data ====
train_dataset, valid_dataset, experiment_dataset, vocab, tag_to_idx = get_data()
NUM_CLASSES = len(tag_to_idx)
VOCAB_SIZE = len(vocab)
EMBED_DIM = 256
FILTER_SIZES = [3, 6, 10, 20, 50, 70, 90]
NUM_FILTERS = 32

# for input, label in valid_dataset:
#     print(len(label))

# ==== Define TextCNN ====
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes, num_filters):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.unsqueeze(1)     # (batch, 1, seq_len, embed_dim)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # List of (batch, num_filters, ~)
        x = [torch.max(pool, dim=2)[0] for pool in x]  # Global max pooling
        x = torch.cat(x, dim=1)  # (batch, num_filters * len(filter_sizes))
        x = self.dropout(x)
        return self.fc(x)
    

# ==== Prepare for training ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, FILTER_SIZES, NUM_FILTERS).to(device)


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=128)
experiment_loader = DataLoader(experiment_dataset, batch_size=128)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
all_f1 = []
right_pr = 0.5

# ==== Training loop ====
for epoch in range(10):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    logger.info(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    alp = []
    all = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > right_pr
            correct += (preds == labels.bool()).sum().item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            preds = (probs > right_pr).astype(int)
            alp.extend(preds)
            all.extend(labels.cpu().numpy())
            total += labels.numel()
            print(correct, total)
            
    f1 = f1_score(alp, all, average='micro')
    all_f1.append(f1)
    logger.info(f"          - Val Accuracy: {correct / total:.4f}")

# evaluete
logger.info("Start evaluation on validation set...")
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in experiment_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > right_pr).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

precision = precision_score(all_labels, all_preds, average='micro')
recall = recall_score(all_labels, all_preds, average='micro')
f1_micro = f1_score(all_labels, all_preds, average='micro')
# recall = recall_score(all_labels, all_preds, average='micro')


# === Hamming Score ===
hamming_score = accuracy_score(all_labels, all_preds)


logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
logger.info(f"Hamming Score: {hamming_score:.4f}")
logger.info(f"f1_micro Score: {f1_micro:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
plt.plot(range(1, len(all_f1)+1), all_f1, label="F1 score", marker='o')
# plt.plot(range(1, len(val_recalls)+1), val_recalls, label="Val Recall", marker='o')
# plt.plot(range(1, len(val_hamming_scores)+1), val_hamming_scores, label="Val Hamming Score", marker='o')

plt.title("Training & Validation Metrics over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()