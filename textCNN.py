import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataprepare import get_data
from loguru import logger

logger.add("train.log", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

# ==== Load data ====
train_dataset, valid_dataset, vocab, tag_to_idx = get_data()
NUM_CLASSES = len(tag_to_idx)
VOCAB_SIZE = len(vocab)
EMBED_DIM = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
MAX_SEQ_LEN = 300

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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

    logger.info(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.numel()
    logger.info(f"          - Val Accuracy: {correct / total:.4f}")

# Save model
torch.save(model.state_dict(), "textcnn_model.pt")
