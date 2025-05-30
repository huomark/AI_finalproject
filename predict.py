import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from textCNN import TextCNN
from dataprepare import get_data
from loguru import logger

# === initial ===
train_dataset, valid_dataset, vocab, tag_to_idx = get_data()
idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}

NUM_CLASSES = len(tag_to_idx)
VOCAB_SIZE = len(vocab)
EMBED_DIM = 128
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS = 100
MAX_SEQ_LEN = 300

# === data ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextCNN(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES, FILTER_SIZES, NUM_FILTERS).to(device)
model.load_state_dict(torch.load("textcnn_model.pt", map_location=device))
model.eval()

logger.info("Start evaluation on validation set...")

# === predict ===
all_preds = []
all_labels = []

valid_loader = DataLoader(valid_dataset, batch_size=32)
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# === compute ===
micro_f1 = f1_score(all_labels, all_preds, average='micro')
macro_f1 = f1_score(all_labels, all_preds, average='macro')
precision = precision_score(all_labels, all_preds, average='micro')
recall = recall_score(all_labels, all_preds, average='micro')

logger.info(f"Micro F1: {micro_f1:.4f}")
logger.info(f"Macro F1: {macro_f1:.4f}")
logger.info(f"Precision: {precision:.4f}")
logger.info(f"Recall: {recall:.4f}")
