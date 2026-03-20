import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.data.load_data import CKDDataLoader
from src.data.preprocessing import CKDPreprocessor
from src.features.features_selection import FeatureSelector
from src.models.model import TransformerGRUModel

# ================= CONFIG =================
BATCH_SIZE = 64
EPOCHS = 3                      # Early stopping to avoid overfitting
LEARNING_RATE = 1e-3
TOP_K_FEATURES = 30
MODEL_PATH = "models/transformer_gru_30features.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD DATA =================
print("\n[TRAINING] Loading dataset...")
loader = CKDDataLoader()
df = loader.load_data()
X, y = loader.split_features_target(df)

# ================= PREPROCESS =================
print("\n[PREPROCESSING]")
preprocessor = CKDPreprocessor()
X_proc, y_proc = preprocessor.preprocess(X, y)

# ================= FEATURE SELECTION =================
print("\n[FEATURE SELECTION]")
selector = FeatureSelector(k=TOP_K_FEATURES)
X_sel = selector.fit_transform(X_proc, y_proc)

# ================= SPLIT =================
print("\n[DATA SPLIT]")
X_train, X_test, y_train, y_test = train_test_split(
    X_sel,
    y_proc,
    test_size=0.2,
    stratify=y_proc,
    random_state=42
)

print("Train class distribution:", np.bincount(y_train.astype(int)))
print("Test class distribution :", np.bincount(y_test.astype(int)))

# ================= CLASS WEIGHTS (IMBALANCE HANDLING) =================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

pos_weight = torch.tensor(
    class_weights[1] / class_weights[0],
    dtype=torch.float32
).to(DEVICE)

print(f"\n[IMBALANCE HANDLING]")
print(f"Computed pos_weight: {pos_weight.item():.4f}")

# ================= TENSORS =================
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= MODEL =================
print("\n[MODEL INIT]")
model = TransformerGRUModel(input_dim=X_sel.shape[1]).to(DEVICE)
print("[MODEL INIT COMPLETED]")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= TRAINING =================
print("\n[TRAINING] Started...")
for epoch in range(1, EPOCHS + 1):
    model.train()

    epoch_losses = []
    all_preds = []
    all_labels = []

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()

        epoch_losses.append(loss.item())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"Loss: {np.mean(epoch_losses):.4f} | "
        f"Acc: {accuracy_score(all_labels, all_preds):.4f} | "
        f"Prec: {precision_score(all_labels, all_preds):.4f} | "
        f"Rec: {recall_score(all_labels, all_preds):.4f} | "
        f"F1: {f1_score(all_labels, all_preds):.4f}"
    )

# ================= SAVE MODEL =================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n[MODEL SAVED] {MODEL_PATH}")
