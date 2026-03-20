import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from src.data.load_data import CKDDataLoader
from src.data.preprocessing import CKDPreprocessor
from src.features.features_selection import FeatureSelector
from src.models.model import TransformerGRUModel

# ================= CONFIG =================
TOP_K_FEATURES = 30
MODEL_PATH = "models/transformer_gru_30features.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "evaluation_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= LOAD DATA =================
print("\n[EVALUATION] Loading dataset...")
loader = CKDDataLoader()
df = loader.load_data()
X, y = loader.split_features_target(df)

# ================= PREPROCESS =================
preprocessor = CKDPreprocessor()
X_proc, y_proc = preprocessor.preprocess(X, y)

# ================= FEATURE SELECTION =================
selector = FeatureSelector(k=TOP_K_FEATURES)
X_sel = selector.fit_transform(X_proc, y_proc)

# ================= SPLIT =================
_, X_test, _, y_test = train_test_split(
    X_sel,
    y_proc,
    test_size=0.2,
    stratify=y_proc,
    random_state=42
)

# ================= TENSORS =================
X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# ================= LOAD MODEL =================
print("\n[MODEL LOADING]")
model = TransformerGRUModel(input_dim=X_sel.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[MODEL LOADED SUCCESSFULLY]")

# ================= EVALUATION =================
print("\n[EVALUATION] Running inference...")
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

y_true = y_test.cpu().numpy()
y_pred = preds.cpu().numpy()
y_prob = probs.cpu().numpy()

# ================= METRICS =================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n[FINAL RESULTS]")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save metrics table
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [acc, prec, rec, f1]
})
metrics_df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

print("[CONFUSION MATRIX SAVED]")

# ================= CLASSIFICATION REPORT =================
report = classification_report(y_true, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(SAVE_DIR, "classification_report.csv"))

print("\n[CLASSIFICATION REPORT]")
print(classification_report(y_true, y_pred))

# ================= ROC CURVE =================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
plt.close()

print("[ROC CURVE SAVED]")

# ================= SAVE METRICS IMAGE =================
plt.figure(figsize=(6, 3))
plt.axis("off")
table = plt.table(cellText=np.round(metrics_df["Value"].values.reshape(-1,1),4),
                  rowLabels=metrics_df["Metric"],
                  colLabels=["Value"],
                  loc="center")
table.scale(1,2)
plt.title("Model Performance Metrics")
plt.savefig(os.path.join(SAVE_DIR, "metrics_table.png"))
plt.close()

print("[METRICS TABLE IMAGE SAVED]")
