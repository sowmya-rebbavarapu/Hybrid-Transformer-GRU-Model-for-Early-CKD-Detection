import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# --------------------------------
# Paths
# --------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data/raw/ckd_dataset.csv")
FIG_DIR = os.path.join(ROOT_DIR, "reports/figures")

os.makedirs(FIG_DIR, exist_ok=True)

# --------------------------------
# Load data
# --------------------------------
df = pd.read_csv(DATA_PATH)

# --------------------------------
# 1. Class Distribution
# --------------------------------
plt.figure()
df['Diagnosis'].value_counts().plot(kind='bar')
plt.title("Class Distribution of CKD")
plt.xlabel("Diagnosis (0: No CKD, 1: CKD)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "class_distribution.png"))
plt.close()

# --------------------------------
# 2. Age vs CKD
# --------------------------------
plt.figure()
sns.boxplot(x='Diagnosis', y='Age', data=df)
plt.title("Age Distribution by CKD Status")
plt.xlabel("Diagnosis")
plt.ylabel("Age")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "age_vs_ckd.png"))
plt.close()

# --------------------------------
# 3. Correlation Heatmap (Numeric Only)
# --------------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "correlation_heatmap.png"))
plt.close()

# --------------------------------
# 4. Feature Importance (Random Forest)
# --------------------------------
X = numeric_df.drop(columns=['Diagnosis'])
y = numeric_df['Diagnosis']

X_scaled = StandardScaler().fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
indices = importances.argsort()[::-1][:10]

plt.figure()
plt.bar(range(10), importances[indices])
plt.xticks(range(10), X.columns[indices], rotation=45, ha='right')
plt.title("Top 10 Important Features for CKD Prediction")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance.png"))
plt.close()

print("✅ EDA completed. Figures saved in reports/figures/")
