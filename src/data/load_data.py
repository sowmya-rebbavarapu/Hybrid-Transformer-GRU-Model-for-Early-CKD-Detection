import pandas as pd
from pathlib import Path

class CKDDataLoader:
    """
    Data Loader for CKD Dataset
    """

    def __init__(self, data_path="data/raw/ckd_dataset.csv"):
        self.data_path = Path(data_path)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError("CKD dataset not found")

        df = pd.read_csv(self.data_path)

        print("\n[DATA LOADER]")
        print("Dataset loaded successfully")
        print("Dataset shape:", df.shape)
        print("Columns:", list(df.columns))
        print("Sample records:\n", df.head())

        return df

    def split_features_target(self, df, target_column="Diagnosis"):
        print("\n[DATA LOADER] Removing leaky features for EARLY CKD prediction")

        # 🔴 Leaky features (post-diagnosis or diagnostic indicators)
        LEAKY_FEATURES = [
            "GFR",
            "SerumCreatinine",
            "ProteinInUrine",
            "ACR",
            "RecommendedVisitsPerMonth",
            "Adherence"
        ]

        for col in LEAKY_FEATURES:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"[DATA LOADER] Dropped leaky feature: {col}")

        # Split features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        print("\n[DATA LOADER]")
        print("Feature matrix shape:", X.shape)
        print("Target vector shape:", y.shape)
        print("Target distribution:\n", y.value_counts())

        print("\n[DATA LOADER] Sample feature rows after leakage removal:")
        print(X.head())

        return X, y


if __name__ == "__main__":
    print("\n[RUNNING load_data.py AS MAIN MODULE]")

    loader = CKDDataLoader()
    df = loader.load_data()

    X, y = loader.split_features_target(df)
