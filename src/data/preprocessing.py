import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class CKDPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def handle_missing_values(self, df):
        print("\n[PREPROCESSING]")
        print("Missing values before handling:\n", df.isnull().sum())

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        print("Missing values after handling:\n", df.isnull().sum())
        return df

    def encode_categorical(self, df):
        print("\n[PREPROCESSING]")
        print("Encoding categorical columns...")

        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")

        return df

    def scale_features(self, X):
        X_scaled = self.scaler.fit_transform(X)

        print("\n[PREPROCESSING]")
        print("Features scaled successfully")
        print("Scaled feature sample:\n", X_scaled[:5])

        return X_scaled

    def preprocess(self, X, y=None):
        print("\n[PREPROCESSING] Starting preprocessing pipeline...")

        X = self.handle_missing_values(X)
        X = self.encode_categorical(X)
        X_scaled = self.scale_features(X)

        if y is not None:
            y = LabelEncoder().fit_transform(y)
            print("Target labels encoded")
            print("Final target shape:", y.shape)
        else:
            print("No target labels provided (inference mode)")

        print("Final feature shape:", X_scaled.shape)

        return X_scaled, y
if __name__ == "__main__":
    print("\n[RUNNING preprocessing.py AS MAIN MODULE]")

    from src.data.load_data import CKDDataLoader

    # Load data
    loader = CKDDataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target(df, target_column="Diagnosis")

    # Preprocess
    preprocessor = CKDPreprocessor()
    X_processed, y_processed = preprocessor.preprocess(X, y)

    print("\n[PREPROCESSING COMPLETED]")
    print("Processed feature shape:", X_processed.shape)
    print("Processed target shape:", y_processed.shape)
