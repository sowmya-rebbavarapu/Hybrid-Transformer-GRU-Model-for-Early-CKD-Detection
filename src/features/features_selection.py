import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelector:
    """
    Automated Feature Selection for CKD Dataset
    """

    def __init__(self, k=20):
        self.k = k
        self.selector = SelectKBest(score_func=f_classif, k=self.k)

    def fit_transform(self, X, y):
        print("\n[FEATURE SELECTION]")
        print("Input feature shape:", X.shape)

        X_selected = self.selector.fit_transform(X, y)

        print("Top-K features selected:", self.k)
        print("Output feature shape:", X_selected.shape)

        return X_selected

    def get_selected_indices(self):
        indices = self.selector.get_support(indices=True)
        print("Selected feature indices:", indices)
        return indices


# ================= RUN AS SCRIPT =================
if __name__ == "__main__":
    print("\n[RUNNING features_selection.py AS MAIN MODULE]")

    # Import previous pipeline steps
    from src.data.load_data import CKDDataLoader
    from src.data.preprocessing import CKDPreprocessor

    # 1. Load data
    loader = CKDDataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target(df, target_column="Diagnosis")

    # 2. Preprocess
    preprocessor = CKDPreprocessor()
    X_processed, y_processed = preprocessor.preprocess(X, y)

    # 3. Feature selection
    selector = FeatureSelector(k=20)
    X_selected = selector.fit_transform(X_processed, y_processed)

    selected_indices = selector.get_selected_indices()

    print("\n[FEATURE SELECTION COMPLETED]")
    print("Final selected feature matrix shape:", X_selected.shape)