from flask import Flask, render_template, request
import torch
import pandas as pd
import numpy as np

from src.models.model import TransformerGRUModel
from src.data.preprocessing import CKDPreprocessor
from src.features.features_selection import FeatureSelector

app = Flask(__name__)

# ================= LOAD MODEL =================
DEVICE = torch.device("cpu")

model = TransformerGRUModel(input_dim=30)
model.load_state_dict(torch.load("models/transformer_gru_30features.pth", map_location=DEVICE))
model.eval()

# ================= LOAD PREPROCESSOR =================
preprocessor = CKDPreprocessor()
selector = FeatureSelector(k=30)

# ================= ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    table = None

    if request.method == "POST":
        try:
            file = request.files['file']

            # -------------------------------
            # 1. Read file
            # -------------------------------
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

            # -------------------------------
            # 2. Apply SAME preprocessing
            # -------------------------------
            X_processed, _ = preprocessor.preprocess(df, None)

            # -------------------------------
            # 3. Feature Selection / Integrity Check
            # -------------------------------
            if X_processed.shape[1] == 30:
                X_selected = X_processed
            else:
                dummy_y = np.zeros(len(X_processed))
                if len(dummy_y) > 1:
                    dummy_y[0] = 1 # Avoid f_classif single-class error
                X_selected = selector.fit_transform(X_processed, dummy_y)

            # -------------------------------
            # 4. Convert to tensor
            # -------------------------------
            X_tensor = torch.tensor(X_selected, dtype=torch.float32)

            # -------------------------------
            # 5. Prediction
            # -------------------------------
            with torch.no_grad():
                logits = model(X_tensor)
                probs = torch.sigmoid(logits).numpy().flatten()

            # -------------------------------
            # 6. Extract results
            # -------------------------------
            def get_risk(p):
                if p >= 0.8:
                    return "High Risk"
                elif p >= 0.6:
                    return "Moderate Risk"
                else:
                    return "Low Risk"

            # Create a clean results DataFrame to show only strictly relevant info
            results_df = pd.DataFrame({
                "Patient ID": range(1, len(probs) + 1),
                "Prediction": ["Kidney Disease" if p >= 0.5 else "No Kidney Disease" for p in probs],
                "Risk Level": [get_risk(p) for p in probs],
                "Confidence": [f"{p*100:.1f}%" for p in probs]
            })

            # -------------------------------
            # 7. Convert to HTML table
            # -------------------------------
            table = results_df.to_html(classes='glass-table', index=False)

        except Exception as e:
            table = f"<p style='color:red;'>Error: {str(e)}</p>"

    return render_template("index.html", table=table)


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)