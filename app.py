from flask import Flask, render_template, request
import torch
import numpy as np
import joblib

from src.models.model import TransformerGRUModel

app = Flask(__name__)

# ================= LOAD MODEL =================
DEVICE = torch.device("cpu")

model = TransformerGRUModel(input_dim=30)
model.load_state_dict(torch.load("models/transformer_gru_30features.pth", map_location=DEVICE))
model.eval()

# ✅ LOAD SAVED PREPROCESSOR & SELECTOR
preprocessor = joblib.load("models/preprocessor.pkl")
selector = joblib.load("models/selector.pkl")

# ================= FEATURE ORDER =================
FEATURE_ORDER = [
    "Gender","SocioeconomicStatus","EducationLevel","BMI","Smoking",
    "PhysicalActivity","DietQuality","SleepQuality",
    "FamilyHistoryKidneyDisease","FamilyHistoryHypertension","UrinaryTractInfections",
    "SystolicBP","DiastolicBP","FastingBloodSugar","HbA1c","BUNLevels",
    "SerumElectrolytesSodium","SerumElectrolytesPotassium","HemoglobinLevels",
    "CholesterolTotal","CholesterolHDL","Diuretics","AntidiabeticMedications",
    "Edema","NauseaVomiting","MuscleCramps","Itching",
    "QualityOfLifeScore","OccupationalExposureChemicals","WaterQuality"
]

# ================= ROUTES =================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    risk = None

    if request.method == "POST":
        try:
            # ✅ Get input in correct order
            input_values = [float(request.form[feature]) for feature in FEATURE_ORDER]
            input_array = np.array(input_values).reshape(1, -1)

            # ✅ APPLY SAME PIPELINE
            input_scaled = preprocessor.transform(input_array)
            input_selected = selector.transform(input_scaled)

            # Convert to tensor
            input_tensor = torch.tensor(input_selected, dtype=torch.float32)

            # Prediction
            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.sigmoid(logits).item()

            # ✅ Slightly safer threshold
            pred = 1 if prob >= 0.6 else 0

            # Risk levels
            if prob >= 0.8:
                risk = "High Risk"
            elif prob >= 0.6:
                risk = "Moderate Risk"
            else:
                risk = "Low Risk"

            prediction = "CKD Detected" if pred == 1 else "No CKD"
            probability = round(prob, 4)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability,
                           risk=risk)


if __name__ == "__main__":
    app.run(debug=True)