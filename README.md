Automated Hybrid Transformer-GRU Model for Early CKD Detection

##  Description

This project implements a hybrid deep learning model combining a Transformer encoder and a GRU network to detect Chronic Kidney Disease (CKD) at an early stage using clinical data.

The system includes data preprocessing, automated feature selection, model training, evaluation, and a simple UI for predictions. It is designed to handle high-dimensional healthcare datasets and achieve high prediction accuracy.

---

## Features

* Hybrid Transformer + GRU architecture
* Early CKD detection using clinical data
* Automated feature selection
* Handles class imbalance
* Modular and scalable pipeline
* UI for predictions
* High accuracy (~99%)

---

##  Model Workflow

Data → Preprocessing → Feature Selection → Transformer → GRU → Classification → Prediction

---

## Dataset

### Training Dataset

* Path: `data/raw/ckd_dataset.csv`
* Full dataset used for training (~83K records)

### Test/Input Dataset

* Path: `ckd_input.csv`
* Used for testing/predictions

---

## ⚙️ Installation

```bash
git clone https://github.com/sowmya-rebbavarapu/Hybrid-Transformer-GRU-Model-for-Early-CKD-Detection.git
cd Hybrid-Transformer-GRU-Model-for-Early-CKD-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```
### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Evaluate the Model

```bash
python -m src.evaluation.evaluate
```

### Run UI Application

```bash
python app.py
```

---

## 📈 Results

* Accuracy: 99.31%
* Precision: 100%
* Recall: 99.25%
* F1 Score: 99.62%
* AUC: ~1.0

---

## 📂 Project Structure

```bash
├── data/
│   └── raw/
│       └── ckd_dataset.csv
├── evaluation_results/
├── models/
├── reports/
│   ├── evaluation/
│   └── figures/
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── eda/
│   │   └── eda.py
│   ├── evaluation/
│   │   └── evaluate.py
│   ├── features/
│   │   └── features_selection.py
│   └── train/
│       └── train.py
├── templates/
├── app.py
├── ckd_input.csv
├── requirements.txt
└── README.md
```

---

## Technologies Used

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn

---

## Key Components

### Data Processing

* Missing value handling
* Categorical encoding
* Feature scaling

### Feature Selection

* Random Forest-based importance ranking
* Top 30 features selected

### Model

* Transformer for feature relationships
* GRU for sequential learning

---
