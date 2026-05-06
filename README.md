# 💳 Credit Card Fraud Detection System

> A machine learning pipeline that detects fraudulent credit card transactions in real time — trained on 284,807 transactions with 92% precision and 80% recall.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189ADA?style=flat&logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Models & Results](#-models--results)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🔍 Overview

Credit card fraud costs the global economy **over $32 billion annually**. This project builds an end-to-end machine learning system to automatically detect fraudulent transactions — addressing the core real-world challenge of **severe class imbalance** (only 0.172% of transactions are fraud).

Three distinct model architectures were developed, evaluated, and compared:

| Model | Approach | Best For |
|---|---|---|
| **Random Forest** | Supervised ensemble | High precision, low false alarms |
| **XGBoost** | Gradient boosting | Speed + strong generalisation |
| **Autoencoder** | Unsupervised deep learning | Anomaly detection without labels |

The **Random Forest** model was selected for deployment based on its superior precision-recall balance.

---

## 🚀 Live Demo

The model is deployed as an interactive web application built with Streamlit:

```
Transaction Amount: 99.01
V14: -9.01
V7:  -0.10

→ Fraud Probability: 0.33
→ Result: ✅ Legitimate Transaction
```

> Enter any transaction values and get an instant fraud probability score.

---

## ✨ Features

- **End-to-end ML pipeline** — from raw data to deployed web app
- **SMOTE oversampling** — handles severe class imbalance without data leakage
- **Three model families** — supervised, boosting, and deep learning approaches
- **Hybrid scoring** — combines XGBoost probability + Autoencoder reconstruction error
- **Feature importance analysis** — identifies V14 and V7 as strongest fraud predictors
- **Live web interface** — real-time predictions via Streamlit app
- **Model serialisation** — saved with `joblib` for production deployment

---

## 📊 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total Transactions | 284,807 |
| Fraudulent Transactions | 492 (0.172%) |
| Legitimate Transactions | 284,315 (99.83%) |
| Features | 31 (V1–V28 + Time, Amount, Class) |
| Missing Values | None |

> **Note:** Features V1–V28 are PCA-transformed to protect cardholder privacy. The original variables cannot be recovered.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── FRAUD_DETECTION.ipynb       # Main notebook — full pipeline
│
├── models/
│   └── fraud_model.pkl         # Trained Random Forest (joblib)
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── data/
│   └── creditcard.csv          # Dataset (download from Kaggle)
│
├── outputs/
│   ├── confusion_matrix.png    # Confusion matrix visualisation
│   ├── roc_curve.png           # ROC-AUC curve
│   └── feature_importance.png  # XGBoost feature importance chart
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 📈 Models & Results

### Confusion Matrix — Random Forest (Deployed Model)

|  | Predicted: Legitimate | Predicted: Fraud |
|---|---|---|
| **Actual: Legitimate** | 56,857 ✅ True Negatives | 7 ❌ False Positives |
| **Actual: Fraud** | 20 ⚠️ False Negatives | 78 ✅ True Positives |

> ⚠️ **False Negatives are the most critical error** — they represent undetected fraud causing direct financial loss.

---

### Model Comparison

| Model | Precision | Recall | F1-Score | False Positives | False Negatives |
|---|---|---|---|---|---|
| **Random Forest** ✅ | **92%** | 80% | **0.86** | **7** | 20 |
| XGBoost | 87% | 80% | 0.83 | 12 | 20 |
| Autoencoder | 15% | **85%** | 0.25 | High | 15 |
| Hybrid (XGB + AE) | 87% | 80% | 0.83 | 12 | 20 |

**Why Random Forest was chosen for deployment:**
- Highest precision (92%) → fewest false alarms
- Same recall as XGBoost (80%) but 5 fewer false positives
- Most suitable for a live banking environment where false alarms damage customer trust

---

### XGBoost Cross-Validation

```
5-Fold CV Score: 0.9994  ✅
```

---

### Top Predictive Features (XGBoost Importance)

```
Rank  Feature   Importance
  1     V14      ████████████████████  0.38
  2      V7      ████████████          0.22
  3      V4      ██████                0.12
  4     V10      █████                 0.09
  5     V12      ████                  0.07
```

> V14 was identified as the strongest fraud predictor in both exploratory analysis and formal importance scoring.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

---

## 🖥️ Usage

### Run the Notebook

```bash
jupyter notebook FRAUD_DETECTION.ipynb
```

### Launch the Web App

```bash
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

### Use the Trained Model Directly

```python
import joblib
import numpy as np

# Load the model
model = joblib.load("models/fraud_model.pkl")

# Example transaction [Amount, V14, V7, ...]
transaction = np.array([[99.01, -9.01, -0.10, ...]])  # 30 features total

# Predict
probability = model.predict_proba(transaction)[0][1]
prediction  = model.predict(transaction)[0]

print(f"Fraud Probability: {probability:.2f}")
print("FRAUD" if prediction == 1 else "Legitimate Transaction")
```

---

## 🔬 How It Works

```
Raw Data (284,807 transactions)
        │
        ▼
Exploratory Data Analysis
  • Class imbalance confirmed: 0.172% fraud
  • V14, V17 show strongest class separation
        │
        ▼
Preprocessing
  • Stratified train/test split (80/20)
  • SMOTE applied to training set only
  • StandardScaler normalisation
        │
        ▼
Model Training
  ┌─────────────────────────────────────┐
  │  Random Forest  │  XGBoost  │  AE  │
  └─────────────────────────────────────┘
        │
        ▼
Evaluation
  • Confusion Matrix
  • Precision / Recall / F1
  • ROC-AUC Curve
  • 5-Fold Cross-Validation
        │
        ▼
Best Model Selected → Random Forest
        │
        ▼
Deployed via Streamlit Web App
```

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **ML Models** | Scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow / Keras |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Web App** | Streamlit |
| **Model Serialisation** | joblib |
| **Environment** | Jupyter Notebook |

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
imbalanced-learn
streamlit
joblib
jupyter
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎓 About This Project

This project was developed as a final project during training at **[Torbita Computer Schools](https://www.torbitacomputerschools.com)**.

It demonstrates practical application of:
- Real-world class imbalance handling
- Multiple ML model development and comparison
- Model deployment as a web application
- Proper evaluation using domain-appropriate metrics

---

## 👤 Author

**Odeh Godwin Adakole**
- 🔗 LinkedIn: [linkedin.com/in/yourprofile]((https://www.linkedin.com/in/godwin-odeh-48000a247/))
- 🐙 GitHub: [github.com/yourusername](https://github.com/odehgodwin196)
- 📧 Email: odehgodwin196@gmail.com

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

## 🌟 Show Your Support

If you found this project useful or interesting, please consider giving it a ⭐ on GitHub — it helps others discover it!

---

*Built with ❤️ at Torbita Computer Schools*
