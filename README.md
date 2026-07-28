# ❤️ Khushi's Heart Care AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Model](https://img.shields.io/badge/Model-KNN%20Classifier-green)
![Accuracy](https://img.shields.io/badge/Accuracy-88.04%25-brightgreen)

A Data Science web app that predicts **cardiovascular (heart disease) risk** from patient health indicators using a **K-Nearest Neighbors (KNN) Classifier**.

**Risk Classes:** Low Risk 🟢 · High Risk 🔴

🔗 **Live Demo:** _(add your Streamlit deployment link here once deployed)_

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Live Features](#-live-features)
- [Machine Learning Model](#-machine-learning-model)
- [How to Use](#-how-to-use)
- [Tech Stack](#️-tech-stack)
- [Setup](#-setup)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Developer](#-developer)

---

## 📖 About the Project

- Predicts the risk of heart disease from 11 clinical patient features using classical ML (no deep learning).
- Trained and evaluated a KNN Classifier, tuned across K values for the best validation accuracy.
- Packaged as a full interactive Streamlit app — risk predictor, EDA explorer, and model benchmark dashboard, not just a notebook.

---

## 🔄 Project Workflow

1. **Input** — patient details entered via form (age, sex, chest pain type, blood pressure, cholesterol, etc.)
2. **Preprocessing** — categorical encoding to match the training feature columns
3. **Scaling** — feature values transformed using a fitted `StandardScaler`
4. **Classification** — KNN (K=5) predicts Heart Disease risk (0 = Healthy, 1 = At Risk)
5. **Output** — risk probability, patient feature radar, what-if simulation, and downloadable reports

---

## 📊 Dataset

- **File:** `heart.csv`
- **Size:** 918 patient records
- **Target values (2 classes):**

  | Label | Result |
  |---|---|
  | 0 | Healthy |
  | 1 | Heart Disease Risk |

- **Features used:** `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`

---

## ⚡ Live Features

- Interactive heart disease risk predictor form
- 🕸️ Patient feature radar & population comparison map
- 🎛️ What-If Scenario Simulator (real-time risk recalculation as features change)
- 💡 Clinical insights with downloadable HTML (printable) & CSV prediction reports
- 📊 Exploratory Data Analysis (EDA) explorer with attribute filters
- 📈 Model benchmark dashboard — confusion matrix, ROC curve, feature importance
- 🎛️ Interactive KNN hyperparameter (K-value) tuner
- 🕓 Session-based evaluation history in sidebar

---

## 🧠 Machine Learning Model

| K (Neighbors) | Validation Accuracy |
|---|---|
| 3 | 86.4% |
| **5** ✅ | **88.04%** |
| 7 | 87.5% |
| 9 | 86.9% |

**Final model:** KNN Classifier (K=5) — 88.04% validation accuracy, 0.912 ROC-AUC.

---

## 🎮 How to Use

1. Run the app → opens in browser
2. **Heart Disease Risk Predictor tab** — fill in patient details → click Predict
3. **Radar & Cohort Map** — view patient feature profile vs. dataset population
4. **What-If Simulator** — tweak features to see real-time risk changes
5. **Insights & Report tab** — download the HTML or CSV prediction report
6. **EDA tab** — explore and filter the underlying patient dataset
7. **Model Benchmarks tab** — view confusion matrix, ROC curve, and feature importance

---

## 🛠️ Tech Stack

- Streamlit
- scikit-learn
- Pandas & NumPy
- Plotly
- Joblib

---

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit application
├── heart.csv             # Training dataset
├── KNN_heart.pkl          # Trained KNN classifier
├── scaler.pkl             # Fitted StandardScaler
├── columns.pkl            # Feature column order used by the model
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🔮 Future Improvements

- Try ensemble models (Random Forest, XGBoost) for higher accuracy
- Add SHAP/LIME-based explainability for individual predictions
- Persistent history (database-backed, not session-only)
- Deploy on Streamlit Cloud / Docker
- Add user authentication for multi-patient tracking

---

## 👩‍💻 Developer

**Khushi Singh** — Machine Learning & AI Engineer
