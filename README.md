# 🫀 Heart Disease Prediction (CRISP-DM Pipeline)

A machine learning pipeline to predict the presence of **heart disease** using the **UCI Heart Disease dataset**.  
Includes: **data prep, exploratory analysis, feature selection, model training (Logistic Regression, Random Forest, XGBoost), calibration, evaluation, fairness slices, and deployment with FastAPI**.

📂 **Dataset:** [UCI Heart Disease Dataset (Kaggle link)](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)  
📰 **Medium Article:** (add link if you publish a blog)

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Workflow](#-workflow)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Explainability](#-explainability)
- [Results & Insights](#-results--insights)
- [Deployment](#-deployment)
- [Reproducibility](#-reproducibility)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔎 Overview

This project builds a **binary classifier** to predict whether a patient is likely to have heart disease.  
It demonstrates a complete **CRISP-DM pipeline**:

✨ Highlights:
- **Data prep** with missing value handling & physiologic validity checks  
- **Feature importance** via Mutual Information & correlations  
- **Baseline model**: Logistic Regression  
- **Advanced models**: Random Forest, XGBoost  
- **Fairness evaluation** by age bands and sex  
- **Deployment-ready API** via FastAPI

---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset (Kaggle mirror)](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)  
- **Samples**: 303 patients  
- **Features**: 13 clinical variables (age, sex, cholesterol, thalach, etc.)  
- **Target**: Presence of heart disease (1 = yes, 0 = no)  

📌 Example rows:

| age | sex | cp | trestbps | chol | thalach | exang | oldpeak | slope | ca | thal | target |
|-----|-----|----|----------|------|---------|-------|---------|-------|----|------|--------|
| 63  | 1   | 3  | 145      | 233  | 150     | 0     | 2.3     | 0     | 0  | 1    | 1      |
| 37  | 1   | 2  | 130      | 250  | 187     | 0     | 3.5     | 0     | 0  | 2    | 1      |

---

## 📂 Project Structure
heart-disease-ml/
│
├── heart_phase2_EDA.ipynb # Phase 2: Exploratory Data Analysis
├── heart_phase3_train.ipynb # Phase 3: Data Preparation & Training
├── heart_phase4_modeling.ipynb # Phase 4: Modeling & Hyperparameter Tuning
├── heart_phase5_eval.ipynb # Phase 5: Evaluation & Model Card
│
├── serve/ # FastAPI inference service
│ ├── app.py
│ └── model/ # staged model artifacts (ignored in git)
│
├── requirements.txt
├── .gitignore
└── README.md

🛠️ Workflow

Data Understanding: EDA, missing values, distributions

Data Preparation: Cleaning, feature typing, binarization

Modeling: Logistic Regression, RandomForest, XGBoost

Evaluation: ROC/PR, calibration, fairness slices

Deployment: FastAPI + Uvicorn

🐕 Training

Minimal Logistic Regression example:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=500, class_weight="balanced")
clf.fit(X_train, y_train)

print("Test AUROC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

📈 Evaluation

Best model: RandomForest (full feature set)

Test AUROC: ~0.88

PR AUC: ~0.86

Brier Score: ~0.16

📊 Charts:

ROC Curve


PR Curve


Calibration Plot


Decision Curve

🏆 Results & Insights

Operating at Sensitivity ≥ 0.90:

Sensitivity: 0.91

Specificity: 0.72

PPV: 0.77

NPV: 0.89

💡 Key insight: Model achieves high recall (few false negatives), making it more suitable as a screening tool rather than final diagnosis.

🚀 Deployment

FastAPI endpoints:

GET /health → returns model info

POST /predict → binary prediction (0/1)

POST /predict_proba → probability + prediction

🙏 Acknowledgments

UCI Heart Disease Dataset

Scikit-learn, FastAPI, Matplotlib

Google Colab for experiments
