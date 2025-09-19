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

