# 📚 Technical Documentation
## Multi-Disease AI Healthcare Risk Prediction Platform

**Last Updated:** January 24, 2026

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Collection & Documentation](#3-data-collection--documentation)
4. [Clinical Feature Mapping](#4-clinical-feature-mapping)

---

## 1. Project Structure

```
PROJECT/
├── 📁 api/                    # FastAPI backend
│   ├── __init__.py
│   └── main.py
│
├── 📁 configs/                # Configuration files
│
├── 📁 data/                   # Processed data storage
│   ├── embeddings/           # NLP embeddings
│   ├── processed/            # Clean datasets
│   └── raw/                  # Original data backups
│
├── 📁 datasets/              # Main datasets (FINAL)
│   ├── heart_disease.csv           # 70,000 samples
│   ├── diabetes_health_indicators.csv  # 100,000 samples
│   ├── Chronic_Kidney_Dsease_data.csv  # 1,659 samples
│   └── liver_disease_30k.csv       # 30,691 samples
│
├── 📁 datasets_evaluated/    # Dataset review records
│   └── rejected/             # Rejected datasets with reasons
│
├── 📁 docs/                  # Documentation
│   ├── thesis/              # Thesis documents
│   └── TECHNICAL_DOCS.md    # This file
│
├── 📁 logs/                  # Training & application logs
│
├── 📁 models/                # Saved models
│   ├── dl/                  # Deep learning models
│   └── ml/                  # Machine learning models
│
├── 📁 notebooks/             # Jupyter notebooks
│   └── eda/                 # Exploratory data analysis
│
├── 📁 src/                   # Source code
│   ├── data/                # Data handling modules
│   ├── explainability/      # XAI components (SHAP, LLM)
│   ├── models/              # Model implementations
│   ├── nlp/                 # NLP components
│   └── utils/               # Utility functions
│
├── 📁 tests/                 # Unit & integration tests
│
├── 📁 webapp/                # Streamlit web application
│   └── app.py
│
├── README.md                 # Project overview
├── PROGRESS_TRACKER.md       # Progress tracking
├── requirements.txt          # Python dependencies
└── activate_env.bat          # Environment activation script
```

---

## 2. Environment Setup

### 2.1 Python Environment

- **Location:** `C:\dev\venvs\btp_env` (outside OneDrive to prevent sync issues)
- **Python Version:** 3.11.9
- **Activation:** Run `activate_env.bat` from project root

### 2.2 Installed Packages

| Category | Packages | Versions |
|----------|----------|----------|
| **Core ML** | scikit-learn, xgboost, lightgbm | 1.8.0, 3.1.3, 4.7.0 |
| **Deep Learning** | torch, transformers | 2.10.0+cpu, 4.57.6 |
| **NLP** | sentence-transformers, tokenizers | 4.1.0, 0.25.0 |
| **Explainability** | shap, lime | 0.50.0, 0.2.0.1 |
| **API/Web** | fastapi, uvicorn, streamlit | 0.128.0, 0.35.0, 1.53.1 |
| **Data** | pandas, numpy | 2.3.0, 2.3.1 |
| **Visualization** | matplotlib, seaborn, plotly | 3.10.1, 0.13.2, 6.1.0 |

### 2.3 Quick Start

```powershell
# Navigate to project
cd "C:\Users\DELL\OneDrive\ドキュメント\BTP\PROJECT"

# Activate environment
.\activate_env.bat

# Verify installation
python -c "import torch, transformers, sklearn; print('All packages working!')"
```

---

## 3. Data Collection & Documentation

### 3.1 Project Data Requirements

**Primary Objective:** Build a multi-disease risk prediction system where users provide:
- Clinical lab values (blood tests, vitals)
- Lifestyle information (text input processed by NLP)
- Checkboxes (medications, family history)

**Output:**
- Risk predictions for Heart Disease, Diabetes, Kidney Disease, Liver Disease
- Explainable AI insights (SHAP + LLM-generated explanations)

### 3.2 Final Dataset Summary

| Disease | File | Samples | Features | Target | Source |
|---------|------|---------|----------|--------|--------|
| Heart | heart_disease.csv | 70,000 | 13 | cardio | Kaggle CC0 |
| Diabetes | diabetes_health_indicators.csv | 100,000 | 31 | diagnosed_diabetes | Kaggle CC0 |
| Kidney | Chronic_Kidney_Dsease_data.csv | 1,659 | 54 | Diagnosis | Kaggle CC BY 4.0 |
| Liver | liver_disease_30k.csv | 30,691 | 11 | Result | Kaggle CC0 |
| **TOTAL** | | **202,350** | | | |

### 3.3 Dataset Details

#### Heart Disease (70,000 samples)
- **Source:** Kaggle - Cardiovascular Disease Dataset
- **Key Features:** age, gender, height, weight, BP (systolic/diastolic), cholesterol, glucose, smoke, alcohol, activity
- **Target:** `cardio` (0=no disease, 1=disease)
- **Selection Reason:** Large sample size, clinical lab values, lifestyle factors

#### Diabetes Health Indicators (100,000 samples)
- **Source:** Kaggle - Diabetes Health Indicators Dataset (UPGRADED from Pima 768)
- **Key Features:** HbA1c (glycated hemoglobin), glucose, insulin, BMI, blood pressure, cholesterol, lifestyle factors
- **Target:** `diagnosed_diabetes`
- **Selection Reason:** 130x larger than original, includes HbA1c (gold standard for diabetes diagnosis)

#### Chronic Kidney Disease (1,659 samples)
- **Source:** Kaggle - CKD Dataset
- **Key Features:** 54 comprehensive features including creatinine, BUN, albumin, hemoglobin, potassium, sodium
- **Target:** `Diagnosis`
- **Selection Reason:** Most comprehensive CKD dataset available, all standard lab values

#### Liver Disease (30,691 samples)
- **Source:** Kaggle - Liver Disease Patient Dataset (UPGRADED from ILPD 583)
- **Key Features:** Bilirubin (total/direct), ALT, AST, ALP, albumin, total protein, A/G ratio
- **Target:** `Result` (1=disease, 2=no disease)
- **Selection Reason:** 52x larger than original, standard liver function panel

### 3.4 Rejected Datasets

| Dataset | Samples | Rejection Reason |
|---------|---------|------------------|
| Pima Indians Diabetes | 768 | Too small for DL, replaced with 100K |
| Indian Liver Patient | 583 | Too small, replaced with 30K |
| NAFLD Survival Dataset | 17,549 | Survival analysis format (time-to-event), not risk prediction |
| Various UCI Datasets | <1,000 | Too small for research-grade analysis |

---

## 4. Clinical Feature Mapping

### 4.1 Input Architecture

```
Patient Input:
├── 📊 Structured Input (Lab Report Values)
│   └── Numerical fields from medical test reports
│       - Blood tests (glucose, cholesterol, enzymes)
│       - Vitals (BP, heart rate, BMI)
│       - Demographics (age, sex)
│
└── 💬 Unstructured Input (Patient History - NLP)
    └── Free text entered by patient
        - Lifestyle: "I smoke 10 cigarettes daily"
        - Family history: "My father had diabetes"
        - Habits: "I drink alcohol occasionally"
```

### 4.2 Feature Categories by Disease

#### Heart Disease Features
| Type | Features |
|------|----------|
| **Lab Values** | BP (systolic/diastolic), cholesterol, glucose |
| **Physical** | Height, weight, BMI (derived) |
| **Demographics** | Age, gender |
| **NLP Extracted** | Smoking status, alcohol, physical activity |

#### Diabetes Features
| Type | Features |
|------|----------|
| **Lab Values** | HbA1c, fasting glucose, insulin, cholesterol |
| **Physical** | BMI, blood pressure |
| **Demographics** | Age, gender |
| **NLP Extracted** | Family history, diet, exercise |

#### Kidney Disease Features
| Type | Features |
|------|----------|
| **Lab Values** | Creatinine, BUN, eGFR, albumin, hemoglobin, electrolytes |
| **Physical** | Blood pressure, specific gravity |
| **Demographics** | Age |
| **NLP Extracted** | Diabetes history, hypertension, medication |

#### Liver Disease Features
| Type | Features |
|------|----------|
| **Lab Values** | Bilirubin, ALT, AST, ALP, albumin, total protein |
| **Physical** | A/G ratio |
| **Demographics** | Age, gender |
| **NLP Extracted** | Alcohol consumption, medication use |

---

## 5. Model Architecture (Planned)

### 5.1 Multi-Task Learning Pipeline

```
Input Layer
    ├── Structured Features → Feature Engineering → Normalization
    ├── Clinical Text → NLP (BioMistral) → Embeddings
    └── Checkboxes → One-hot Encoding

Feature Fusion Layer
    └── Concatenate all feature types

Multi-Task Prediction Head
    ├── Heart Disease Risk
    ├── Diabetes Risk
    ├── Kidney Disease Risk
    └── Liver Disease Risk

Explainability Layer
    ├── SHAP Values (feature importance)
    └── LLM Explanation (natural language)
```

### 5.2 Target Metrics
- AUC-ROC > 0.85 for all diseases
- Sensitivity > 0.80 (clinical requirement)
- Specificity > 0.75

---

*Document maintained by BTP Project Team*
