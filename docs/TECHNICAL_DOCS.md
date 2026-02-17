# 📚 Technical Documentation
## Multi-Disease AI Healthcare Risk Prediction Platform

**Last Updated:** February 11, 2026

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

## 5. Missing Value Handling Strategy (Week 4)

### 5.1 Missingness Analysis

**Overall:** 60.46% missing across 24 features

| Category | Missing % | Features | Count |
|----------|-----------|----------|-------|
| **Complete** | 0-5% | age, gender | 2 |
| **Low** | 5-25% | bmi, BP, cholesterol, glucose | 5 |
| **Medium** | 25-50% | HDL, LDL, triglycerides, HbA1c | 4 |
| **High** | 50-90% | Liver enzymes (ALT, AST, etc.) | 8 |
| **Very High** | 90%+ | Kidney markers (GFR, creatinine, etc.) | 5 |

### 5.2 Why Standard Imputation Fails

**Problem: Structural vs Random Missingness**

Our data has **structural missingness** (features never collected for certain diseases), NOT random missingness (values accidentally lost).

```
Example - Glucose (15% missing):
- Present in: Diabetes (100K), Heart (70K), Kidney (2K)
- Missing in: Liver (30K) ← NEVER measured for liver patients

If we use KNN/MICE imputation:
- Find liver patient: age=55, ALT=95, glucose=???
- Neighbors are mostly diabetes patients (they have glucose)
- Impute glucose=140 (from diabetics)

PROBLEM: We're INVENTING medical measurements!
- Liver patient was never tested for glucose
- Creates false correlation: "High ALT → glucose=140"
- Not medically safe
```

### 5.3 Our Approach: Two Versions for Comparison

We created TWO imputation approaches for ablation study:

#### Version A: Masked Approach (Primary for FT-Transformer)

**Strategy:**
```
Original: glucose = [120, 135, NaN, NaN]

Becomes TWO columns:
  glucose       = [120, 135,   0,   0]  ← 0 is placeholder
  glucose_mask  = [  1,   1,   0,   0]  ← 1=present, 0=missing
```

**Why:**
- NO AMBIGUITY: mask=0 explicitly means "ignore"
- FT-Transformer learns attention weights from masks
- Model ignores placeholder 0 values automatically
- Interpretable: "Used glucose (mask=1)" vs "Ignored ALT (mask=0)"

**Output:** `datasets/combined_unified_masked.csv`
- 46 feature columns + 4 targets + 1 metadata = 51 columns
- 202,723 rows

#### Version B: Sentinel Approach (Comparison & Tree Models)

**Strategy:**
```
Original: glucose = [120, 135, NaN, NaN]

Becomes:
  glucose = [120, 135, -999, -999]  ← -999 = not applicable
```

**Why -999 instead of -1:**
```
After StandardScaler:
  Real values: [-1.5, -0.5, 0, 0.5, 1.5, 2.5]  ← typical range
  Sentinel -1:  Overlaps with scaled low values! ← AMBIGUITY
  Sentinel -999: Clearly outside range ← NO AMBIGUITY
```

**Output:** `datasets/combined_unified_sentinel.csv`
- 24 feature columns + 4 targets + 1 metadata = 29 columns
- 202,723 rows

### 5.4 Imputation Summary

| Feature Type | Approach | Justification |
|--------------|----------|---------------|
| **age, gender** | Median/Mode | True random errors (<1%), safe to fill |
| **Disease-specific (22 features)** | Masked OR Sentinel | Structural missingness, keep as "not applicable" |

### 5.5 Target Column Handling

Each sample has only ONE disease label (from source dataset):
- Diabetes samples: `diabetes_risk=0/1`, other targets=NaN
- Heart samples: `heart_disease_risk=0/1`, other targets=NaN
- Liver samples: `liver_disease_risk=0/1`, other targets=NaN
- Kidney samples: `kidney_disease_risk=0/1`, other targets=NaN

**Training Strategy:** Masked loss function
- Only compute loss for diseases where labels exist
- Ignore NaN targets (not counted in loss)
- Each sample trains only its relevant disease head

### 5.6 Scaling Strategy for Each Version

**Masked Version:**
```python
# Scale value columns only (exclude mask columns)
# Mask columns stay as 0/1 integers
value_features = ['bmi', 'glucose', ...]  # Scale these
mask_features = ['bmi_mask', 'glucose_mask', ...]  # Keep as is
```

**Sentinel Version:**
```python
# Calculate mean/std EXCLUDING -999
real_values = df[df != -999]
scaler.fit(real_values)

# Apply to real values only
df_scaled = df.copy()
df_scaled[df != -999] = scaler.transform(df[df != -999])
# Keep -999 unchanged
```

### 5.7 Defense for Evaluation Panel

**Q: "Why didn't you use standard imputation?"**

A: "Our data has structural missingness - liver patients were never tested for glucose, so we cannot infer their glucose from other features. Imputing would create false medical data. Instead, we use masking (explicit 'ignore' signal) or sentinel values (clearly outside range), allowing the model to learn which features apply to each disease type."

**Q: "Won't 60% missing hurt model performance?"**

A: "No, because:
1. Each disease uses its own feature subset - effective missingness per disease is 15-50%
2. FT-Transformer attention mechanism learns to weight relevant features
3. Multi-task learning shares patterns from common features (age, BMI)
4. Research shows this approach outperforms forced imputation on medical data"

**Q: "Why create two versions?"**

A: "Ablation study - we empirically compare both approaches:
- Masked: Better for FT-Transformer (explicit attention signals)
- Sentinel: Works with tree models (XGBoost/RF) for comparison
- Report both in thesis for scientific rigor"

### 5.8 Feature Scaling Implementation

**Script:** `src/data/scale_features.py`

**Process:**
1. Load imputed datasets (masked and sentinel versions)
2. Apply StandardScaler (z-score normalization)
3. Handle special cases:
   - Mask columns: Excluded from scaling (remain 0/1 integers)
   - Sentinel -999: Excluded from mean/std, preserved in output
   - Target columns: Excluded from scaling
   - source_dataset: Excluded (metadata)

**Scaler Statistics Saved:**
- `models/scaler_masked.joblib` - For masked version inference
- `models/scaler_sentinel.joblib` - For sentinel version inference

**Output Files:**
- `datasets/combined_unified_masked_scaled.csv` (98.9 MB, 191,363 × 51)
- `datasets/combined_unified_sentinel_scaled.csv` (55.4 MB, 191,363 × 29)

### 5.9 QA Validation & Data Fixes

**Script:** `src/data/qa_validation.py`, `src/data/fix_data_issues.py`

**Issues Found & Fixed:**

| Issue | Original | Fixed | Solution |
|-------|----------|-------|----------|
| Duplicate rows | 11,497 total | 137 | Removed in fix_data_issues.py |
| Liver target encoding | 1=disease, 2=no | 1=disease, 0=no | Converted 2→0 |
| Diabetes target | Continuous 0.027-0.672 | Binary 0/1 | Binarized at 0.5 threshold |

**Duplicate Breakdown:**
- Liver dataset: 11,323 duplicates (37%) - data collection artifact
- Heart dataset: 37 duplicates (<0.1%)
- Diabetes/Kidney: 0 duplicates

**Final Clean Data:** 191,363 rows (reduced from 202,723)

**Acceptable Edge Cases:**
- BMI up to 55 std → Super morbid obesity (medically real)
- ALT/AST up to 30x normal → Severe liver disease (expected)
- 137 remaining near-duplicates → Different source datasets, valid

### 5.10 Train/Val/Test Splits

**Script:** `src/data/create_splits.py`

**Split Configuration:**
| Split | Ratio | Samples | Purpose |
|-------|-------|---------|---------|
| Train | 70% | 133,953 | Model training |
| Val | 15% | 28,705 | Hyperparameter tuning |
| Test | 15% | 28,705 | Final evaluation |

**Stratification:** By `source_dataset` to ensure proportional disease representation:

| Disease | Train % | Val % | Test % |
|---------|---------|-------|--------|
| Diabetes | 52.3% | 52.3% | 52.3% |
| Heart | 36.6% | 36.6% | 36.6% |
| Liver | 10.1% | 10.1% | 10.1% |
| Kidney | 1.1% | 1.1% | 1.1% |

**Target Positive Rates (Verified Consistent):**

| Target | Train | Val | Test |
|--------|-------|-----|------|
| diabetes_risk | 2.5% | 2.6% | 2.8% |
| heart_disease_risk | 50.2% | 49.2% | 49.8% |
| liver_disease_risk | 71.1% | 71.3% | 72.1% |
| kidney_disease_risk | 76.2% | 71.1% | 73.1% |

**Output Structure:**
```
datasets/splits/
├── masked/
│   ├── train.csv (67.2 MB, 133,953 × 51)
│   ├── val.csv (14.4 MB, 28,705 × 51)
│   └── test.csv (14.4 MB, 28,705 × 51)
├── sentinel/
│   ├── train.csv (38.3 MB, 133,953 × 29)
│   ├── val.csv (8.2 MB, 28,705 × 29)
│   └── test.csv (8.2 MB, 28,705 × 29)
└── split_indices.joblib (reproducibility)
```

**Random Seed:** 42 (for reproducibility)

**Data Leakage Prevention:**
1. Same indices used for both masked and sentinel versions
2. Scaling fitted on training data, applied to all splits
3. Split indices saved for future reference

---

## 6. Model Architecture (Planned)

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
