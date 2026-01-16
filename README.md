# 🏥 Intelligent Multi-Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Bachelor Thesis Project (BTP) 2025**  
> IIIT Sri City | B.Tech Computer Science & Engineering  
> Supervisor: Dr. R. Selvi

---

## 📊 PROJECT STATUS DASHBOARD

| Metric | Status |
|--------|--------|
| **Current Week** | Week 0 (Pre-Development) |
| **Overall Progress** | 🟡 15% |
| **Phase** | Setup & Planning |
| **Last Updated** | January 2025 |
| **Next Milestone** | EDA Completion (Week 2) |

### 🎯 Quick Status
```
✅ Project Structure Created
✅ Datasets Collected & Verified  
✅ Documentation Ready
⬜ Environment Setup Pending
⬜ EDA Not Started
⬜ Model Development Not Started
```

---

## 🗓️ WEEKLY PROGRESS LOG

### 📅 Week 0: Project Initialization (Pre-Semester)
**Status:** ✅ COMPLETED  
**Date Range:** January 2025

#### Completed Tasks:
- [x] Analyzed BTP proposal document thoroughly
- [x] Created comprehensive project folder structure (50+ directories)
- [x] Created detailed 14-week execution plan with daily breakdowns
- [x] Set up progress tracking system
- [x] **Finalized 4 clinical datasets with 100% lab report values**
- [x] **Heart (70K) + Diabetes (768) + Kidney (1.6K) + Liver (583)**
- [x] **Designed multi-modal input: Structured + Checklist + NLP**
- [x] Verified all datasets for clinical applicability and XAI explanations
- [x] Created clinical feature mapping document
- [x] Created workflow risk assessment document
- [x] Updated all documentation for real-world clinical workflow

#### Dataset Verification Results:
| Dataset | File | Records | Features | Target Column | Status |
|---------|------|---------|----------|---------------|--------|
| Heart Disease | `heart.csv` | 70,000 | 13 | `cardio` (0/1) | ✅ Verified |
| Diabetes | `diabetes.csv` | 768 | 9 | `Outcome` (0/1) | ✅ Verified |
| Chronic Kidney Disease | `Chronic_Kidney_Dsease_data.csv` | 1,659 | 54 | `Diagnosis` | ✅ Verified |
| Liver Disease | `liver.csv` | 583 | 11 | `Dataset` (1/2) | ✅ Verified |

#### Files Created:
- `PROJECT_STRUCTURE.md` - Complete folder hierarchy
- `WEEKLY_EXECUTION_PLAN.md` - 14-week detailed plan
- `PROGRESS_TRACKER.md` - Milestone tracking dashboard
- `requirements.txt` - All Python dependencies
- `.env.example` - API keys template
- `.gitignore` - Comprehensive ignore rules
- Skeleton code files for API, WebApp, and source modules

#### GitHub:
- Repository: `Intelligent-Multi-Disease-Risk-Prediction-System`
- Initial push: Pending

---

### 📅 Week 1: Environment Setup & Data Loading
**Status:** ⬜ NOT STARTED  
**Date Range:** TBD

#### Planned Tasks:
- [ ] Set up Python virtual environment
- [ ] Install all dependencies from requirements.txt
- [ ] Configure VS Code with necessary extensions
- [ ] Create data loading utilities
- [ ] Load and validate all datasets
- [ ] Initial commit to GitHub

#### Expected Deliverables:
- Working development environment
- Data loading scripts
- Initial data validation report

---

### 📅 Week 2: Exploratory Data Analysis (EDA)
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] Statistical analysis for all 4 datasets
- [ ] Visualizations (distributions, correlations, class balance)
- [ ] Missing value analysis
- [ ] Outlier detection
- [ ] EDA notebooks completion

#### Expected Deliverables:
- 4 complete EDA notebooks
- Data quality report
- Feature insights document

---

### 📅 Weeks 3-5: Data Preprocessing & Feature Engineering
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] Handle missing values (imputation strategies)
- [ ] Normalize/standardize features
- [ ] Create unified multi-disease schema
- [ ] Feature engineering pipelines
- [ ] Cross-disease feature mappings

---

### 📅 Weeks 6-7: Classical ML Models
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] Implement XGBoost models (4 diseases)
- [ ] Implement LightGBM models
- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validation setup
- [ ] Model comparison analysis

---

### 📅 Week 8: Deep Learning - FT-Transformer
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] Implement FT-Transformer architecture
- [ ] Multi-task learning setup
- [ ] Training pipeline with early stopping
- [ ] GPU optimization

---

### 📅 Week 9: Clinical NLP Integration
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] PubMedBERT/BioMistral setup
- [ ] Text preprocessing pipeline
- [ ] Embedding generation
- [ ] Multimodal fusion

---

### 📅 Week 10: Explainability (XAI)
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] SHAP integration for all models
- [ ] LLM explanation generation
- [ ] Visualization components

---

### 📅 Weeks 11-12: API & Web Application
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] FastAPI backend implementation
- [ ] Streamlit frontend
- [ ] API documentation

---

### 📅 Weeks 13-14: Testing & Documentation
**Status:** ⬜ NOT STARTED

#### Planned Tasks:
- [ ] Unit and integration tests
- [ ] Performance benchmarking
- [ ] Thesis first draft
- [ ] Final documentation

---

## 🎯 Project Overview

A comprehensive AI platform that simultaneously predicts risks for **four major diseases** from routine clinical data:
- 🫀 **Heart Disease** - Cardiovascular risk assessment
- 🩸 **Diabetes (Type 2)** - Metabolic disorder risk prediction
- � **Chronic Kidney Disease** - Renal function risk assessment
- 🫁 **Liver Disease** - Hepatic function risk analysis

### 🌟 Key Innovation: Real-World Clinical Workflow
**Handles incomplete patient data** - works with whatever tests are available (vitals only, partial labs, or complete panel). **Multi-modal input** - structured lab values + binary checklists (medications) + NLP text (lifestyle). **Comorbidity-aware** - models disease interactions (diabetes → kidney damage). **LLM-enhanced explanations** - converts technical SHAP outputs to patient-friendly language with personalized diet and lifestyle recommendations.

### Key Innovation
**First undergraduate multi-disease AI system** combining:
- Classical Machine Learning (XGBoost, LightGBM)
- Deep Learning (FT-Transformer)
- Clinical NLP (BioMistral/PubMedBERT)
- LLM-enhanced Explainability (GPT-4/Llama)

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **ML/DL** | PyTorch, scikit-learn, XGBoost, LightGBM, Optuna |
| **NLP** | Transformers, PubMedBERT, BioMistral (lifestyle extraction) |
| **XAI** | SHAP (technical explanations) |
| **LLM** | GPT-4/Llama (human-friendly explanations + diet recommendations) |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit |
| **Data** | Pandas, NumPy, Polars |
| **Visualization** | Plotly, Matplotlib, Seaborn |

---

## 📁 Project Structure

```
Intelligent-Multi-Disease-Risk-Prediction-System/
├── datasets/           # Raw dataset files
│   ├── heart.csv      # Cardiovascular Disease (70,000 records)
│   ├── diabetes.csv   # Pima Indians Diabetes (768 records)
│   ├── Chronic_Kidney_Dsease_data.csv # Chronic Kidney Disease (1,659 records)
│   └── liver.csv      # Indian Liver Patient (583 records)
├── data/               # Processed data
│   ├── raw/           # Raw data backups
│   ├── processed/     # Cleaned datasets
│   └── embeddings/    # Text embeddings
├── notebooks/          # Jupyter notebooks
│   └── eda/           # Exploratory Data Analysis
├── src/                # Source code modules
│   ├── data/          # Data processing utilities
│   ├── models/        # ML/DL model implementations
│   ├── nlp/           # NLP components
│   ├── explainability/# XAI modules
│   └── utils/         # Helper functions
├── api/                # FastAPI backend
├── webapp/             # Streamlit frontend
├── models/             # Saved model weights
│   ├── ml/            # Classical ML models
│   └── dl/            # Deep learning models
├── tests/              # Test suite
├── docs/               # Documentation
│   └── thesis/        # BTP thesis documents
├── configs/            # Configuration files
└── logs/               # Training logs
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Intelligent-Multi-Disease-Risk-Prediction-System.git
cd Intelligent-Multi-Disease-Risk-Prediction-System

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env with your API keys
```

### Running the Application

```bash
# Start API server
uvicorn api.main:app --reload --port 8000

# Start web application (new terminal)
streamlit run webapp/app.py
```

---

## 📊 Model Performance (To Be Updated)

| Disease | Model | AUC-ROC | Precision | Recall | F1-Score |
|---------|-------|---------|-----------|--------|----------|
| Heart Disease | XGBoost | - | - | - | - |
| Heart Disease | FT-Transformer | - | - | - | - |
| Diabetes | XGBoost | - | - | - | - |
| Diabetes | FT-Transformer | - | - | - | - |
| Chronic Kidney Disease | XGBoost | - | - | - | - |
| Chronic Kidney Disease | FT-Transformer | - | - | - | - |
| Liver Disease | XGBoost | - | - | - | - |
| Liver Disease | FT-Transformer | - | - | - | - |

**Target: >0.85 AUC-ROC for all diseases**

---

## 🔌 API Endpoints (To Be Implemented)

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/health` | GET | Health check | ⬜ |
| `/predict` | POST | Multi-disease risk prediction | ⬜ |
| `/predict/{disease}` | POST | Single disease prediction | ⬜ |
| `/explain` | POST | SHAP-based explanations | ⬜ |
| `/explain-nlp` | POST | LLM-generated explanations | ⬜ |
| `/batch-predict` | POST | Batch predictions | ⬜ |

---

## 📚 Datasets Summary

| Disease | Dataset | Records | Features | Target | Key Features |
|---------|---------|---------|----------|--------|---------------|
| Heart | Cardiovascular Disease | 70,000 | 13 | `cardio` (0/1) | Age, BP, cholesterol, glucose, BMI |
| Diabetes | Pima Indians | 768 | 9 | `Outcome` (0/1) | **Glucose, insulin**, BP, BMI, family history |
| Chronic Kidney Disease | CKD Dataset | 1,659 | 54 | `Diagnosis` | **Creatinine, BUN, GFR**, BP, glucose, HbA1c, electrolytes |
| Liver | Indian Liver Patient | 583 | 11 | `Dataset` (1/2) | **Bilirubin, ALT, AST**, albumin, proteins |

**Total samples: ~72,000 | 100% clinical lab values | Multi-modal: Structured + Checklist + NLP**

**Note:** MIMIC-III clinical notes will be integrated in Semester 2.

---

## 📄 Documentation

- [Project Structure](PROJECT_STRUCTURE.md) - Complete folder hierarchy
- [Weekly Execution Plan](WEEKLY_EXECUTION_PLAN.md) - 14-week detailed plan
- [Progress Tracker](PROGRESS_TRACKER.md) - Milestone tracking dashboard

---

## 🔄 Git Workflow

```bash
# Weekly push routine
git add .
git commit -m "Week X: [Description of work done]"
git push origin main
```

**Branch Strategy:**
- `main` - Stable, tested code
- `develop` - Active development
- `feature/*` - New features
- `experiment/*` - ML experiments

---

## 📞 Contact

- **Student**: [Your Name]
- **Supervisor**: Dr. R. Selvi
- **Institution**: IIIT Sri City
- **GitHub**: [Intelligent-Multi-Disease-Risk-Prediction-System](https://github.com/YOUR_USERNAME/Intelligent-Multi-Disease-Risk-Prediction-System)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- IIIT Sri City for academic support
- Dr. R. Selvi for project guidance
- Open-source community for datasets and tools

---

## 📝 Change Log

| Date | Week | Changes | Commit |
|------|------|---------|--------|
| Jan 2025 | 0 | Initial project setup, structure creation, dataset verification | Pending |
| - | 1 | - | - |
| - | 2 | - | - |

---

*Last Updated: January 2025 | Week 0*
- PhysioNet for MIMIC-III access
- Open-source ML/DL community

---

**⭐ If you find this project useful, please consider giving it a star!**
