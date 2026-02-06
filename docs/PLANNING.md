# 📅 BTP Planning & Execution
## Multi-Disease AI Healthcare Platform

**Last Updated:** January 24, 2026

---

## 📊 Semester Overview

| Phase | Weeks | Focus Area | Deliverable |
|-------|-------|------------|-------------|
| **Foundation** | 1-2 | Setup & EDA | Environment ready, data understood |
| **Data Pipeline** | 3-5 | Fusion & Features | Unified multimodal dataset |
| **Modeling** | 6-9 | ML/DL/NLP/XAI | Hybrid model with dual explanations |
| **Production** | 10-11 | API & Web App | Deployed application |
| **Finalization** | 12-14 | Testing & Docs | Complete submission package |

---

## 📆 Weekly Plan

### Week 1: Environment Setup ✅ COMPLETE
- [x] Project folder structure
- [x] Python virtual environment (C:\dev\venvs\btp_env)
- [x] All dependencies installed
- [x] Datasets collected (202K samples total)

### Week 2: Exploratory Data Analysis
- [ ] EDA on Heart Disease (70K samples)
- [ ] EDA on Diabetes (100K samples)
- [ ] EDA on Kidney Disease (1.6K samples)
- [ ] EDA on Liver Disease (30K samples)
- [ ] Missing value analysis
- [ ] Distribution plots
- [ ] Correlation analysis

### Week 3-4: Data Preprocessing
- [ ] Handle missing values
- [ ] Outlier detection and treatment
- [ ] Feature scaling/normalization
- [ ] Class imbalance handling (SMOTE)
- [ ] Train/val/test splits

### Week 5: Feature Engineering
- [ ] Create derived features (BMI, ratios)
- [ ] Feature selection (mutual information, importance)
- [ ] Unified feature schema across diseases

### Week 6: ML Baselines
- [ ] Logistic Regression
- [ ] Random Forest
- [ ] XGBoost
- [ ] LightGBM
- [ ] Cross-validation
- [ ] Hyperparameter tuning

### Week 7: Deep Learning
- [ ] FT-Transformer implementation
- [ ] TabNet exploration
- [ ] Multi-task learning architecture
- [ ] Training pipeline

### Week 8: NLP Integration
- [ ] Clinical text preprocessing
- [ ] BioMistral/PubMedBERT embeddings
- [ ] Feature extraction from text
- [ ] Integration with structured features

### Week 9: Explainability (XAI)
- [ ] SHAP implementation
- [ ] Feature importance visualization
- [ ] LLM explanation generation
- [ ] Dual explanation system

### Week 10: API Development
- [ ] FastAPI endpoints
- [ ] Model serving
- [ ] Input validation
- [ ] Response formatting

### Week 11: Web Application
- [ ] Streamlit UI
- [ ] Input forms
- [ ] Results visualization
- [ ] Explanation display

### Week 12-13: Testing & Documentation
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance testing
- [ ] Thesis writing

### Week 14: Final Submission
- [ ] Code cleanup
- [ ] Documentation finalization
- [ ] Thesis submission
- [ ] Defense preparation

---

## 🎯 Key Milestones

| # | Milestone | Target | Status |
|---|-----------|--------|--------|
| 1 | Environment Setup | Week 1 | ✅ Complete |
| 2 | EDA Complete | Week 2 | ⬜ Not Started |
| 3 | Unified Dataset Ready | Week 4 | ⬜ Not Started |
| 4 | ML Baseline >85% AUC | Week 6 | ⬜ Not Started |
| 5 | DL Model Trained | Week 7 | ⬜ Not Started |
| 6 | NLP Integration Done | Week 8 | ⬜ Not Started |
| 7 | XAI System Working | Week 9 | ⬜ Not Started |
| 8 | API Deployed | Week 10 | ⬜ Not Started |
| 9 | Web App Live | Week 11 | ⬜ Not Started |
| 10 | Thesis Submitted | Week 14 | ⬜ Not Started |

---

## 📚 Literature Review

### Key Papers to Read

| # | Paper | Authors | Year | Status |
|---|-------|---------|------|--------|
| 1 | FT-Transformer | Gorishniy et al. | 2021 | ⬜ |
| 2 | SHAP Explainability | Lundberg et al. | 2017 | ⬜ |
| 3 | PubMedBERT | Gu et al. | 2021 | ⬜ |
| 4 | BioMistral | Labrak et al. | 2024 | ⬜ |
| 5 | TabNet | Arik et al. | 2021 | ⬜ |
| 6 | XGBoost | Chen et al. | 2016 | ⬜ |

---

## 📝 Thesis Writing Progress

| Chapter | Target Pages | Status |
|---------|-------------|--------|
| Abstract | 1 | ⬜ |
| Introduction | 4 | ⬜ |
| Literature Review | 6 | ⬜ |
| Methodology | 10 | ⬜ |
| Results | 10 | ⬜ |
| Discussion | 5 | ⬜ |
| Conclusion | 2 | ⬜ |
| **Total** | **40** | **0%** |

---

## 🔧 Technical Decisions Log

### Dataset Decisions
- **Heart:** Using Cardiovascular Disease (70K) - large, clinical features
- **Diabetes:** Upgraded from Pima (768) to Health Indicators (100K)
- **Kidney:** Keeping CKD (1.6K) - most comprehensive available
- **Liver:** Upgraded from ILPD (583) to LPD (30K)

### Architecture Decisions
- **Environment:** External venv to avoid OneDrive sync issues
- **Models:** Multi-task learning for shared representations
- **XAI:** Dual system (SHAP + LLM explanations)

---

*Last Updated: January 24, 2026*
