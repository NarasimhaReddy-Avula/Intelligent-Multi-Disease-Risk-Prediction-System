#  Intelligent Multi-Disease Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com)

> **Bachelor Thesis Project (BTP) 2025-2026**  
> IIIT Sri City | B.Tech CSE | Supervisor: Dr. R. Selvi

---

## 🚀 **"What Should I Do?" → [START HERE!](START_HERE.md)**

**Current Week:** Week 3 - Data Preprocessing  
**Quick Links:**
- 📖 [Comprehensive Guide](NEXT_STEPS.md) - Detailed instructions with code
- ✅ [Daily Checklist](WEEK_3_CHECKLIST.md) - Track your daily progress
- 📁 [Dataset Setup](DATASET_SETUP_REQUIRED.md) - If datasets aren't ready yet

---

##  Project Overview

A multi-disease AI healthcare platform that predicts risk for:
-  **Heart Disease** (70,000 samples)
-  **Diabetes** (100,000 samples)  
-  **Kidney Disease** (1,659 samples)
-  **Liver Disease** (30,691 samples)

**Total Dataset Size:** 202,350 samples

### Key Features
- **Multi-modal Input:** Lab values + NLP text extraction + checkboxes
- **Explainable AI:** SHAP visualizations + LLM-generated explanations
- **Production Ready:** FastAPI backend + Streamlit frontend

---

##  Quick Start

```powershell
# Activate environment
.\activate_env.bat

# Run API
cd api && uvicorn main:app --reload

# Run Web App  
cd webapp && streamlit run app.py
```

---

##  Project Structure

```
PROJECT/
 api/              # FastAPI backend
 datasets/         # Final datasets (4 diseases)
 docs/             # Documentation
    TECHNICAL_DOCS.md
    PLANNING.md
 models/           # Saved ML/DL models
 notebooks/        # Jupyter notebooks
 src/              # Source code
 tests/            # Test suite
 webapp/           # Streamlit UI
```

---

##  Current Status

| Milestone | Status |
|-----------|--------|
| Environment Setup |  Complete |
| Dataset Collection |  Complete (202K samples) |
| EDA |  Complete (Week 2) |
| **Data Preprocessing** | **🔄 In Progress (Week 3)** |
| ML Baselines |  Not Started |
| Deep Learning |  Not Started |
| NLP Integration |  Not Started |
| XAI Implementation |  Not Started |

### 📋 What to Do Next?
**See [NEXT_STEPS.md](NEXT_STEPS.md)** for detailed guidance on Week 3 tasks  
**See [WEEK_3_CHECKLIST.md](WEEK_3_CHECKLIST.md)** for daily action items

---

##  Documentation

- [Technical Documentation](docs/TECHNICAL_DOCS.md) - Setup, data, architecture
- [Planning & Progress](docs/PLANNING.md) - Weekly plan, milestones
- [Progress Tracker](PROGRESS_TRACKER.md) - Detailed tracking

---

##  Tech Stack

| Category | Tools |
|----------|-------|
| **ML** | scikit-learn, XGBoost, LightGBM |
| **DL** | PyTorch, Transformers |
| **NLP** | BioMistral, Sentence-Transformers |
| **XAI** | SHAP, LIME |
| **API** | FastAPI, Uvicorn |
| **UI** | Streamlit |

---

*BTP Project 2025-2026 | IIIT Sri City*
