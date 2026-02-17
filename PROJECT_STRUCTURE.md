# 🏗️ BTP Project Structure
## Multi-Disease AI Healthcare Platform

```
BTP-Healthcare-AI/
│
├── 📁 data/
│   ├── 📁 raw/                          # Original unprocessed datasets
│   │   ├── heart/                       # Framingham Heart Study data
│   │   ├── diabetes/                    # Pima Indians Dataset
│   │   ├── breast_cancer/               # Wisconsin Diagnostic data
│   │   ├── liver/                       # Indian Liver Patient data
│   │   └── mimic/                       # MIMIC-III clinical notes (after access)
│   │
│   ├── 📁 processed/                    # Cleaned and preprocessed data
│   │   ├── unified_schema.csv           # Fused multi-disease dataset
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   │
│   ├── 📁 embeddings/                   # NLP embeddings from clinical notes
│   │   └── clinical_embeddings.pkl
│   │
│   └── 📁 external/                     # External reference data
│       └── medical_knowledge_base/      # For RAG (Semester 2)
│
├── 📁 notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_EDA_heart_disease.ipynb
│   ├── 02_EDA_diabetes.ipynb
│   ├── 03_EDA_breast_cancer.ipynb
│   ├── 04_EDA_liver_disease.ipynb
│   ├── 05_data_fusion_exploration.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_ML_baseline_experiments.ipynb
│   ├── 08_FT_transformer_experiments.ipynb
│   ├── 09_NLP_embeddings_analysis.ipynb
│   ├── 10_hybrid_model_evaluation.ipynb
│   └── 11_explainability_analysis.ipynb
│
├── 📁 src/                              # Source code (modular Python package)
│   ├── __init__.py
│   │
│   ├── 📁 data/                         # Data handling modules
│   │   ├── __init__.py
│   │   ├── loaders.py                   # Dataset loading utilities
│   │   ├── preprocessors.py             # Data cleaning & preprocessing
│   │   ├── fusion.py                    # Multi-dataset fusion pipeline
│   │   ├── feature_engineering.py       # Feature creation & selection
│   │   └── balancing.py                 # SMOTE & class balancing
│   │
│   ├── 📁 models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── ml_baselines.py              # XGBoost, LightGBM, RF, LR
│   │   ├── ft_transformer.py            # FT-Transformer architecture
│   │   ├── hybrid_model.py              # Combined ML + DL + NLP model
│   │   └── model_utils.py               # Training, evaluation utilities
│   │
│   ├── 📁 nlp/                          # NLP components
│   │   ├── __init__.py
│   │   ├── text_preprocessor.py         # Clinical notes preprocessing
│   │   ├── embeddings.py                # BioMistral/PubMedBERT embeddings
│   │   └── clinical_ner.py              # Named entity recognition (optional)
│   │
│   ├── 📁 explainability/               # Explainable AI components
│   │   ├── __init__.py
│   │   ├── shap_explainer.py            # SHAP analysis
│   │   ├── llm_explainer.py             # LLM-generated explanations
│   │   └── visualization.py             # Explanation visualizations
│   │
│   └── 📁 utils/                        # Utility functions
│       ├── __init__.py
│       ├── config.py                    # Configuration management
│       ├── logger.py                    # Logging utilities
│       └── metrics.py                   # Custom evaluation metrics
│
├── 📁 api/                              # FastAPI backend
│   ├── __init__.py
│   ├── main.py                          # FastAPI application entry
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── predict.py                   # /predict endpoint
│   │   ├── explain_shap.py              # /explain endpoint
│   │   └── explain_nlp.py               # /explain-nlp endpoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── patient.py                   # Patient data schemas
│   │   └── prediction.py                # Prediction response schemas
│   └── middleware/
│       ├── __init__.py
│       └── error_handler.py             # Error handling middleware
│
├── 📁 webapp/                           # Streamlit/Gradio frontend
│   ├── app.py                           # Main Streamlit application
│   ├── pages/
│   │   ├── 01_🏠_Home.py
│   │   ├── 02_📊_Risk_Assessment.py
│   │   ├── 03_📈_Visualizations.py
│   │   ├── 04_📝_Patient_Report.py
│   │   └── 05_ℹ️_About.py
│   ├── components/
│   │   ├── input_forms.py               # Data input components
│   │   ├── result_display.py            # Prediction result display
│   │   └── shap_plots.py                # SHAP visualization components
│   └── assets/
│       ├── styles.css
│       └── images/
│
├── 📁 models/                           # Saved model artifacts
│   ├── ml/
│   │   ├── xgboost_heart.pkl
│   │   ├── xgboost_diabetes.pkl
│   │   ├── xgboost_breast.pkl
│   │   └── xgboost_liver.pkl
│   ├── dl/
│   │   └── ft_transformer_hybrid.pt
│   └── embeddings/
│       └── pubmedbert_model/
│
├── 📁 tests/                            # Test suite
│   ├── __init__.py
│   ├── test_data/
│   │   └── sample_patients.json
│   ├── test_preprocessors.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_explainability.py
│
├── 📁 docs/                             # Documentation
│   ├── 📁 thesis/                       # BTP Thesis documents
│   │   ├── chapters/
│   │   │   ├── 01_introduction.tex
│   │   │   ├── 02_literature_review.tex
│   │   │   ├── 03_methodology.tex
│   │   │   ├── 04_results.tex
│   │   │   ├── 05_discussion.tex
│   │   │   └── 06_conclusion.tex
│   │   ├── figures/
│   │   ├── tables/
│   │   ├── references.bib
│   │   └── main.tex
│   │
│   ├── 📁 paper/                        # IEEE Conference Paper
│   │   ├── paper.tex
│   │   └── figures/
│   │
│   ├── 📁 api_docs/                     # API documentation
│   │   └── openapi.yaml
│   │
│   └── 📁 presentations/                # Presentation slides
│       ├── mid_review.pptx
│       └── final_defense.pptx
│
├── 📁 literature/                       # Research papers & references
│   ├── key_papers/                      # 15 key papers (PDF)
│   └── literature_summary.md            # Curated summary
│
├── 📁 configs/                          # Configuration files
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── api_config.yaml
│
├── 📁 scripts/                          # Utility scripts
│   ├── download_datasets.py
│   ├── train_models.py
│   ├── generate_report.py
│   └── deploy.sh
│
├── 📁 logs/                             # Application logs
│   └── .gitkeep
│
├── 📁 outputs/                          # Generated outputs
│   ├── reports/                         # Generated patient reports
│   ├── visualizations/                  # Saved plots & figures
│   └── experiments/                     # Experiment tracking results
│
├── 📄 .gitignore
├── 📄 .env.example                      # Environment variables template
├── 📄 requirements.txt                  # Python dependencies
├── 📄 environment.yml                   # Conda environment file
├── 📄 Dockerfile                        # Docker configuration
├── 📄 docker-compose.yml                # Docker compose for services
├── 📄 Makefile                          # Common commands
├── 📄 setup.py                          # Package installation
├── 📄 pyproject.toml                    # Modern Python project config
├── 📄 README.md                         # Project documentation
├── 📄 CHANGELOG.md                      # Version history
├── 📄 LICENSE                           # MIT/Apache license
└── 📄 CONTRIBUTING.md                   # Contribution guidelines
```

---

## 📂 Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `data/` | All datasets - raw, processed, and embeddings |
| `notebooks/` | Exploratory analysis and experimentation |
| `src/` | Core Python package with modular code |
| `api/` | FastAPI backend service |
| `webapp/` | Streamlit/Gradio frontend application |
| `models/` | Saved trained model artifacts |
| `tests/` | Unit and integration tests |
| `docs/` | Thesis, paper, and documentation |
| `literature/` | Research papers and summaries |
| `configs/` | Configuration YAML files |
| `scripts/` | Utility and automation scripts |
| `outputs/` | Generated reports and visualizations |

---

## 🔧 Key Files

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python dependencies |
| `README.md` | Project overview and setup guide |
| `Dockerfile` | Containerization for deployment |
| `.env.example` | Template for API keys (OpenAI, HuggingFace) |
| `Makefile` | Quick commands (train, test, deploy) |

---

## 📋 Initial Setup Checklist

- [ ] Create folder structure
- [ ] Initialize Git repository
- [ ] Set up virtual environment
- [ ] Install core dependencies
- [ ] Download initial datasets
- [ ] Apply for MIMIC-III access
- [ ] Configure API keys (.env file)
