Project Structure — Snapshot (2026-03-22)

This file lists the repository files and folders as currently present in the workspace. Use this as a quick navigational README.

Top-level files
- activate_env.bat — helper to activate virtual environment on Windows
- app.py — (root) application entry / demo
- demo.py — small demo / script
- train.py — training script
- test_app.py — test file for app
- README.md — project README (existing)
- PROJECT_STRUCTURE_README.md — (this file)
- requirements.txt — Python dependencies
- START_HERE.md, README.md, IMPLEMENTATION_SUMMARY.md, PROJECT_SUMMARY.md, etc. — project docs and planning files

Top-level folders
- api/
  - __init__.py
  - main.py  — API entry points

- configs/
  - unified_schema.yaml
  - week7_ft_transformer.yaml (referenced by notebooks)

- data/
  - embeddings/
  - processed/
  - raw/
    - CLINICAL_TEXT_DATASETS.md
    - sample_patient_histories.json
    - ChatDoctor/, MTS-Dialog/  — raw clinical text resources

- datasets/
  - chronic_kidney_disease_balanced.csv
  - diabetes.csv, heart.csv, liver.csv, etc. — CSV datasets
  - splits/
    - split_indices.joblib
    - masked/
    - sentinel/
  - unified/
    - diabetes_unified.csv, heart_unified.csv, kidney_unified.csv, liver_unified.csv

- datasets_evaluated/
  - DATASET_EVALUATION_LOG.md
  - rejected/ — rejected datasets archive

- docs/
  - PLANNING.md, TECHNICAL_DOCS.md, TESTING_STRATEGY.md
  - thesis/ — latex reports

- logs/ — runtime logs

- models/
  - dl/ — deep learning model artifacts
  - ml/ — classical ML artifacts

- notebooks/
  - eda/ — exploratory notebooks
  - ml_baselines/
    - week6_ml_baselines.ipynb  — reference ML baseline tests (feature removal experiments)
  - dl_baselines/
    - ft_test.ipynb  — FT-Transformer manual tests (we updated this)

- src/
  - data/, explainability/, models/, nlp/, utils/ — source modules

- static/, templates/, webapp/
  - webapp/app.py — small web application entry

Notes and key locations
- Primary DL model checkpoint referenced by notebooks: week7_ft_transformer_artifacts (models/dl/week7_ft_transformer/best_model.pt)
- Test split used by notebooks: path from `configs/week7_ft_transformer.yaml` (datasets/splits/masked/test.csv)
- Manual patient testing cells live in: notebooks/dl_baselines/ft_test.ipynb
- Week‑6 ML baseline manual patient examples: notebooks/ml_baselines/week6_ml_baselines.ipynb

How to use
- Open the notebooks listed above to reproduce experiments: run cells in order starting from the top (imports, data load, model load).
- Manual patient scenarios are in `notebooks/dl_baselines/ft_test.ipynb` (edit and run per patient cell).

If you want, I can:
- Replace `PROJECT_STRUCTURE_README.md` into `README.md` (overwrite)
- Add brief `table-of-contents` links to the important notebooks and scripts
- Export a plain text tree output (e.g., `tree`-like) to a file

