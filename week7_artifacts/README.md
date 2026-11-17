# week7_artifacts — PRODUCTION artifacts

Despite the directory name, this folder holds the **deployed** model stack:

- `scaler.pkl` — StandardScaler fit on the sentinel-encoded combined dataset (24 cols).
- `{diabetes,heart,liver,kidney}_XGBoost.pkl` — per-disease XGBoost classifiers.
- `summary.json`, `decision_thresholds_per_disease.csv`, metrics — evaluation outputs.
- `best_model.pt` — **experimental** FT-Transformer checkpoint. NOT loaded by the app,
  API, or report generator. Kept for research comparison only; its training runs were
  inconsistent — do not cite its metrics without re-running training.

Loaded by `src/inference/engine.py` (`load_artifacts()`).
