# ❌ Rejected Datasets

This folder contains datasets that were evaluated but **not selected** for the final project.

## 📁 Contents (5 Files)

### 1. **Breast_Cancer (3).csv**
- **Rejection Reason:** Diagnostic tumor classification data (not risk prediction)
- **Issue:** Contains tumor morphology measurements (radius, texture, perimeter), not clinical lab values from blood reports
- **Size:** 569 samples, 30 imaging-based features

### 2. **diabetes_012_health_indicators_BRFSS2015.csv**
- **Rejection Reason:** Survey-based data without clinical lab values
- **Issue:** 253,680 samples but 0% clinical measurements (no glucose, insulin, or objective biomarkers)
- **Size:** 253,680 samples, 21 survey questions

### 3. **healthcare-dataset-stroke-data.csv**
- **Rejection Reason:** Minimal clinical lab values (mostly demographics)
- **Issue:** Only glucose and BMI are measurable; ~90% features are survey/demographic data
- **Size:** 5,110 samples, <10% clinical features

### 4. **heart.csv**
- **Rejection Reason:** Replaced with larger cardiovascular dataset
- **Issue:** Initially considered but upgraded to 70,000-sample cardiovascular disease dataset
- **Size:** Smaller sample size compared to final selection

### 5. **kidney_disease.csv**
- **Rejection Reason:** Insufficient samples and limited features
- **Issue:** Only 400 samples with 24 features; replaced with comprehensive CKD dataset (1,659 samples, 54 features)
- **Size:** 400 samples (inadequate for robust model training)

---

## 📊 Evaluation Summary

| Dataset | Samples | Lab Values | Status |
|---------|---------|------------|--------|
| Breast Cancer | 569 | 0% | ❌ Diagnostic imaging |
| BRFSS Diabetes | 253,680 | 0% | ❌ Survey only |
| Stroke | 5,110 | 10% | ❌ Demographics heavy |
| Heart (old) | ~3K | 100% | ⚠️ Upgraded |
| Kidney (old) | 400 | 90% | ⚠️ Too small |

---

## 📄 Documentation

See [DATASET_EVALUATION_LOG.md](../DATASET_EVALUATION_LOG.md) for detailed evaluation methodology and rejection criteria.

---

**Note:** These datasets were downloaded, thoroughly analyzed, and rejected based on clinical applicability criteria. This demonstrates comprehensive dataset research methodology for the Bachelor Thesis Project.
