# 🚨 Important: Dataset Setup Required

## Current Status: Datasets Not Yet in Repository

The code and guidance have been prepared for Week 3 (Data Preprocessing), but the actual dataset CSV files need to be added to the repository first.

---

## Required Datasets

According to the project documentation, you need these 4 datasets:

### 1. Heart Disease (70,000 samples)
- **Filename:** `datasets/heart.csv`
- **Source:** Cardiovascular Disease dataset
- **Expected columns:** Age, Gender, Blood Pressure, Cholesterol, etc.

### 2. Diabetes (100,000 samples) 
- **Filename:** `datasets/diabetes.csv`
- **Source:** Health Indicators dataset (not the small Pima Indians dataset)
- **Expected samples:** 100,000 rows

### 3. Kidney Disease (1,659 samples)
- **Filename:** `datasets/Chronic_Kidney_Dsease_data.csv` *(Note: 'Dsease' typo is intentional)*
- **Source:** Chronic Kidney Disease dataset
- **Expected samples:** ~1,600 rows

### 4. Liver Disease (30,691 samples)
- **Filename:** `datasets/liver.csv`
- **Source:** Liver Patient Dataset (LPD)
- **Expected samples:** ~30,000 rows

---

## Next Steps - Dataset Setup

### Option 1: If You Have the Datasets Locally

If the datasets are on your local machine:

```bash
# Create datasets directory
mkdir datasets

# Copy your CSV files to the datasets directory
# Make sure they have the exact names expected:
# - heart.csv
# - diabetes.csv  
# - Chronic_Kidney_Dsease_data.csv
# - liver.csv
```

Then:
```bash
# Add to git
git add datasets/*.csv

# Commit
git commit -m "Add disease datasets for preprocessing"

# Push
git push
```

### Option 2: If You Need to Download the Datasets

Common sources for medical datasets:

1. **Kaggle Datasets:**
   - Heart Disease: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
   - Diabetes: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
   - Kidney: https://www.kaggle.com/datasets/mansoordaku/ckdisease
   - Liver: https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset

2. **UCI Machine Learning Repository:**
   - https://archive.ics.uci.edu/ml/index.php

3. **Other Public Health Datasets:**
   - CDC Data
   - NHS Digital
   - Various research repositories

### Option 3: Using Sample/Synthetic Data for Development

If you want to test the code before getting the real datasets:

```python
# Create sample datasets for testing
import pandas as pd
import numpy as np

# Create sample heart dataset
np.random.seed(42)
heart_data = pd.DataFrame({
    'age': np.random.randint(20, 80, 1000),
    'gender': np.random.choice([0, 1], 1000),
    'blood_pressure': np.random.randint(80, 180, 1000),
    'cholesterol': np.random.randint(150, 300, 1000),
    'target': np.random.choice([0, 1], 1000)
})

# Save to datasets directory
import os
os.makedirs('datasets', exist_ok=True)
heart_data.to_csv('datasets/heart.csv', index=False)

# Repeat for other diseases...
```

---

## After Adding Datasets

Once your datasets are in place:

1. **Test the data loader:**
   ```bash
   cd /path/to/project
   python src/data/load_datasets.py
   ```

2. **If successful, proceed with Week 3:**
   - Follow [NEXT_STEPS.md](NEXT_STEPS.md)
   - Use [WEEK_3_CHECKLIST.md](WEEK_3_CHECKLIST.md)

3. **If there are errors:**
   - Check file paths match exactly
   - Verify CSV files are not corrupted
   - Ensure column names are reasonable
   - Check for any loading errors

---

## Directory Structure (Target)

```
Intelligent-Multi-Disease-Risk-Prediction-System/
├── datasets/                           # ← CREATE THIS
│   ├── heart.csv                       # ← ADD THESE
│   ├── diabetes.csv
│   ├── Chronic_Kidney_Dsease_data.csv
│   └── liver.csv
├── data/
│   ├── raw/                           # For clinical text data
│   └── processed/                     # Will be created by preprocessing
├── src/
│   └── data/
│       ├── load_datasets.py           # ✅ Created
│       └── preprocessing.py           # ✅ Created
├── NEXT_STEPS.md                      # ✅ Created
└── WEEK_3_CHECKLIST.md                # ✅ Created
```

---

## ⚠️ Important Notes

1. **Large Files:** If your datasets are large (>100MB), consider:
   - Using Git LFS (Large File Storage)
   - Storing datasets externally (Google Drive, S3)
   - Adding to .gitignore and documenting how to obtain them

2. **Data Privacy:** Ensure your datasets:
   - Don't contain personally identifiable information (PII)
   - Are properly anonymized
   - Have appropriate usage permissions
   - Comply with data protection regulations

3. **.gitignore:** The `.gitignore` might already exclude CSV files. Check and modify if needed:
   ```
   # Remove or comment out if you want to commit datasets
   # *.csv
   ```

---

## Questions?

If you're unsure about:
- Where your datasets are located
- How to obtain specific datasets
- Dataset format requirements
- File naming conventions

Please clarify before proceeding with Week 3 preprocessing tasks.

---

**Status:** 📁 Waiting for datasets to be added to repository  
**Next:** Add datasets → Test loading → Begin Week 3 preprocessing

**Created:** February 12, 2026
