# ✅ Week 3 Checklist - Dataset Fusion Part 1

**Goal:** Complete data preprocessing pipeline for all 4 diseases  
**Timeline:** 5 days (Feb 12-16, 2026)  
**Status:** 0/25 tasks completed

---

## 📋 Day 1: Load & Validate Datasets

### Morning (2-3 hours)
- [ ] Create `src/data/load_datasets.py` file
- [ ] Implement `load_all_datasets()` function
- [ ] Load all 4 CSV files successfully
- [ ] Print shape and basic info for each dataset
- [ ] Verify no loading errors

### Afternoon (2-3 hours)
- [ ] Create `notebooks/preprocessing/01_data_loading.ipynb`
- [ ] Document dataset shapes and statistics
- [ ] Identify any data quality issues
- [ ] Calculate basic statistics (mean, std, min, max) per dataset
- [ ] Commit and push Day 1 work

**Expected Output:** All datasets loaded, basic statistics documented

---

## 📋 Day 2: Missing Values

### Morning (2-3 hours)
- [ ] Review EDA notebooks to understand missing value patterns
- [ ] Create `notebooks/preprocessing/02_missing_values.ipynb`
- [ ] Implement `handle_missing_values()` in `src/data/preprocessing.py`
- [ ] Apply median imputation for numeric features
- [ ] Apply mode imputation for categorical features

### Afternoon (2-3 hours)
- [ ] Validate no missing values remain in any dataset
- [ ] Document imputation strategy and rationale
- [ ] Create before/after comparison visualizations
- [ ] Test on all 4 datasets
- [ ] Commit and push Day 2 work

**Expected Output:** Zero missing values in all datasets, strategy documented

---

## 📋 Day 3: Outlier Handling

### Morning (2-3 hours)
- [ ] Create `notebooks/preprocessing/03_outlier_handling.ipynb`
- [ ] Implement IQR-based outlier detection
- [ ] Visualize outliers with box plots
- [ ] Decide on capping strategy (don't remove rows)
- [ ] Implement outlier capping in `src/data/preprocessing.py`

### Afternoon (2-3 hours)
- [ ] Apply outlier handling to all datasets
- [ ] Create before/after distribution plots
- [ ] Validate distributions are reasonable
- [ ] Document outlier handling strategy
- [ ] Commit and push Day 3 work

**Expected Output:** Outliers capped, distributions normalized

---

## 📋 Day 4: Feature Scaling

### Morning (2-3 hours)
- [ ] Create `notebooks/preprocessing/04_feature_scaling.ipynb`
- [ ] Implement StandardScaler in `src/data/preprocessing.py`
- [ ] Scale all numeric features
- [ ] Save scaler objects for each disease
- [ ] Verify scaled distributions (mean≈0, std≈1)

### Afternoon (2-3 hours)
- [ ] Create directory: `models/scalers/`
- [ ] Save all scaler objects as `.pkl` files
- [ ] Document scaling approach
- [ ] Test loading saved scalers
- [ ] Commit and push Day 4 work

**Expected Output:** All features scaled, scalers saved for inference

---

## 📋 Day 5: Train/Val/Test Split

### Morning (2-3 hours)
- [ ] Create `notebooks/preprocessing/05_train_test_split.ipynb`
- [ ] Implement stratified split function in `src/data/preprocessing.py`
- [ ] Create 70/10/20 train/val/test splits
- [ ] Verify stratification maintains class distribution
- [ ] Check split sizes are correct

### Afternoon (2-3 hours)
- [ ] Save all splits to `data/processed/`
- [ ] Create manifest file documenting all saved datasets
- [ ] Update PROGRESS_TRACKER.md (Week 3: 100%)
- [ ] Update README.md status table
- [ ] Final commit: "Week 3 Complete: Data Preprocessing Done"

**Expected Output:** 12 files in `data/processed/` (3 splits × 4 diseases)

---

## 📁 Expected File Structure After Week 3

```
data/processed/
├── heart_train.csv
├── heart_val.csv
├── heart_test.csv
├── diabetes_train.csv
├── diabetes_val.csv
├── diabetes_test.csv
├── kidney_train.csv
├── kidney_val.csv
├── kidney_test.csv
├── liver_train.csv
├── liver_val.csv
├── liver_test.csv
└── preprocessing_manifest.json

models/scalers/
├── heart_scaler.pkl
├── diabetes_scaler.pkl
├── kidney_scaler.pkl
└── liver_scaler.pkl

notebooks/preprocessing/
├── 01_data_loading.ipynb
├── 02_missing_values.ipynb
├── 03_outlier_handling.ipynb
├── 04_feature_scaling.ipynb
└── 05_train_test_split.ipynb

src/data/
├── load_datasets.py (NEW)
├── preprocessing.py (NEW)
└── create_unified_dataset.py (EXISTING)
```

---

## 🎯 Success Metrics

At the end of Week 3, verify:

✅ **Code Quality**
- [ ] All functions have docstrings
- [ ] Code follows PEP 8 style
- [ ] No hardcoded paths (use Path from pathlib)

✅ **Data Quality**
- [ ] Zero missing values across all datasets
- [ ] No extreme outliers remaining
- [ ] All features scaled to similar ranges
- [ ] Class balance maintained in splits

✅ **Documentation**
- [ ] 5 new preprocessing notebooks completed
- [ ] PROGRESS_TRACKER.md updated
- [ ] README.md reflects current status
- [ ] All preprocessing decisions documented with rationale

✅ **Version Control**
- [ ] At least 5 commits (one per day)
- [ ] Meaningful commit messages
- [ ] No temporary files committed

---

## 🚨 Common Pitfalls to Avoid

1. **Don't remove too many rows** - Use imputation over deletion
2. **Don't leak test data** - Scale AFTER splitting, not before
3. **Don't forget to save artifacts** - Scalers, splits, etc.
4. **Don't skip documentation** - Your future self will thank you
5. **Don't work on all datasets at once** - Validate on one, then scale to all

---

## 📞 Getting Unstuck

**If something fails:**
1. Check your EDA notebooks for insights
2. Print intermediate results to debug
3. Test on a small sample first (e.g., 100 rows)
4. Verify file paths are correct
5. Check for typos in column names

**If unsure about strategy:**
1. Review similar projects on GitHub
2. Check scikit-learn examples
3. Consult your thesis papers
4. Document your decision and move forward

---

## ⏰ Daily Time Commitment

- **Morning:** 2-3 hours of coding
- **Afternoon:** 2-3 hours of documentation & validation
- **Total:** 4-6 hours per day
- **Week Total:** 20-30 hours

---

## 🎓 Thesis Connection

As you work, think about:

**For Methodology Section:**
- WHY you chose each preprocessing technique
- HOW it addresses specific data challenges
- WHAT alternatives you considered

**For Results Section:**
- Statistics before/after preprocessing
- Impact on data distributions
- Validation that preprocessing improved model readiness

---

## 📸 Screenshots to Capture

For your thesis and presentation:
- [ ] Before/after missing value handling
- [ ] Outlier detection box plots
- [ ] Feature distributions after scaling
- [ ] Class distribution in train/val/test splits

---

**Start Now:** Open your IDE, create `src/data/load_datasets.py`, and begin Day 1!

**Daily Habit:** At end of each day, update this checklist and commit your work.

**Week 3 Target Completion:** February 16, 2026

---

*Stay focused, work systematically, document thoroughly. You've got this! 💪*
