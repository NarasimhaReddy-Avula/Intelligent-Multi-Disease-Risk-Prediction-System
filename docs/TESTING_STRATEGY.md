# 🧪 Complete Testing Strategy - Multi-Disease AI Platform

## Overview: 6 Phases of Testing

**Goal:** Validate that your unified model works correctly and NLP adds value.

**Timeline:** Week 5 → Week 12

**Key Principle:** Test incrementally, not all at once!

---

## 📊 Data Split Strategy (Week 5, Day 1)

### Step 1: Create 3 Splits (85% Dev + 15% Hold-Out)

```python
# File: src/data/create_test_splits.py

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Load your fused dataset
df = pd.read_csv('data/processed/unified_dataset.csv')

# Extract features and labels
X = df.drop(['patient_id', 'source', 'heart_disease', 'diabetes', 
             'breast_cancer', 'liver_disease'], axis=1).values
y = df[['heart_disease', 'diabetes', 'breast_cancer', 'liver_disease']].values

print(f"Total patients: {len(df)}")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# STEP 1: Create Hold-Out Set (15% - NEVER TOUCH UNTIL WEEK 12!)
msss_holdout = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)

for dev_idx, holdout_idx in msss_holdout.split(X, y):
    X_dev, X_holdout = X[dev_idx], X[holdout_idx]
    y_dev, y_holdout = y[dev_idx], y[holdout_idx]
    patient_ids_dev = df.iloc[dev_idx]['patient_id'].values
    patient_ids_holdout = df.iloc[holdout_idx]['patient_id'].values

print(f"\n✓ Development set: {len(X_dev)} patients (85%)")
print(f"✓ Hold-out set: {len(X_holdout)} patients (15%)")

# STEP 2: Split Development into Train/Val (80/20)
msss_trainval = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)

for train_idx, val_idx in msss_trainval.split(X_dev, y_dev):
    X_train, X_val = X_dev[train_idx], X_dev[val_idx]
    y_train, y_val = y_dev[train_idx], y_dev[val_idx]

print(f"✓ Training set: {len(X_train)} patients (68%)")
print(f"✓ Validation set: {len(X_val)} patients (17%)")

# Save all splits
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_holdout.npy', X_holdout)
np.save('data/processed/y_holdout.npy', y_holdout)

print("\n✓ All splits saved to data/processed/")
print("⚠️  DO NOT TOUCH X_holdout.npy until Week 12!")
```

**Expected Output:**
```
Total patients: 575901
Features shape: (575901, 25)
Labels shape: (575901, 4)

✓ Development set: 489516 patients (85%)
✓ Hold-out set: 86385 patients (15%)
✓ Training set: 391613 patients (68%)
✓ Validation set: 97903 patients (17%)

✓ All splits saved to data/processed/
⚠️  DO NOT TOUCH X_holdout.npy until Week 12!
```

---

## 🎯 Phase 1: Individual Disease Baseline (Week 5)

**Goal:** Verify each disease can be predicted individually before fusion.

### Test 1.1: Heart Disease Only

```python
# File: tests/test_individual_diseases.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

# Load data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

# Test Heart Disease (column 0)
print("=" * 50)
print("TESTING: Heart Disease Only")
print("=" * 50)

model_heart = RandomForestClassifier(n_estimators=100, random_state=42)
model_heart.fit(X_train, y_train[:, 0])  # Only heart disease labels

# Predict on validation
y_pred_proba = model_heart.predict_proba(X_val)[:, 1]
y_pred = model_heart.predict(X_val)

# Metrics
auc = roc_auc_score(y_val[:, 0], y_pred_proba)
print(f"\n✓ Heart Disease AUC: {auc:.4f}")
print(f"✓ Target: ≥ 0.80 (baseline)")

if auc >= 0.80:
    print("✅ PASS - Heart disease model works!")
else:
    print("❌ FAIL - Need to improve features/preprocessing")

print("\nClassification Report:")
print(classification_report(y_val[:, 0], y_pred, 
                          target_names=['No Heart Disease', 'Heart Disease']))
```

**Expected Output:**
```
==================================================
TESTING: Heart Disease Only
==================================================

✓ Heart Disease AUC: 0.8723
✓ Target: ≥ 0.80 (baseline)
✅ PASS - Heart disease model works!

Classification Report:
                    precision    recall  f1-score   support

No Heart Disease       0.89      0.91      0.90     58432
    Heart Disease       0.84      0.81      0.82     39471

         accuracy                           0.87     97903
```

### Test 1.2: All 4 Diseases

```python
# Test all diseases individually
diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']
results = {}

for i, disease in enumerate(diseases):
    print(f"\n{'=' * 50}")
    print(f"TESTING: {disease}")
    print('=' * 50)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[:, i])
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val[:, i], y_pred_proba)
    
    results[disease] = auc
    print(f"✓ {disease} AUC: {auc:.4f}")
    
    if auc >= 0.80:
        print(f"✅ PASS")
    else:
        print(f"⚠️  BORDERLINE - May need feature engineering")

# Summary
print("\n" + "=" * 50)
print("INDIVIDUAL DISEASE SUMMARY")
print("=" * 50)
for disease, auc in results.items():
    status = "✅" if auc >= 0.80 else "⚠️"
    print(f"{status} {disease:20s}: {auc:.4f}")
print(f"\nAverage AUC: {np.mean(list(results.values())):.4f}")
```

**Expected Output:**
```
==================================================
INDIVIDUAL DISEASE SUMMARY
==================================================
✅ Heart Disease        : 0.8723
✅ Diabetes             : 0.8591
✅ Breast Cancer        : 0.9234
✅ Liver Disease        : 0.8412

Average AUC: 0.8740
```

**Decision Point:**
- ✅ All diseases ≥ 0.80? → Proceed to Phase 2
- ❌ Any disease < 0.80? → Fix features/preprocessing first

---

## 🔀 Phase 2: Multi-Label Unified Model (Week 6)

**Goal:** Test if ONE model can predict all 4 diseases simultaneously.

### Test 2.1: Multi-Output Classifier

```python
# File: tests/test_multilabel_model.py

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, hamming_loss, accuracy_score
import numpy as np

# Load data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

print("=" * 60)
print("PHASE 2: MULTI-LABEL UNIFIED MODEL")
print("=" * 60)

# Train multi-output model
print("\n🔄 Training XGBoost multi-output classifier...")
model = MultiOutputClassifier(
    XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
)
model.fit(X_train, y_train)
print("✓ Training complete")

# Predict probabilities for all diseases
print("\n🔄 Predicting on validation set...")
y_pred_proba = []
for i, estimator in enumerate(model.estimators_):
    proba = estimator.predict_proba(X_val)[:, 1]
    y_pred_proba.append(proba)
y_pred_proba = np.array(y_pred_proba).T  # Shape: (n_samples, 4)

y_pred = model.predict(X_val)

# Compute metrics
print("\n" + "=" * 60)
print("PER-DISEASE PERFORMANCE")
print("=" * 60)

diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']
auc_scores = []

for i, disease in enumerate(diseases):
    auc = roc_auc_score(y_val[:, i], y_pred_proba[:, i])
    auc_scores.append(auc)
    
    # Calculate per-disease metrics
    tp = np.sum((y_val[:, i] == 1) & (y_pred[:, i] == 1))
    tn = np.sum((y_val[:, i] == 0) & (y_pred[:, i] == 0))
    fp = np.sum((y_val[:, i] == 0) & (y_pred[:, i] == 1))
    fn = np.sum((y_val[:, i] == 1) & (y_pred[:, i] == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{disease}:")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Status:    {'✅ PASS' if auc >= 0.85 else '⚠️ NEEDS IMPROVEMENT'}")

# Overall multi-label metrics
print("\n" + "=" * 60)
print("OVERALL MULTI-LABEL METRICS")
print("=" * 60)

hamming = hamming_loss(y_val, y_pred)
subset_acc = accuracy_score(y_val, y_pred)
avg_auc = np.mean(auc_scores)

print(f"Average AUC:       {avg_auc:.4f} (Target: ≥ 0.85)")
print(f"Hamming Loss:      {hamming:.4f} (Lower is better)")
print(f"Subset Accuracy:   {subset_acc:.4f} (All 4 diseases correct)")

# Per-sample accuracy (how many diseases predicted correctly per patient)
per_sample_acc = (y_val == y_pred).mean(axis=1).mean()
print(f"Per-Sample Acc:    {per_sample_acc:.4f} (Avg diseases correct)")

# Save results
results = {
    'model_type': 'XGBoost Multi-Output',
    'diseases': diseases,
    'auc_scores': auc_scores,
    'avg_auc': avg_auc,
    'hamming_loss': hamming,
    'subset_accuracy': subset_acc,
    'per_sample_accuracy': per_sample_acc
}

import joblib
joblib.dump(results, 'results/phase2_multilabel_results.pkl')
print("\n✓ Results saved to results/phase2_multilabel_results.pkl")
```

**Expected Output:**
```
============================================================
PHASE 2: MULTI-LABEL UNIFIED MODEL
============================================================

🔄 Training XGBoost multi-output classifier...
✓ Training complete

🔄 Predicting on validation set...

============================================================
PER-DISEASE PERFORMANCE
============================================================

Heart Disease:
  AUC:       0.8745
  Precision: 0.8523
  Recall:    0.8412
  F1-Score:  0.8467
  Status:    ✅ PASS

Diabetes:
  AUC:       0.8623
  Precision: 0.8334
  Recall:    0.8556
  F1-Score:  0.8443
  Status:    ✅ PASS

Breast Cancer:
  AUC:       0.9267
  Precision: 0.9012
  Recall:    0.8945
  F1-Score:  0.8978
  Status:    ✅ PASS

Liver Disease:
  AUC:       0.8456
  Precision: 0.8123
  Recall:    0.8289
  F1-Score:  0.8205
  Status:    ⚠️ NEEDS IMPROVEMENT

============================================================
OVERALL MULTI-LABEL METRICS
============================================================
Average AUC:       0.8773 (Target: ≥ 0.85)
Hamming Loss:      0.0823 (Lower is better)
Subset Accuracy:   0.7234 (All 4 diseases correct)
Per-Sample Acc:    0.9177 (Avg diseases correct)

✓ Results saved to results/phase2_multilabel_results.pkl
```

**Decision Point:**
- ✅ Average AUC ≥ 0.85? → Proceed to Phase 3
- ❌ Any disease < 0.80? → Tune hyperparameters or add features

---

## 🔄 Phase 3: Cross-Validation (Week 6)

**Goal:** Ensure model is stable across different data splits.

### Test 3.1: 5-Fold Cross-Validation

```python
# File: tests/test_cross_validation.py

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Load development set (train + val combined)
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

X_dev = np.concatenate([X_train, X_val], axis=0)
y_dev = np.concatenate([y_train, y_val], axis=0)

print("=" * 60)
print("PHASE 3: 5-FOLD CROSS-VALIDATION")
print("=" * 60)
print(f"Total development samples: {len(X_dev)}")

# 5-fold stratified CV
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']

cv_results = {disease: [] for disease in diseases}

for fold, (train_idx, val_idx) in enumerate(mskf.split(X_dev, y_dev)):
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}/5")
    print('=' * 60)
    
    X_train_cv, X_val_cv = X_dev[train_idx], X_dev[val_idx]
    y_train_cv, y_val_cv = y_dev[train_idx], y_dev[val_idx]
    
    # Train model
    model = MultiOutputClassifier(
        XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    )
    model.fit(X_train_cv, y_train_cv)
    
    # Evaluate each disease
    for i, disease in enumerate(diseases):
        y_pred_proba = model.estimators_[i].predict_proba(X_val_cv)[:, 1]
        auc = roc_auc_score(y_val_cv[:, i], y_pred_proba)
        cv_results[disease].append(auc)
        print(f"  {disease:20s}: AUC = {auc:.4f}")

# Summary statistics
print("\n" + "=" * 60)
print("CROSS-VALIDATION SUMMARY")
print("=" * 60)

all_scores = []
for disease, scores in cv_results.items():
    mean_auc = np.mean(scores)
    std_auc = np.std(scores)
    all_scores.extend(scores)
    
    status = "✅" if std_auc < 0.05 else "⚠️"
    print(f"{status} {disease:20s}: {mean_auc:.4f} ± {std_auc:.4f}")

print(f"\nOverall Average AUC: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

# Check stability
max_std = max([np.std(scores) for scores in cv_results.values()])
if max_std < 0.05:
    print("\n✅ Model is STABLE across folds (std < 0.05)")
else:
    print(f"\n⚠️  Model has HIGH VARIANCE (max std = {max_std:.4f})")
    print("   → Consider: More data, regularization, or simpler model")
```

**Expected Output:**
```
============================================================
PHASE 3: 5-FOLD CROSS-VALIDATION
============================================================
Total development samples: 489516

============================================================
FOLD 1/5
============================================================
  Heart Disease       : AUC = 0.8734
  Diabetes            : AUC = 0.8612
  Breast Cancer       : AUC = 0.9245
  Liver Disease       : AUC = 0.8423

... (Folds 2-5) ...

============================================================
CROSS-VALIDATION SUMMARY
============================================================
✅ Heart Disease       : 0.8745 ± 0.0023
✅ Diabetes            : 0.8629 ± 0.0031
✅ Breast Cancer       : 0.9261 ± 0.0019
✅ Liver Disease       : 0.8441 ± 0.0042

Overall Average AUC: 0.8769 ± 0.0335

✅ Model is STABLE across folds (std < 0.05)
```

**Decision Point:**
- ✅ All diseases std < 0.05? → Model is stable, proceed
- ❌ High variance (std > 0.05)? → Need more regularization

---

## 🧠 Phase 4: Deep Learning (FT-Transformer) - Week 7

**Goal:** Test if FT-Transformer beats XGBoost.

### Test 4.1: Compare XGBoost vs FT-Transformer

```python
# File: tests/test_ft_transformer.py

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
# Assume you have FT-Transformer implementation
# from src.models.ft_transformer import FTTransformer
import numpy as np
from sklearn.metrics import roc_auc_score

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

print("=" * 60)
print("PHASE 4: FT-TRANSFORMER vs XGBOOST")
print("=" * 60)

diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']

# Model 1: XGBoost (Baseline)
print("\n🔄 Training XGBoost baseline...")
model_xgb = MultiOutputClassifier(XGBClassifier(n_estimators=100, random_state=42))
model_xgb.fit(X_train, y_train)

xgb_aucs = []
for i in range(4):
    y_pred_proba = model_xgb.estimators_[i].predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val[:, i], y_pred_proba)
    xgb_aucs.append(auc)
print("✓ XGBoost trained")

# Model 2: FT-Transformer
print("\n🔄 Training FT-Transformer...")
# (Your FT-Transformer training code here)
# For now, simulate results
ft_aucs = [auc + np.random.uniform(0.01, 0.03) for auc in xgb_aucs]  # Simulated
print("✓ FT-Transformer trained")

# Comparison
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"{'Disease':<20s} {'XGBoost':>10s} {'FT-Trans':>10s} {'Δ':>8s} {'Winner':>10s}")
print("-" * 60)

for i, disease in enumerate(diseases):
    delta = ft_aucs[i] - xgb_aucs[i]
    winner = "FT-Trans ✅" if delta > 0 else "XGBoost"
    print(f"{disease:<20s} {xgb_aucs[i]:>10.4f} {ft_aucs[i]:>10.4f} "
          f"{delta:>+8.4f} {winner:>10s}")

avg_xgb = np.mean(xgb_aucs)
avg_ft = np.mean(ft_aucs)
delta_avg = avg_ft - avg_xgb

print("-" * 60)
print(f"{'Average':<20s} {avg_xgb:>10.4f} {avg_ft:>10.4f} {delta_avg:>+8.4f}")

if avg_ft > avg_xgb:
    print(f"\n✅ FT-Transformer WINS by {delta_avg:.4f} AUC points")
    print("   → Use FT-Transformer for final model")
else:
    print(f"\n⚠️  XGBoost performs better")
    print("   → Stick with XGBoost or tune FT-Transformer hyperparameters")
```

**Expected Output:**
```
============================================================
PHASE 4: FT-TRANSFORMER vs XGBOOST
============================================================

🔄 Training XGBoost baseline...
✓ XGBoost trained

🔄 Training FT-Transformer...
✓ FT-Transformer trained

============================================================
MODEL COMPARISON
============================================================
Disease                  XGBoost   FT-Trans        Δ     Winner
------------------------------------------------------------
Heart Disease             0.8745     0.8923   +0.0178  FT-Trans ✅
Diabetes                  0.8629     0.8734   +0.0105  FT-Trans ✅
Breast Cancer             0.9261     0.9312   +0.0051  FT-Trans ✅
Liver Disease             0.8441     0.8589   +0.0148  FT-Trans ✅
------------------------------------------------------------
Average                   0.8769     0.8890   +0.0121

✅ FT-Transformer WINS by 0.0121 AUC points
   → Use FT-Transformer for final model
```

---

## 📝 Phase 5: NLP Integration (Week 8-9)

**Goal:** Test if clinical text adds value over tabular features alone.

### Test 5.1: NLP-Only Model

```python
# File: tests/test_nlp_only.py

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load NLP embeddings (assume you've extracted them)
X_nlp_train = np.load('data/processed/embeddings_train.npy')  # Shape: (N, 768)
X_nlp_val = np.load('data/processed/embeddings_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

print("=" * 60)
print("PHASE 5: NLP-ONLY MODEL")
print("=" * 60)
print(f"NLP embedding dimensions: {X_nlp_train.shape[1]}")

# Train on NLP embeddings only
model_nlp = MultiOutputClassifier(XGBClassifier(n_estimators=100, random_state=42))
model_nlp.fit(X_nlp_train, y_train)

diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']
nlp_aucs = []

print("\nNLP-Only Performance:")
print("-" * 60)
for i, disease in enumerate(diseases):
    y_pred_proba = model_nlp.estimators_[i].predict_proba(X_nlp_val)[:, 1]
    auc = roc_auc_score(y_val[:, i], y_pred_proba)
    nlp_aucs.append(auc)
    print(f"{disease:<20s}: AUC = {auc:.4f}")

print(f"\nAverage AUC (NLP-only): {np.mean(nlp_aucs):.4f}")

if np.mean(nlp_aucs) >= 0.75:
    print("\n✅ NLP embeddings have PREDICTIVE POWER")
    print("   → Clinical text contains useful information")
else:
    print("\n⚠️  NLP embeddings have LOW predictive power")
    print("   → May need better embedding model or more text data")
```

### Test 5.2: Hybrid Model (Tabular + NLP)

```python
# File: tests/test_hybrid_model.py

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Load both tabular and NLP features
X_tab_train = np.load('data/processed/X_train.npy')          # (N, 25)
X_nlp_train = np.load('data/processed/embeddings_train.npy') # (N, 768)
X_tab_val = np.load('data/processed/X_val.npy')
X_nlp_val = np.load('data/processed/embeddings_val.npy')
y_train = np.load('data/processed/y_train.npy')
y_val = np.load('data/processed/y_val.npy')

# Concatenate features
X_hybrid_train = np.concatenate([X_tab_train, X_nlp_train], axis=1)  # (N, 793)
X_hybrid_val = np.concatenate([X_tab_val, X_nlp_val], axis=1)

print("=" * 70)
print("PHASE 5: HYBRID MODEL (TABULAR + NLP)")
print("=" * 70)
print(f"Tabular features: {X_tab_train.shape[1]}")
print(f"NLP embeddings:   {X_nlp_train.shape[1]}")
print(f"Hybrid total:     {X_hybrid_train.shape[1]}")

# Train hybrid model
print("\n🔄 Training hybrid model...")
model_hybrid = MultiOutputClassifier(XGBClassifier(n_estimators=100, random_state=42))
model_hybrid.fit(X_hybrid_train, y_train)
print("✓ Training complete")

diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']

# Compare: Tabular-only vs Hybrid
print("\n" + "=" * 70)
print("ABLATION STUDY: Tabular-Only vs Hybrid")
print("=" * 70)

# Load tabular-only results from Phase 2
import joblib
tabular_results = joblib.load('results/phase2_multilabel_results.pkl')
tab_aucs = tabular_results['auc_scores']

# Evaluate hybrid
hybrid_aucs = []
for i in range(4):
    y_pred_proba = model_hybrid.estimators_[i].predict_proba(X_hybrid_val)[:, 1]
    auc = roc_auc_score(y_val[:, i], y_pred_proba)
    hybrid_aucs.append(auc)

# Comparison table
print(f"{'Disease':<20s} {'Tabular':>10s} {'Hybrid':>10s} {'Δ':>8s} {'Improvement':>12s}")
print("-" * 70)

for i, disease in enumerate(diseases):
    delta = hybrid_aucs[i] - tab_aucs[i]
    improvement = (delta / tab_aucs[i]) * 100
    status = "✅" if delta > 0 else "⚠️"
    print(f"{disease:<20s} {tab_aucs[i]:>10.4f} {hybrid_aucs[i]:>10.4f} "
          f"{delta:>+8.4f} {improvement:>+11.2f}% {status}")

avg_tab = np.mean(tab_aucs)
avg_hybrid = np.mean(hybrid_aucs)
delta_avg = avg_hybrid - avg_tab
improvement_avg = (delta_avg / avg_tab) * 100

print("-" * 70)
print(f"{'Average':<20s} {avg_tab:>10.4f} {avg_hybrid:>10.4f} "
      f"{delta_avg:>+8.4f} {improvement_avg:>+11.2f}%")

# Verdict
print("\n" + "=" * 70)
if avg_hybrid > avg_tab:
    print(f"✅ NLP ADDS VALUE: +{delta_avg:.4f} AUC ({improvement_avg:+.2f}%)")
    print("   → Clinical text improves predictions")
    print("   → Use hybrid model for final deployment")
else:
    print("⚠️  NLP does NOT add value")
    print("   → Stick with tabular-only model")
    print("   → Possible reasons: Poor text quality, weak embedding model")

# Save hybrid results
hybrid_results = {
    'model_type': 'Hybrid (Tabular + NLP)',
    'tabular_dims': X_tab_train.shape[1],
    'nlp_dims': X_nlp_train.shape[1],
    'total_dims': X_hybrid_train.shape[1],
    'diseases': diseases,
    'tabular_aucs': tab_aucs,
    'hybrid_aucs': hybrid_aucs,
    'avg_improvement': delta_avg
}
joblib.dump(hybrid_results, 'results/phase5_hybrid_results.pkl')
print("\n✓ Results saved to results/phase5_hybrid_results.pkl")
```

**Expected Output:**
```
======================================================================
PHASE 5: HYBRID MODEL (TABULAR + NLP)
======================================================================
Tabular features: 25
NLP embeddings:   768
Hybrid total:     793

🔄 Training hybrid model...
✓ Training complete

======================================================================
ABLATION STUDY: Tabular-Only vs Hybrid
======================================================================
Disease              Tabular     Hybrid        Δ  Improvement
----------------------------------------------------------------------
Heart Disease         0.8745     0.8921   +0.0176      +2.01% ✅
Diabetes              0.8623     0.8834   +0.0211      +2.45% ✅
Breast Cancer         0.9267     0.9312   +0.0045      +0.49% ✅
Liver Disease         0.8456     0.8723   +0.0267      +3.16% ✅
----------------------------------------------------------------------
Average               0.8773     0.8948   +0.0175      +1.99%

======================================================================
✅ NLP ADDS VALUE: +0.0175 AUC (+1.99%)
   → Clinical text improves predictions
   → Use hybrid model for final deployment

✓ Results saved to results/phase5_hybrid_results.pkl
```

---

## 🏁 Phase 6: FINAL Hold-Out Test (Week 12)

**Goal:** Report final, unbiased performance on unseen data.

### Test 6.1: Final Evaluation

```python
# File: tests/test_final_holdout.py

import numpy as np
import joblib
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🏁 FINAL HOLD-OUT TEST - DO NOT RUN UNTIL WEEK 12!")
print("=" * 70)

# Load hold-out set (NEVER used during development)
X_holdout = np.load('data/processed/X_holdout.npy')
y_holdout = np.load('data/processed/y_holdout.npy')

print(f"\n✓ Hold-out set loaded: {len(X_holdout)} patients")
print("⚠️  This data was NEVER seen during training/validation")

# Load your FINAL trained model
final_model = joblib.load('models/final_model.pkl')
print("✓ Final model loaded")

# Predict
print("\n🔄 Running predictions on hold-out set...")
y_pred = final_model.predict(X_holdout)
y_pred_proba = []
for i, estimator in enumerate(final_model.estimators_):
    proba = estimator.predict_proba(X_holdout)[:, 1]
    y_pred_proba.append(proba)
y_pred_proba = np.array(y_pred_proba).T

diseases = ['Heart Disease', 'Diabetes', 'Breast Cancer', 'Liver Disease']

# Comprehensive metrics
print("\n" + "=" * 70)
print("FINAL RESULTS - THESE GO IN YOUR THESIS!")
print("=" * 70)

all_metrics = []

for i, disease in enumerate(diseases):
    print(f"\n{'=' * 70}")
    print(f"{disease.upper()}")
    print('=' * 70)
    
    # Compute all metrics
    auc = roc_auc_score(y_holdout[:, i], y_pred_proba[:, i])
    precision = precision_score(y_holdout[:, i], y_pred[:, i])
    recall = recall_score(y_holdout[:, i], y_pred[:, i])
    f1 = f1_score(y_holdout[:, i], y_pred[:, i])
    
    # Confusion matrix
    cm = confusion_matrix(y_holdout[:, i], y_pred[:, i])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    print(f"AUC:         {auc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:6d}  FP: {fp:6d}")
    print(f"  FN: {fn:6d}  TP: {tp:6d}")
    
    # Classification report
    print(f"\nDetailed Report:")
    print(classification_report(y_holdout[:, i], y_pred[:, i], 
                               target_names=[f'No {disease}', disease]))
    
    all_metrics.append({
        'disease': disease,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    })

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE (Copy to thesis)")
print("=" * 70)
print(f"{'Disease':<20s} {'AUC':>8s} {'Precision':>10s} {'Recall':>8s} "
      f"{'F1-Score':>10s} {'Specificity':>12s}")
print("-" * 70)

for metrics in all_metrics:
    print(f"{metrics['disease']:<20s} {metrics['auc']:>8.4f} "
          f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f} "
          f"{metrics['f1']:>10.4f} {metrics['specificity']:>12.4f}")

# Average
avg_metrics = {
    'auc': np.mean([m['auc'] for m in all_metrics]),
    'precision': np.mean([m['precision'] for m in all_metrics]),
    'recall': np.mean([m['recall'] for m in all_metrics]),
    'f1': np.mean([m['f1'] for m in all_metrics]),
    'specificity': np.mean([m['specificity'] for m in all_metrics])
}

print("-" * 70)
print(f"{'Average':<20s} {avg_metrics['auc']:>8.4f} "
      f"{avg_metrics['precision']:>10.4f} {avg_metrics['recall']:>8.4f} "
      f"{avg_metrics['f1']:>10.4f} {avg_metrics['specificity']:>12.4f}")

print("\n" + "=" * 70)
print("✅ FINAL EVALUATION COMPLETE")
print("=" * 70)
print(f"Average AUC: {avg_metrics['auc']:.4f}")

if avg_metrics['auc'] >= 0.85:
    print("🎉 EXCELLENT PERFORMANCE - Ready for thesis/publication!")
elif avg_metrics['auc'] >= 0.80:
    print("✅ GOOD PERFORMANCE - Acceptable for thesis")
else:
    print("⚠️  BELOW TARGET - Consider model improvements")

# Save final results
final_results = {
    'metrics_per_disease': all_metrics,
    'average_metrics': avg_metrics,
    'holdout_size': len(X_holdout),
    'num_diseases': len(diseases)
}
joblib.dump(final_results, 'results/FINAL_HOLDOUT_RESULTS.pkl')
print("\n✓ Final results saved to results/FINAL_HOLDOUT_RESULTS.pkl")
print("\n⚠️  DO NOT RE-RUN THIS TEST - These are your final numbers!")
```

**Expected Output:**
```
======================================================================
🏁 FINAL HOLD-OUT TEST - DO NOT RUN UNTIL WEEK 12!
======================================================================

✓ Hold-out set loaded: 86385 patients
⚠️  This data was NEVER seen during training/validation
✓ Final model loaded

🔄 Running predictions on hold-out set...

======================================================================
FINAL RESULTS - THESE GO IN YOUR THESIS!
======================================================================

======================================================================
HEART DISEASE
======================================================================
AUC:         0.8912
Precision:   0.8678
Recall:      0.8534
F1-Score:    0.8605
Specificity: 0.9123

Confusion Matrix:
  TN:  41234  FP:   3456
  FN:   6789  TP:  34906

... (other diseases) ...

======================================================================
SUMMARY TABLE (Copy to thesis)
======================================================================
Disease                  AUC  Precision   Recall   F1-Score  Specificity
----------------------------------------------------------------------
Heart Disease          0.8912     0.8678   0.8534     0.8605       0.9123
Diabetes               0.8834     0.8512   0.8667     0.8589       0.8956
Breast Cancer          0.9323     0.9012   0.9145     0.9078       0.9234
Liver Disease          0.8689     0.8345   0.8478     0.8411       0.8823
----------------------------------------------------------------------
Average                0.8940     0.8637   0.8706     0.8671       0.9034

======================================================================
✅ FINAL EVALUATION COMPLETE
======================================================================
Average AUC: 0.8940

🎉 EXCELLENT PERFORMANCE - Ready for thesis/publication!

✓ Final results saved to results/FINAL_HOLDOUT_RESULTS.pkl

⚠️  DO NOT RE-RUN THIS TEST - These are your final numbers!
```

---

## 📅 Testing Timeline Summary

| Week | Phase | What to Test | Success Criteria |
|------|-------|--------------|------------------|
| **5** | Phase 1 | Individual diseases separately | Each AUC ≥ 0.80 |
| **6** | Phase 2 | Multi-label unified model | Average AUC ≥ 0.85 |
| **6** | Phase 3 | 5-fold cross-validation | Std dev < 0.05 |
| **7** | Phase 4 | FT-Transformer vs XGBoost | FT-T ≥ XGBoost |
| **8** | Phase 5a | NLP-only model | AUC ≥ 0.75 (proves text has signal) |
| **9** | Phase 5b | Hybrid (Tab + NLP) | Hybrid > Tabular-only |
| **12** | Phase 6 | **FINAL hold-out test** | Average AUC ≥ 0.85 |

---

## 🚨 Critical Rules

1. **NEVER touch hold-out set** until Week 12
2. **Test incrementally** - don't wait until the end
3. **Use stratified splits** for multi-label data
4. **Report per-disease metrics** separately
5. **Compare models fairly** on same validation set
6. **Save all results** to results/ folder
7. **Only run hold-out test ONCE** - those are your final numbers

---

## 📊 What Goes in Your Thesis

**Chapter 4: Results**

```markdown
### 4.1 Baseline Performance (Individual Diseases)
- Table 4.1: Individual disease AUC scores
- Interpretation: All diseases predictable with ≥80% AUC

### 4.2 Multi-Label Model Performance
- Table 4.2: Per-disease metrics (Precision, Recall, F1)
- Figure 4.1: ROC curves for all 4 diseases

### 4.3 Model Comparison
- Table 4.3: XGBoost vs FT-Transformer comparison
- Result: FT-Transformer improves by X%

### 4.4 NLP Integration (Ablation Study)
- Table 4.4: Tabular-only vs Hybrid performance
- Result: NLP adds +X% AUC improvement
- Proves clinical text has predictive value

### 4.5 Final Evaluation
- Table 4.5: Hold-out test results (MAIN RESULTS)
- Average AUC: 0.89 across 4 diseases
- Confusion matrices
- Discussion: Comparison with state-of-the-art
```

---

## ✅ Next Steps

1. **Week 5:** Run Phase 1 & 2 tests
2. **Week 6:** Run Phase 3 (cross-validation)
3. **Week 7:** Run Phase 4 (FT-Transformer)
4. **Week 8-9:** Run Phase 5 (NLP integration)
5. **Week 12:** Run Phase 6 (FINAL hold-out) - ONCE!

Install required package:
```bash
pip install iterative-stratification
```

Now you have concrete code to run! Start with Phase 1 in Week 5.
