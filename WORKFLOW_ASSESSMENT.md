# 🔄 Workflow Assessment & Risk Mitigation

> **Last Updated:** January 2025  
> **Purpose:** Document potential workflow issues and contingency plans

---

## 📊 Risk Assessment Matrix

| Risk | Probability | Impact | Priority | Mitigation |
|------|-------------|--------|----------|------------|
| Dataset class imbalance | High | High | 🔴 Critical | SMOTE, class weights |
| GPU/compute limitations | Medium | High | 🟠 High | Cloud GPU fallback |
| MIMIC-III access delay | Medium | Medium | 🟡 Medium | Defer to Semester 2 |
| Model overfitting | High | High | 🔴 Critical | Early stopping, CV |
| API performance issues | Low | Medium | 🟢 Low | Caching, optimization |
| LLM API costs | Medium | Low | 🟡 Medium | Local models (Llama) |

---

## 🚨 Potential Issues & Solutions

### 1. Dataset Issues

#### Issue 1.1: Class Imbalance
**Status:** ⚠️ Expected for all datasets

**Datasets at risk:**
- Heart Disease (`cardio`): ~50% balanced (good!)
- Diabetes (`Diabetes_012`): 3-class problem (0=no diabetes, 1=prediabetes, 2=diabetes) - potential imbalance
- Stroke (`stroke`): Typically ~5% positive class (SEVERE imbalance)
- Liver Disease (`Dataset`): ~71% liver disease positive (moderate imbalance)

**Mitigation Strategies:**
1. **SMOTE (Synthetic Minority Over-sampling)**
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

2. **Class Weights in Models**
   ```python
   # XGBoost
   scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
   
   # PyTorch
   class_weights = torch.tensor([1.0, weight_positive])
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```

3. **Stratified Sampling** for train/test splits
   ```python
   from sklearn.model_selection import StratifiedKFold
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   ```

4. **Threshold Optimization** for final predictions
   - Use precision-recall curves instead of ROC
   - Optimize threshold for F1-score

---

#### Issue 1.2: Missing Values
**Status:** ⚠️ Known issue in some datasets

**Expected missing data:**
- Heart Disease: `education`, `BPMeds`, `glucose` may have missing values
- Liver Disease: `Albumin_and_Globulin_Ratio` often has NaN
- Diabetes: Has some zero values that represent missing (Glucose, BMI, etc.)

**Mitigation Strategies:**
1. **For Heart/Liver:**
   ```python
   # Simple imputation
   from sklearn.impute import SimpleImputer
   imputer = SimpleImputer(strategy='median')
   
   # Advanced: KNN Imputation
   from sklearn.impute import KNNImputer
   imputer = KNNImputer(n_neighbors=5)
   ```

2. **For Diabetes (zero = missing):**
   ```python
   # Replace zeros with NaN, then impute
   cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
   df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
   ```

3. **Document imputation in thesis** - important for reproducibility

---

#### Issue 1.3: Feature Scale Differences
**Problem:** Features have vastly different scales across datasets

**Solution:**
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# Use RobustScaler for outlier-heavy data
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 2. Model Development Issues

#### Issue 2.1: Overfitting Risk
**Status:** � LOW risk - Large datasets available (70K heart, 253K diabetes)

**Small dataset concern:** Only Liver (583 samples) requires extra attention

**Mitigation:**
1. **Aggressive Cross-Validation for Liver dataset**
   ```python
   # Use 10-fold CV for small datasets
   from sklearn.model_selection import cross_val_score
   cv_scores = cross_val_score(model, X, y, cv=10, scoring='roc_auc')
   ```

2. **Standard CV for large datasets**
   ```python
   # 5-fold CV sufficient for 70K+ samples
   cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
   ```

2. **Early Stopping for Deep Learning**
   ```python
   early_stopping = EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True
   )
   ```

3. **Strong Regularization**
   ```python
   # XGBoost
   params = {
       'reg_alpha': 0.1,      # L1 regularization
       'reg_lambda': 1.0,     # L2 regularization
       'max_depth': 5,        # Limit tree depth
       'min_child_weight': 5  # Minimum samples in leaf
   }
   ```

4. **Data Augmentation** (if applicable)

---

#### Issue 2.2: FT-Transformer Complexity
**Problem:** FT-Transformer may be overkill for small datasets (Liver: 583)

**Updated Strategy:**
1. **Prioritize by dataset size:**
   - Heart (70K): FT-Transformer likely beneficial
   - Diabetes (253K): FT-Transformer highly recommended
   - Stroke (5K): Test FT-Transformer, compare with XGBoost
   - Liver (583): Skip FT-Transformer, use XGBoost + strong regularization

**Decision Point:** Week 8 - evaluate if FT-Transformer adds >3% improvement over XGBoost

---

#### Issue 2.3: Multi-Task Learning Challenges
**Problem:** Joint training for 4 diseases may cause negative transfer

**Mitigation:**
1. **Start with separate models** - establish baselines
2. **Gradual integration:**
   - Shared encoder, separate heads
   - Weighted loss function per disease
3. **Task weighting:**
   ```python
   total_loss = w1*loss_heart + w2*loss_diabetes + w3*loss_cancer + w4*loss_liver
   # Adjust weights based on dataset size and difficulty
   ```

---

### 3. NLP Integration Issues

#### Issue 3.1: MIMIC-III Access
**Status:** 🟡 May require IRB approval, CITI training

**Contingency Plan:**
1. **Phase 1 (Semester 1):** Complete project WITHOUT clinical notes
2. **Phase 2 (Semester 2):** Add NLP component once access granted
3. **Alternative:** Use synthetic clinical text for demo purposes

**Action Items:**
- [ ] Apply for MIMIC-III credentialing NOW
- [ ] Complete CITI training
- [ ] IRB exemption if needed

---

#### Issue 3.2: BioMistral/PubMedBERT Size
**Problem:** Large models (7B parameters) may not fit in limited GPU memory

**Mitigation:**
1. **Quantization:**
   ```python
   from transformers import BitsAndBytesConfig
   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   ```

2. **Use smaller models first:**
   - BioClinicalBERT (base)
   - PubMedBERT (110M parameters)
   
3. **Cloud GPU for inference** (Colab Pro, Lambda Labs)

---

### 4. Infrastructure Issues

#### Issue 4.1: GPU Availability
**Problem:** Local GPU may be insufficient for training

**Options:**
| Option | Cost | GPU | Pros | Cons |
|--------|------|-----|------|------|
| Google Colab Free | Free | T4 | Easy | Time limits |
| Colab Pro | $10/mo | V100/A100 | Good balance | Still limited |
| Kaggle | Free | P100 | Free notebooks | 30hr/week |
| Lambda Labs | $0.50/hr | A10 | Pay per use | Can get expensive |

**Recommendation:** Start with Kaggle (free), upgrade to Colab Pro if needed

---

#### Issue 4.2: API Rate Limits
**Problem:** OpenAI API costs for LLM explanations

**Mitigation:**
1. **Cache explanations** - don't regenerate for same input
2. **Use local LLM for development:**
   ```python
   # Ollama for local inference
   # Use GPT-4 only for final evaluation
   ```
3. **Budget limit:** Set max $20/month for API calls

---

### 5. Timeline Risks

#### Risk 5.1: Week 6-9 Bottleneck
**Problem:** Core ML/DL development compressed into 4 weeks

**Mitigation:**
1. **Start model experimentation in Week 4** (parallel with preprocessing)
2. **Use templates/boilerplate code** - don't reinvent wheel
3. **Prioritize:** XGBoost > FT-Transformer > NLP (if time-constrained)

**Adjusted Timeline If Behind:**
- Week 6-7: ML models (must complete)
- Week 8: FT-Transformer (can simplify if needed)
- Week 9: Basic XAI (SHAP) - defer LLM explanations to Sem 2

---

#### Risk 5.2: Thesis Writing Overlap
**Problem:** Writing thesis while developing may slow progress

**Mitigation:**
1. **Document AS you code** - don't wait until end
2. **Weekly thesis snippets:**
   - EDA results → Chapter 3
   - Model results → Chapter 4
   - Screenshots of webapp → Chapter 5
3. **Use literature review from proposal** - already 70% done

---

## ✅ Pre-Implementation Checklist

### Before Starting Week 1:
- [ ] Verify Python 3.9+ installed
- [ ] Create virtual environment
- [ ] Install requirements.txt
- [ ] Test GPU availability (if applicable)
- [ ] Set up .env with API keys
- [ ] Create GitHub repo and push initial structure

### Before EDA (Week 2):
- [ ] Confirm all 4 datasets load correctly
- [ ] Check for obvious data errors
- [ ] Verify target column names match code

### Before Model Training (Week 6):
- [ ] EDA completed for all datasets
- [ ] Preprocessing pipeline tested
- [ ] Class imbalance strategy decided
- [ ] Cross-validation setup confirmed

---

## 📋 Decision Points

| Week | Decision | Options | Default if Stuck |
|------|----------|---------|------------------|
| 2 | Imputation strategy | Mean/Median/KNN | Median |
| 4 | Multi-task vs separate | Joint/Separate | Separate first |
| 8 | FT-Transformer worth it? | Yes/No | Use if >2% improvement |
| 9 | NLP feasible? | Yes/No | Defer to Sem 2 |
| 10 | Streamlit vs Gradio | Either | Streamlit (simpler) |

---

## 🔄 Adaptation Strategy

This project is designed to be **modular and adaptable**:

1. **Core (Must Complete):**
   - Classical ML models (XGBoost/LightGBM)
   - Basic SHAP explanations
   - Simple web interface
   - API endpoints

2. **Enhanced (Should Complete):**
   - FT-Transformer deep learning
   - Multi-disease unified model
   - Advanced visualizations

3. **Advanced (Nice to Have):**
   - Clinical NLP integration
   - LLM-generated explanations
   - MIMIC-III integration

**Philosophy:** Complete core → enhance → advance. Never sacrifice core for advanced features.

---

## 📝 Notes for Future Updates

_Add notes here as issues arise during development:_

1. [Date] - [Issue] - [Resolution]
2. ...

---

*Document Version: 1.0 | Created: January 2025*
