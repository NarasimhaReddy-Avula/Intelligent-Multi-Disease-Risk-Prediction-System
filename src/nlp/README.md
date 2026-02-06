# 🧠 Clinical NLP Module

> Extract lifestyle and medical history features from patient-provided text

---

## 📋 What This Module Does

Converts free-form patient text into structured features for disease risk prediction:

**Input (Patient Text):**
```
"I smoke about 10 cigarettes per day, drink alcohol on weekends, 
and have a sedentary desk job. My father had a heart attack at age 60."
```

**Output (Structured Features):**
```python
{
    'smoking_current': 1,
    'smoking_frequency': '10 cigarettes/day',
    'alcohol_regular': 1,
    'physical_activity_sedentary': 1,
    'family_history_heart': 1
}
```

---

## 🛠️ Components

### 1. Sample Data
**File:** `data/raw/sample_patient_histories.json`
- 10 example patient histories with labels
- Covers: smoking, alcohol, exercise, family history, symptoms
- Can expand to 200-300 examples for full training

### 2. Extraction Pipeline
**File:** `src/nlp/clinical_text_extractor.py`
- Hybrid approach: Regex patterns + PubMedBERT
- Extracts: smoking, alcohol, physical activity, family history, symptoms
- Converts to binary features for ML models

---

## 🚀 Quick Start

### Option 1: Rule-Based Only (No Dependencies)
```python
from src.nlp.clinical_text_extractor import ClinicalTextExtractor

# Initialize (rule-based, works immediately)
extractor = ClinicalTextExtractor(use_bert=False)

# Extract features
text = "I smoke 10 cigarettes daily, father had diabetes"
features = extractor.extract_features(text)

print(features.smoking_status)      # "current"
print(features.family_history)      # ["diabetes"]

# Convert to model input
feature_dict = extractor.to_dict(features)
# {'smoking_current': 1, 'family_history_diabetes': 1, ...}
```

### Option 2: With PubMedBERT (Better Accuracy)
```bash
# Install dependencies
pip install transformers torch

# Use in code
extractor = ClinicalTextExtractor(use_bert=True)
```

---

## 📊 Feature Extraction Details

### 1. Smoking Detection
**Patterns:**
- Current: "I smoke", "10 cigarettes per day", "current smoker"
- Former: "quit smoking", "former smoker", "ex-smoker"
- Never: "non-smoker", "never smoked"

**Extracted:**
- `smoking_status`: never / former / current / unknown
- `smoking_frequency`: e.g., "10 cigarettes/day" (if current)

### 2. Alcohol Consumption
**Patterns:**
- Heavy: "4-5 beers daily", "heavy drinker"
- Regular: "drink regularly", "social drinker"
- Occasional: "weekends only", "occasional drinker"
- Never: "non-drinker", "don't drink"

**Extracted:**
- `alcohol_status`: never / occasional / regular / heavy / unknown
- `alcohol_frequency`: e.g., "4-5 beers" (if regular/heavy)

### 3. Physical Activity
**Patterns:**
- Sedentary: "sedentary", "desk job", "no exercise"
- Moderate: "exercise 3 times weekly", "moderately active"
- High: "exercise daily", "very active"

**Extracted:**
- `physical_activity`: sedentary / low / moderate / high / unknown

### 4. Family History
**Method:**
- Detect family member mention: father, mother, sibling, etc.
- Match disease keywords: diabetes, heart disease, kidney disease, etc.

**Extracted:**
- `family_history`: List of diseases (e.g., ["diabetes", "cardiovascular_disease"])

### 5. Symptoms
**Detected Symptoms:**
- chest_pain, fatigue, shortness_of_breath
- weight_loss, frequent_urination, excessive_thirst
- jaundice, edema

---

## 🧪 Testing the Extractor

Run the demo:
```bash
python src/nlp/clinical_text_extractor.py
```

Output:
```
📝 Case 1:
Text: I smoke about 10 cigarettes per day, drink alcohol on weekends...

Extracted Features:
  🚬 Smoking: current (10 cigarettes/day)
  🍺 Alcohol: occasional
  🏃 Activity: sedentary
  👨‍👩‍👦 Family History: cardiovascular_disease

  📊 Feature Vector (for ML model):
     {'smoking_current': 1, 'alcohol_regular': 0, ...}
```

---

## 📈 Expansion Plan (Week 9)

### Current Status
- ✅ 10 example patient histories
- ✅ Basic extraction pipeline
- ✅ Rule-based + PubMedBERT support

### Week 9 Tasks
- [ ] Expand to 200-300 patient history examples
- [ ] Fine-tune extraction patterns based on examples
- [ ] Integrate with main prediction pipeline
- [ ] Test extraction accuracy

### Week 10 Validation
- [ ] Download MTS-Dialog dataset (1.7K real clinical dialogues)
- [ ] Test extractor on real clinical text
- [ ] Calculate precision/recall for each feature
- [ ] Report in thesis

---

## 🔄 Integration with Main System

### In Prediction Pipeline:
```python
from src.nlp.clinical_text_extractor import ClinicalTextExtractor
from src.models.heart_model import predict_heart_risk

# Initialize
extractor = ClinicalTextExtractor()

# Patient input
lab_values = {'age': 55, 'bp_systolic': 140, 'glucose': 148, ...}
patient_text = "I smoke 10 cigarettes daily, father had heart attack"

# Extract NLP features
nlp_features = extractor.extract_features(patient_text)
nlp_dict = extractor.to_dict(nlp_features)

# Combine structured + NLP features
combined_features = {**lab_values, **nlp_dict}

# Predict risk
risk = predict_heart_risk(combined_features)
```

---

## 📚 References

**Pre-trained Models:**
- PubMedBERT: [microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)
- ClinicalBERT: [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

**Validation Datasets (Optional):**
- MTS-Dialog: https://github.com/UCSD-AI4H/MTS-Dialog
- ChatDoctor: https://github.com/Kent0n-Li/ChatDoctor

---

## ⚠️ Important Notes

### What You DON'T Need:
- ❌ MIMIC-III clinical notes
- ❌ Hospital EHR data
- ❌ Training clinical language models from scratch

### What You DO Have:
- ✅ Pre-trained PubMedBERT (understands clinical terminology)
- ✅ Rule-based patterns (handles structured extraction)
- ✅ Synthetic examples (tailored to your features)
- ✅ Public validation datasets (MTS-Dialog, ChatDoctor)

### Why This Works:
Your NLP module processes **patient-provided text** (what they type into your web app), not hospital clinical notes. Simple patient language like "I smoke" or "father had diabetes" is easier to extract than complex medical documentation.

---

**Created:** January 23, 2026  
**Status:** Basic implementation complete, ready for expansion in Week 9
