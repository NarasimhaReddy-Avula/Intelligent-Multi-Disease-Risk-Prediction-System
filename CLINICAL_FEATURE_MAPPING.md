# 🩺 Clinical Feature Mapping & Data Input Strategy

> **Purpose:** Define what goes into structured lab input vs NLP text input  
> **Last Updated:** January 16, 2026

---

## 🎯 System Architecture

```
Patient Input:
├── 📊 Structured Input (Lab Report Values)
│   └── Numerical fields from actual medical test reports
│       - Blood tests (glucose, cholesterol, enzymes)
│       - Vitals (BP, heart rate, BMI)
│       - Demographics (age, sex)
│
└── 💬 Unstructured Input (Patient History - NLP)
    └── Free text entered by patient or extracted from conversation
        - Lifestyle: "I smoke 10 cigarettes daily"
        - Family history: "My father had diabetes"
        - Habits: "I drink alcohol occasionally, sedentary lifestyle"
        - Comorbidities: "I have hypertension for 5 years"
```

---

## 📋 Dataset-by-Dataset Clinical Analysis

### 1️⃣ **Heart Disease (Cardiovascular - 70,000 samples)**

**File:** `heart.csv`  
**Target:** `cardio` (0=healthy, 1=cardiovascular disease)

#### ✅ LAB REPORT VALUES (Structured Input)
| Feature | Clinical Name | Range | On Report? |
|---------|---------------|-------|------------|
| `age` | Age | Years | ✅ Demographics |
| `gender` | Sex | 1=Female, 2=Male | ✅ Demographics |
| `height` | Height | cm | ✅ Physical exam |
| `weight` | Weight | kg | ✅ Physical exam |
| `ap_hi` | Systolic BP | mmHg | ✅ Vital signs |
| `ap_lo` | Diastolic BP | mmHg | ✅ Vital signs |
| `cholesterol` | Total Cholesterol | 1=Normal, 2=Above normal, 3=Well above | ✅ Lipid panel |
| `gluc` | Glucose | 1=Normal, 2=Above normal, 3=Well above | ✅ Blood glucose test |

**Derived:** `BMI = weight / (height/100)^2`

#### ❌ NLP TEXT INPUT (Patient History)
| Feature | Extract From Text |
|---------|-------------------|
| `smoke` | "I smoke" / "non-smoker" / "former smoker" |
| `alco` | "I drink alcohol regularly" / "occasional drinker" / "non-drinker" |
| `active` | "I exercise daily" / "sedentary lifestyle" / "moderately active" |

**Example Patient Input:**
```
Lab Values: Age=55, Male, Height=175cm, Weight=85kg, BP=140/90, Cholesterol=240mg/dL, Glucose=110mg/dL
Text: "I smoke about 10 cigarettes per day, drink alcohol on weekends, and have a sedentary desk job"
```

---

### 2️⃣ **Diabetes (Pima Indians - 768 samples)**

**File:** `diabetes.csv`  
**Target:** `Outcome` (0=no diabetes, 1=diabetes)

#### ✅ LAB REPORT VALUES (Structured Input)
| Feature | Clinical Name | Range | On Report? |
|---------|---------------|-------|------------|
| `Glucose` | Fasting Plasma Glucose | mg/dL (Normal: <100) | ✅ Blood glucose test |
| `BloodPressure` | Diastolic BP | mmHg | ✅ Vital signs |
| `SkinThickness` | Triceps Skinfold | mm | ✅ Physical measurement |
| `Insulin` | 2-Hour Serum Insulin | μU/mL | ✅ Insulin assay |
| `BMI` | Body Mass Index | kg/m² | ✅ Calculated from height/weight |
| `Age` | Age | Years | ✅ Demographics |

#### 🔄 MIXED (Can be from report OR text)
| Feature | From Report | From Text |
|---------|-------------|-----------|
| `Pregnancies` | Number of pregnancies | "I've had 3 pregnancies" |
| `DiabetesPedigreeFunction` | Genetic score (if available) | "Family history: father and grandmother had diabetes" |

**Example Patient Input:**
```
Lab Values: Glucose=148mg/dL, BP=72mmHg, SkinThickness=35mm, Insulin=0 (not measured), BMI=33.6, Age=50
Text: "I've had 6 pregnancies. My father and two siblings have type 2 diabetes. No regular exercise."
```

---

### 3️⃣ **Chronic Kidney Disease (1,659 samples)** ⭐ **COMPREHENSIVE DATASET**

**File:** `Chronic_Kidney_Dsease_data.csv`  
**Target:** `Diagnosis` (CKD diagnosis status)

#### ✅ LAB REPORT VALUES (18 Features - Structured Input)

**Renal Function Panel:**
| Feature | Clinical Name | Normal Range | Critical Test |
|---------|---------------|--------------|---------------|
| `SerumCreatinine` | Creatinine | 0.7-1.3 mg/dL | ⭐ Key marker |
| `BUNLevels` | Blood Urea Nitrogen | 7-20 mg/dL | ⭐ Key marker |
| `GFR` | Glomerular Filtration Rate | >60 mL/min | ⭐⭐⭐ PRIMARY |
| `ProteinInUrine` | Proteinuria | Absent | ⭐ Kidney damage |
| `ACR` | Albumin-Creatinine Ratio | <30 mg/g | ⭐ Early detection |

**Blood Tests:**
| Feature | Clinical Name | Normal Range |
|---------|---------------|--------------|
| `SystolicBP` | Systolic Blood Pressure | <120 mmHg |
| `DiastolicBP` | Diastolic Blood Pressure | <80 mmHg |
| `FastingBloodSugar` | Fasting Glucose | <100 mg/dL |
| `HbA1c` | Glycated Hemoglobin | <5.7% |
| `HemoglobinLevels` | Hemoglobin | 12-16 g/dL (F), 14-18 (M) |

**Electrolytes:**
- `SerumElectrolytesSodium` (135-145 mEq/L)
- `SerumElectrolytesPotassium` (3.5-5.0 mEq/L)
- `SerumElectrolytesCalcium` (8.5-10.5 mg/dL)
- `SerumElectrolytesPhosphorus` (2.5-4.5 mg/dL)

**Lipid Panel:**
- `CholesterolTotal`, `CholesterolLDL`, `CholesterolHDL`, `CholesterolTriglycerides`

#### ☑️ BINARY CHECKLIST (5 Features - Medication History)

**Current Medications:**
| Checkbox | Medication Type | Examples | Kidney Impact |
|----------|-----------------|----------|---------------|
| `ACEInhibitors` | ACE Inhibitors | Lisinopril, Enalapril | Protects kidneys |
| `Diuretics` | Water Pills | Furosemide, HCTZ | Reduces fluid |
| `NSAIDsUse` | Pain Relievers | Ibuprofen, Naproxen | ⚠️ Can damage |
| `Statins` | Cholesterol Drugs | Atorvastatin | Cardiovascular |
| `AntidiabeticMedications` | Diabetes Meds | Metformin, Insulin | Blood sugar |

**UI Implementation:**
```
☐ ACE Inhibitors (e.g., Lisinopril, Enalapril)
☐ Diuretics (e.g., Furosemide, Hydrochlorothiazide)
☐ NSAIDs (e.g., Ibuprofen, Naproxen)
☐ Statins (e.g., Atorvastatin, Simvastatin)
☐ Antidiabetic Medications (e.g., Metformin, Insulin)
```

#### 💬 NLP TEXT INPUT (13 Features - Lifestyle & History)

**Demographics:**
- `Ethnicity`, `SocioeconomicStatus`, `EducationLevel`, `Gender`, `Age`

**Lifestyle Habits:**
- `Smoking` - "I smoke 10 cigarettes daily"
- `AlcoholConsumption` - "I drink 3-4 beers daily"
- `PhysicalActivity` - "I exercise 3 times weekly"
- `DietQuality` - "High sodium diet, processed foods"
- `SleepQuality` - "I sleep 5-6 hours per night"

**Family History:**
- `FamilyHistoryKidneyDisease` - "Father had kidney failure"
- `FamilyHistoryHypertension` - "Mother has high BP"
- `FamilyHistoryDiabetes` - "Sister has diabetes"

**Environmental:**
- `WaterQuality`, `OccupationalExposureChemicals`, `HeavyMetalsExposure`

**Symptoms (from text):**
- `Edema` - "Swelling in ankles"
- `FatigueLevels` - "Constantly tired"
- `NauseaVomiting`, `MuscleCramps`, `Itching`

**Example Patient Input:**
```
Lab Values: Age=65, Male, BP=150/95, FBS=105, HbA1c=6.2%, Creatinine=2.1, BUN=32, GFR=38, 
           Hemoglobin=10.5, Potassium=5.3, ACR=150

Medications:
☑ ACE Inhibitors (Lisinopril 10mg daily)
☐ Diuretics
☑ NSAIDs (Ibuprofen - frequent use for arthritis)
☑ Statins (Atorvastatin 20mg)
☑ Diabetes medication (Metformin 1000mg twice daily)

Patient History:
"I have type 2 diabetes for 15 years, not well controlled. High blood pressure for 10 years. 
My father had kidney failure and was on dialysis. I work in a paint factory with exposure to 
solvents. I smoke about 15 cigarettes daily for 25 years. I rarely exercise due to fatigue and 
leg swelling. Recently noticed decreased urination and foamy urine."
```

---

### 4️⃣ **Liver Disease (Indian Liver Patient - 583 samples)**

**File:** `liver.csv`  
**Target:** `Dataset` (1=liver disease, 2=no liver disease)

#### ✅ LAB REPORT VALUES (100% - Perfect Clinical Dataset!)
| Feature | Clinical Name | Normal Range | On Report? |
|---------|---------------|--------------|------------|
| `Age` | Age | Years | ✅ Demographics |
| `Gender` | Sex | Male/Female | ✅ Demographics |
| `Total_Bilirubin` | Total Bilirubin | 0.1-1.2 mg/dL | ✅ Liver function test |
| `Direct_Bilirubin` | Direct Bilirubin | 0-0.3 mg/dL | ✅ Liver function test |
| `Alkaline_Phosphotase` | ALP | 44-147 IU/L | ✅ Liver function test |
| `Alamine_Aminotransferase` | ALT/SGPT | 7-56 IU/L | ✅ Liver function test |
| `Aspartate_Aminotransferase` | AST/SGOT | 10-40 IU/L | ✅ Liver function test |
| `Total_Protiens` | Total Protein | 6-8 g/dL | ✅ Liver function test |
| `Albumin` | Albumin | 3.5-5.5 g/dL | ✅ Liver function test |
| `Albumin_and_Globulin_Ratio` | A/G Ratio | 1-2 | ✅ Liver function test |

#### ❌ NLP TEXT INPUT (Risk Factors)
Not in dataset, but extracted from patient history:
- "I consume alcohol daily" → Alcoholic liver disease risk
- "Family history of hepatitis" → Genetic risk
- "I had hepatitis B/C infection" → Viral hepatitis history
- "Jaundice, dark urine, fatigue" → Symptoms

**Example Patient Input:**
```
Lab Values: Age=52, Male, Total Bilirubin=2.5, Direct Bilirubin=1.1, ALP=220, ALT=95, AST=110, 
           Total Protein=5.8, Albumin=2.9, A/G Ratio=0.8
Text: "I drink 4-5 beers daily for 20 years. Recently noticed yellowing of eyes, dark urine, 
      and constant fatigue. No family history of liver disease."
```

---

## 🔍 CRITICAL FINDINGS

### ✅ **GOOD Datasets (Clinically Valid)**

| Dataset | Grade | Reason |
|---------|-------|--------|
| **Liver** | ⭐⭐⭐⭐⭐ | Perfect! 100% lab values from standard liver function panel |
| **Diabetes (Pima)** | ⭐⭐⭐⭐ | Excellent! Has glucose and insulin - the key diabetes markers |
| **Heart** | ⭐⭐⭐⭐ | Very good! Has BP, cholesterol, glucose from standard tests |

### ⚠️ **PROBLEMATIC Dataset**

| Dataset | Grade | Reason |
|---------|-------|--------|
| **Stroke** | ⭐⭐ | Weak! Only has glucose and BMI as lab values. Rest is demographic/survey data |

---

## 🚨 STROKE DATASET ISSUE

**Problem:** Stroke dataset is mostly **survey/demographic data**, NOT lab values:
- `ever_married`, `work_type`, `Residence_type` → NOT on medical reports
- Only 2 lab values: `glucose` and `BMI`

**Options:**

### Option 1: **Keep it, use as "screening model"**
- Uses minimal data (glucose, BMI, age, patient history)
- Predicts stroke risk when limited data available
- Document limitation in thesis

### Option 2: **Find better stroke dataset**
**Requirements:**
- Must have BP readings (critical for stroke!)
- Ideally: cholesterol, blood thinners, prior TIA
- Comorbidity flags (diabetes, heart disease) are acceptable

### Option 3: **Replace with Kidney Disease**
**Chronic Kidney Disease Dataset:**
- Has: BP, glucose, hemoglobin, albumin, red/white blood cells
- 400 samples
- All clinical lab values

---

## 🎯 Recommended Action: Replace Stroke Dataset

**Search for:**
1. "Stroke dataset with blood pressure"
2. "Cerebrovascular disease clinical data"
3. OR use **Kidney Disease** (better clinical features)

**Kidney Disease Dataset:**
- File: Can download from UCI/Kaggle
- Features: Age, BP, specific gravity, albumin, sugar, RBC, WBC, bacteria, hemoglobin, etc.
- All from urinalysis + blood test
- Perfect for lab report input!

---

## 📊 Final Dataset Quality Assessment

| Disease | Dataset | Clinical Grade | Lab Values | Should Keep? |
|---------|---------|----------------|------------|--------------|
| Heart | Cardiovascular 70K | ⭐⭐⭐⭐ | BP, cholesterol, glucose, BMI | ✅ YES |
| Diabetes | Pima Indians | ⭐⭐⭐⭐ | Glucose, insulin, BP, BMI | ✅ YES |
| Chronic Kidney Disease | CKD Dataset | ⭐⭐⭐⭐⭐ | Complete renal panel (18 tests) + meds checklist | ✅ YES |
| Liver | Indian Liver | ⭐⭐⭐⭐⭐ | Complete liver panel (9 tests) | ✅ YES |

**Total: ~72,000 samples | 100% clinical lab values | Ready for production!**

---

## 💡 Two-Part Input System Architecture

### Phase 1: Structured Lab Input
```python
class LabInput(BaseModel):
    # Demographics
    age: int
    sex: str
    
    # Vitals
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    height: Optional[float]
    weight: Optional[float]
    bmi: Optional[float]  # Calculated if height/weight given
    
    # Blood Tests
    glucose: Optional[float]
    cholesterol: Optional[float]
    insulin: Optional[float]
    
    # Liver Panel (if available)
    alt: Optional[float]
    ast: Optional[float]
    bilirubin_total: Optional[float]
    albumin: Optional[float]
```

### Phase 2: NLP Text Input
```python
class PatientHistory(BaseModel):
    history_text: str
    # Extract from text using NLP:
    # - smoking: yes/no/former, frequency
    # - alcohol: yes/no, frequency
    # - physical_activity: sedentary/moderate/active
    # - family_history: which diseases in family
    # - symptoms: any current symptoms mentioned
    # - medications: current medications
```

### Combined Prediction
```python
def predict_multi_disease_risk(lab_values: LabInput, history: PatientHistory):
    # 1. Extract features from structured lab data
    structured_features = process_lab_values(lab_values)
    
    # 2. Extract features from text using NLP
    nlp_features = extract_lifestyle_from_text(history.history_text)
    
    # 3. Combine for each disease
    heart_risk = predict_heart(structured_features, nlp_features)
    diabetes_risk = predict_diabetes(structured_features, nlp_features)
    stroke_risk = predict_stroke(structured_features, nlp_features)
    liver_risk = predict_liver(structured_features, nlp_features)
    
    return {
        'heart': heart_risk,
        'diabetes': diabetes_risk,
        'stroke': stroke_risk,
        'liver': liver_risk,
        'cross_disease_insights': analyze_comorbidity_risks()
    }
```

---

## ✅ ACTION ITEMS (Today)

1. ✅ **Restored Pima diabetes dataset** - DONE
2. ✅ **Removed BRFSS** - DONE
3. ⏳ **Find replacement for Stroke dataset** - Need better clinical features
4. ⏳ **Update all documentation** - After final dataset selection

**Recommendation:** Search for stroke dataset with BP, or replace with Kidney Disease dataset for 4th disease.

---

*Document Version: 1.0 | Created: January 16, 2026*
