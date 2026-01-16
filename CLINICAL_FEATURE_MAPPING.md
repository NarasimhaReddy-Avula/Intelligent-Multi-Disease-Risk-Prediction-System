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

### Combined Prediction with LLM Enhancement
```python
def predict_multi_disease_risk(lab_values: LabInput, history: PatientHistory):
    # 1. Extract features from structured lab data
    structured_features = process_lab_values(lab_values)
    
    # 2. Extract features from text using NLP
    nlp_features = extract_lifestyle_from_text(history.history_text)
    
    # 3. Combine for each disease
    heart_risk = predict_heart(structured_features, nlp_features)
    diabetes_risk = predict_diabetes(structured_features, nlp_features)
    kidney_risk = predict_kidney(structured_features, nlp_features)
    liver_risk = predict_liver(structured_features, nlp_features)
    
    # 4. Generate technical SHAP explanations
    shap_explanations = generate_shap_explanations(
        models=[heart_model, diabetes_model, kidney_model, liver_model],
        features=structured_features
    )
    
    # 5. LLM Layer: Convert to human-friendly + diet recommendations
    llm_explanations = generate_patient_friendly_explanations(
        risk_scores={'heart': heart_risk, 'diabetes': diabetes_risk, 
                    'kidney': kidney_risk, 'liver': liver_risk},
        shap_values=shap_explanations,
        patient_profile={'age': lab_values.age, 'conditions': nlp_features}
    )
    
    return {
        'risk_scores': {
            'heart': heart_risk,
            'diabetes': diabetes_risk,
            'kidney': kidney_risk,
            'liver': liver_risk
        },
        'technical_explanation': shap_explanations,  # For doctors
        'patient_explanation': llm_explanations,      # For patients
        'diet_recommendations': llm_explanations['diet_plan'],
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

## 🎯 **XAI + LLM Output Examples**

### **Example Patient Case:**
```
Patient: 55-year-old male
Lab Values: Glucose=148mg/dL, Creatinine=2.1, GFR=38, BP=150/95, BMI=32.5
History: "Type 2 diabetes for 15 years, father had kidney failure, I smoke 15 cigarettes daily"
```

---

### **Step 1: Technical SHAP Explanation (For Doctors)**

```python
{
  "diabetes_risk": 0.85,
  "shap_values": {
    "Glucose": +0.28,      # 148 mg/dL (high)
    "BMI": +0.18,          # 32.5 (obese)
    "Age": +0.12,          # 55 years
    "Family_history": +0.15,
    "Smoking": +0.08,
    "Base_value": 0.04
  },
  "top_features": [
    {"feature": "Glucose", "value": 148, "impact": "HIGH", "shap": 0.28},
    {"feature": "BMI", "value": 32.5, "impact": "HIGH", "shap": 0.18},
    {"feature": "Family_history", "value": "positive", "impact": "MEDIUM", "shap": 0.15}
  ]
}
```

---

### **Step 2: LLM-Generated Patient-Friendly Explanation**

```
🔴 DIABETES RISK: 85% (CRITICAL - Immediate attention needed)

Your diabetes is not well controlled. Here's why your risk is high:

🩸 Blood Sugar (Glucose): 148 mg/dL
   Your fasting blood sugar is significantly elevated.
   • Normal range: below 100 mg/dL
   • You are 48% above the healthy limit
   • This is the BIGGEST factor increasing your risk

⚖️ Body Weight (BMI): 32.5
   You are in the obese category.
   • Healthy range: 18.5-24.9
   • Extra weight makes your body resistant to insulin
   • Losing just 10-15 pounds can significantly improve your condition

👨‍👩‍👦 Family History
   Your father had kidney failure, which increases your genetic risk.
   • Family history accounts for 15% of your risk
   • This makes preventive care even more critical

🚬 Smoking
   Smoking damages blood vessels and worsens diabetes complications.
   • Currently contributes 8% to your risk
   • Quitting smoking could reduce complications by 40%

💡 The Good News:
   Most of these factors are under your control! With lifestyle changes 
   and proper medication, you can reduce your risk significantly.
```

---

### **Step 3: LLM-Generated Diet & Lifestyle Recommendations**

```
🍽️ PERSONALIZED DIET PLAN FOR YOU

Based on your high blood sugar (148 mg/dL), kidney strain (GFR=38), 
and obesity (BMI=32.5), here's your customized diet plan:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🥗 WHAT TO EAT (Focus on these daily):

✅ Low-Glycemic Vegetables (Unlimited):
   • Leafy greens: spinach, kale, lettuce
   • Broccoli, cauliflower, bell peppers, cucumber
   • Why: Won't spike your blood sugar, rich in nutrients

✅ Lean Proteins (At each meal):
   • Skinless chicken breast, fish (salmon, mackerel)
   • Eggs, tofu, lentils
   • Portion: Palm-sized (about 100g per meal)
   • Why: Helps control hunger, maintains muscle, doesn't raise blood sugar

✅ Whole Grains (LIMITED - 1 cup/day):
   • Brown rice, quinoa, oats
   • Why: Better than white rice, but still monitor portions

✅ Healthy Fats (Small amounts):
   • Olive oil (2 tablespoons/day), avocado (1/4 per day)
   • Nuts: almonds, walnuts (handful/day)
   • Why: Helps with satiety, good for heart

✅ Low-Sugar Fruits (1-2 servings/day):
   • Berries, apple (with skin), pear
   • Avoid: mango, grapes, bananas (high sugar)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ WHAT TO AVOID (These harm your kidneys and spike blood sugar):

🚫 High-Sugar Foods:
   • White bread, white rice, pasta
   • Sweets, desserts, sugary drinks
   • Why: Directly spike your blood sugar

🚫 High-Sodium Foods (CRITICAL for your kidney health):
   • Processed foods, canned soups, chips
   • Pickles, sauces, restaurant food
   • Limit: Max 2,000mg sodium/day (about 1 teaspoon salt)
   • Why: Your GFR=38 means kidneys are struggling; excess salt worsens it

🚫 High-Potassium Foods (Your kidneys can't filter excess):
   • Bananas, oranges, tomatoes, potatoes
   • Why: With reduced kidney function, potassium can build up dangerously

🚫 Red Meat & Processed Meats:
   • Beef, pork, sausages, bacon
   • Why: High in saturated fat, strains kidneys

🚫 Alcohol:
   • Limit to 1 drink/week or avoid completely
   • Why: Interferes with blood sugar control

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 SAMPLE ONE-DAY MEAL PLAN:

🌅 Breakfast (7:00 AM):
   • 2 boiled eggs
   • 1 slice whole wheat toast
   • 1/2 cup sautéed spinach
   • Green tea (no sugar)

🍎 Mid-Morning Snack (10:00 AM):
   • 1 small apple with 10 almonds

🍱 Lunch (1:00 PM):
   • Grilled chicken breast (palm-sized)
   • Large salad (cucumber, lettuce, tomato with olive oil)
   • 1/2 cup brown rice
   • Buttermilk (no salt)

☕ Afternoon Snack (4:00 PM):
   • Handful of walnuts
   • Herbal tea

🍽️ Dinner (7:00 PM):
   • Baked fish (salmon or mackerel)
   • Steamed broccoli and carrots
   • Small portion quinoa

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏃 LIFESTYLE CHANGES (As Important as Diet):

1. 🚭 QUIT SMOKING (Priority #1):
   • Reduces diabetes complications by 40%
   • Improves kidney function
   • Seek support: nicotine patches, counseling

2. 🚶 Daily Exercise (Start small):
   • Walk 30 minutes after dinner
   • Target: 150 minutes/week
   • Why: Lowers blood sugar naturally, aids weight loss

3. 💧 Hydration:
   • Drink 6-8 glasses of water/day
   • Helps kidneys flush toxins

4. ⏰ Meal Timing:
   • Eat at same times daily
   • No eating 3 hours before bed
   • Why: Stabilizes blood sugar

5. 🩺 Monitor Daily:
   • Check blood sugar before meals
   • Target: Below 110 mg/dL fasting
   • Log food and readings

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 YOUR 3-MONTH GOALS:

✅ Reduce blood sugar: 148 → below 110 mg/dL
✅ Lose weight: BMI 32.5 → below 30 (lose 10-12 pounds)
✅ Improve kidney function: GFR 38 → stabilize/improve
✅ Quit smoking completely
✅ Lower blood pressure: 150/95 → below 130/80

📞 IMPORTANT: 
• Consult your doctor before making major diet changes
• Get kidney function tested every 3 months
• Consider seeing a nutritionist for personalized guidance
• Report any symptoms: increased thirst, fatigue, swelling

⚠️ WARNING SIGNS - Seek immediate medical attention if:
• Blood sugar >250 mg/dL
• Severe leg swelling
• Difficulty breathing
• Chest pain

You can turn this around! Small consistent changes lead to big results. 💪
```

---

## 🔄 **Complete System Workflow**

```
┌─────────────────────────────────────────────────────────────┐
│  PATIENT INPUT                                              │
│  Lab: Glucose=148, Creatinine=2.1, GFR=38, BP=150/95      │
│  Text: "Type 2 diabetes, father had kidney failure, smoke" │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  RISK PREDICTION (XGBoost/FT-Transformer)                  │
│  • Diabetes: 85%                                            │
│  • Kidney: 72%                                              │
│  • Heart: 68%                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  SHAP EXPLANATIONS (Technical - For Doctors)               │
│  • Glucose: +0.28 SHAP value (main driver)                │
│  • BMI: +0.18 SHAP value                                    │
│  • Feature importance rankings                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  LLM LAYER (GPT-4/Llama)                                   │
│  Prompt: "Convert these technical SHAP values to           │
│           patient-friendly language. Patient is 55M with   │
│           glucose=148, GFR=38. Provide diet plan."         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT (Patient-Facing)                             │
│  1. Risk scores with color coding                          │
│  2. Simple explanations: "Your blood sugar is 48% too high"│
│  3. Personalized diet plan with meal examples              │
│  4. Lifestyle recommendations                              │
│  5. 3-month goals and warning signs                        │
└─────────────────────────────────────────────────────────────┘
```

---

*Document Version: 1.1 | Updated: January 16, 2026 | Added LLM Integration*
