"""
================================================================================
UNIFIED DATASET CREATION - Week 3 Dataset Fusion Part 1
================================================================================
Multi-Disease Risk Prediction Platform
IIIT Sri City BTP Project

Purpose: Map all 4 disease datasets to unified schema (24 features + 4 targets)

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This script is the foundation of our multi-disease risk prediction platform.
It transforms 4 heterogeneous medical datasets into a unified format that can
be processed by a single FT-Transformer model with 4 output heads.

INPUT DATASETS:
---------------
1. Heart Disease (70,000 samples)
   - Source: Cardiovascular Disease dataset
   - Features: Basic vitals, categorical cholesterol/glucose
   - Target: Binary cardio (0/1)
   
2. Diabetes (100,000 samples)
   - Source: Comprehensive diabetes health indicators
   - Features: Full lipid panel, glucose markers, demographics
   - Target: diabetes_risk_score (0-100) - ONLY dataset with actual risk scores
   
3. Liver Disease (30,691 samples)
   - Source: Indian Liver Patient Dataset (expanded)
   - Features: Complete liver function panel (ALT, AST, ALP, bilirubin, albumin)
   - Target: Binary Result (1=disease, 0=healthy)
   
4. Kidney Disease (2,032 samples - SMOTE balanced)
   - Source: Synthetic CKD dataset
   - Features: Kidney biomarkers (creatinine, BUN, GFR), vitals, lipids
   - Target: Binary Diagnosis (0=healthy, 1=CKD)
   - Note: Applied SMOTE 1:3 ratio in Week 2 (original was 11:1 imbalanced)

OUTPUT:
-------
- Unified dataframes with 24 standardized features + 4 risk targets
- Ready for fusion in Week 4 (imputation, concatenation)
- Final model will predict 4 simultaneous risk scores (0-100%)

================================================================================
UNIFIED SCHEMA DESIGN RATIONALE
================================================================================

WHY 24 FEATURES?
----------------
We selected features based on three criteria:
1. OVERLAP: Present in 2+ datasets (stronger signal, less imputation needed)
2. CLINICAL IMPORTANCE: Critical biomarkers for specific diseases
3. NLP SEPARATION: Lifestyle factors (smoking, alcohol, activity) reserved for 
   NLP extraction in Week 8 to demonstrate text processing capability

FEATURE CATEGORIES:
-------------------
1. Demographics (3): age, gender, bmi
   - Universal across all medical assessments
   - BMI derived from height/weight for heart dataset
   
2. Vital Signs (2): systolic_bp, diastolic_bp
   - Available in heart, diabetes, kidney (3/4 datasets)
   - Critical for cardiovascular and kidney risk
   
3. Lipid Panel (4): cholesterol_total, hdl, ldl, triglycerides
   - HDL/LDL only in diabetes, kidney (2/4 datasets)
   - Total cholesterol available in 3/4 (heart has categorical)
   
4. Glucose Metabolism (2): glucose_fasting, hba1c
   - Essential for diabetes risk assessment
   - HbA1c superior to insulin (3-month average vs point-in-time)
   
5. Kidney Function (4): serum_creatinine, bun, gfr, protein_urine
   - ONLY available in kidney dataset
   - Critical: GFR <60 indicates CKD Stage 3+
   
6. Liver Function (8): alt, ast, alp, bilirubin_total, bilirubin_direct,
                       albumin, total_protein, ag_ratio
   - ONLY available in liver dataset
   - Complete liver panel for accurate assessment
   
7. Hematology (1): hemoglobin
   - Only in kidney dataset
   - Important: Anemia common in CKD patients

EXCLUDED FEATURES (with rationale):
-----------------------------------
1. insulin_level (diabetes only)
   - Technical: 75% missing values across unified dataset
   - Medical: Highly variable (timing, medication, individual variation)
   - Alternative: HbA1c is more reliable for risk prediction
   
2. heart_rate (diabetes only)
   - Only 1/4 datasets - insufficient for multi-disease model
   
3. Lifestyle (smoking, alcohol, physical_activity)
   - INTENTIONALLY EXCLUDED from structured data
   - Will be extracted from clinical notes in Week 8 (NLP component)
   - Reason: Demonstrate NLP capability to guide (important BTP contribution)
   
4. Medications (ACE inhibitors, statins, etc.)
   - These are TREATMENTS, not PREDICTORS
   - Including them would leak information about known conditions
   
5. Symptoms (fatigue, edema, nausea)
   - Reserved for NLP extraction from clinical notes
   - Text: "Patient reports fatigue and swelling" → extract features

================================================================================
RISK SCORE APPROACH
================================================================================

Our platform predicts CONTINUOUS RISK SCORES (0-100%), not binary labels.

Current Target State:
- Diabetes: ✅ Already has diabetes_risk_score (0-100)
- Heart/Liver/Kidney: ❌ Only binary labels (0/1)

Risk Score Generation Strategy (Week 4-5):
1. For diabetes: Normalize existing score to 0-1 (divide by 100)
2. For others: Train model with binary labels → use sigmoid output as risk
   - Model learns: high-risk features → probability closer to 1.0
   - Example: GFR=25, creatinine=5.0 → kidney_risk = 0.92 (92%)

Why This Works:
- Binary labels are extreme cases (definitely sick vs definitely healthy)
- Model generalizes to intermediate cases based on feature patterns
- Sigmoid output naturally represents probability/risk

================================================================================
FILES STRUCTURE
================================================================================

configs/
└── unified_schema.yaml      # Complete schema definition

src/data/
├── __init__.py
└── create_unified_dataset.py  # THIS FILE - mapping + preprocessing

datasets/
├── heart_disease.csv
├── diabetes_health_indicators.csv
├── liver_disease_30k.csv
├── chronic_kidney_disease_balanced.csv  # SMOTE balanced
└── unified/                  # Created in Week 4
    ├── heart_unified.csv
    ├── diabetes_unified.csv
    ├── liver_unified.csv
    ├── kidney_unified.csv
    └── combined_unified.csv  # All datasets merged

================================================================================
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_PATH = PROJECT_ROOT / "datasets"
CONFIG_PATH = PROJECT_ROOT / "configs"

# Unified feature names (24 features)
UNIFIED_FEATURES = [
    # Demographics (3)
    'age', 'gender', 'bmi',
    # Vitals (2)
    'systolic_bp', 'diastolic_bp',
    # Lipid Panel (4)
    'cholesterol_total', 'hdl', 'ldl', 'triglycerides',
    # Glucose (2)
    'glucose_fasting', 'hba1c',
    # Kidney Markers (4)
    'serum_creatinine', 'bun', 'gfr', 'protein_urine',
    # Liver Markers (8)
    'alt', 'ast', 'alp', 'bilirubin_total', 'bilirubin_direct',
    'albumin', 'total_protein', 'ag_ratio',
    # Hematology (1)
    'hemoglobin'
]

# Target columns
TARGET_COLUMNS = [
    'heart_disease_risk',
    'diabetes_risk', 
    'kidney_disease_risk',
    'liver_disease_risk'
]


# =============================================================================
# HEART DISEASE MAPPING
# =============================================================================

def load_and_map_heart(filepath: Path) -> pd.DataFrame:
    """
    Map heart disease dataset to unified schema.
    
    Original columns (13):
    - id, age (days), gender (1=F, 2=M), height, weight
    - ap_hi (systolic), ap_lo (diastolic)
    - cholesterol (1/2/3), gluc (1/2/3)
    - smoke, alco, active (binary) - EXCLUDED (NLP)
    - cardio (target)
    
    Transformations:
    - age: days → years (divide by 365.25)
    - gender: 1/2 → 0/1 (1=Female→0, 2=Male→1)
    - bmi: DERIVE from height/weight
    - cholesterol: categorical 1/2/3 → estimated mg/dL (180/239/280)
    - glucose: categorical 1/2/3 → estimated mg/dL (90/115/150)
    """
    print("\n" + "="*60)
    print("Loading HEART DISEASE dataset...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")
    
    # Create unified dataframe
    unified = pd.DataFrame(index=df.index)
    
    # Demographics
    unified['age'] = df['age'] / 365.25  # days to years
    unified['gender'] = df['gender'].map({1: 0, 2: 1})  # 1=F→0, 2=M→1
    unified['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)  # Derive BMI
    
    # Vitals
    unified['systolic_bp'] = df['ap_hi']
    unified['diastolic_bp'] = df['ap_lo']
    
    # Lipid Panel - expand categorical to estimated values
    cholesterol_map = {1: 180, 2: 239, 3: 280}  # Normal, Above, High
    unified['cholesterol_total'] = df['cholesterol'].map(cholesterol_map)
    unified['hdl'] = np.nan  # Not available
    unified['ldl'] = np.nan  # Not available
    unified['triglycerides'] = np.nan  # Not available
    
    # Glucose - expand categorical to estimated values
    glucose_map = {1: 90, 2: 115, 3: 150}  # Normal, Above, High
    unified['glucose_fasting'] = df['gluc'].map(glucose_map)
    unified['hba1c'] = np.nan  # Not available
    
    # Kidney markers - Not available
    unified['serum_creatinine'] = np.nan
    unified['bun'] = np.nan
    unified['gfr'] = np.nan
    unified['protein_urine'] = np.nan
    
    # Liver markers - Not available
    unified['alt'] = np.nan
    unified['ast'] = np.nan
    unified['alp'] = np.nan
    unified['bilirubin_total'] = np.nan
    unified['bilirubin_direct'] = np.nan
    unified['albumin'] = np.nan
    unified['total_protein'] = np.nan
    unified['ag_ratio'] = np.nan
    
    # Hematology - Not available
    unified['hemoglobin'] = np.nan
    
    # Target: Binary label (risk score generation in Week 4/5)
    unified['heart_disease_risk'] = df['cardio'].astype(float)  # Keep as 0/1 for now
    unified['diabetes_risk'] = np.nan
    unified['kidney_disease_risk'] = np.nan
    unified['liver_disease_risk'] = np.nan
    
    # Add source identifier
    unified['source_dataset'] = 'heart'
    
    # Data quality checks
    _validate_ranges(unified, 'heart')
    
    print(f"  Mapped shape: {unified.shape}")
    print(f"  Features available: {unified[UNIFIED_FEATURES].notna().sum().sum()} / {len(UNIFIED_FEATURES) * len(unified)}")
    coverage = unified[UNIFIED_FEATURES].notna().mean() * 100
    print(f"  Feature coverage: {coverage.mean():.1f}%")
    
    return unified


# =============================================================================
# DIABETES MAPPING
# =============================================================================

def load_and_map_diabetes(filepath: Path) -> pd.DataFrame:
    """
    Map diabetes dataset to unified schema.
    
    Original columns (31):
    - age, gender (M/F), ethnicity, education, income, employment - PARTIAL
    - smoking_status, alcohol, physical_activity - EXCLUDED (NLP)
    - diet_score, sleep, screen_time, family_history - EXCLUDED
    - hypertension_history, cardiovascular_history - medical history
    - bmi, waist_to_hip_ratio
    - systolic_bp, diastolic_bp, heart_rate
    - cholesterol_total, hdl_cholesterol, ldl_cholesterol, triglycerides
    - glucose_fasting, glucose_postprandial, insulin_level, hba1c
    - diabetes_risk_score (0-100), diabetes_stage, diagnosed_diabetes
    
    Transformations:
    - gender: 'Male'/'Female' → 1/0
    - diabetes_risk: normalize from 0-100 to 0-1
    """
    print("\n" + "="*60)
    print("Loading DIABETES dataset...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")
    
    # Create unified dataframe
    unified = pd.DataFrame(index=df.index)
    
    # Demographics
    unified['age'] = df['age'].astype(float)
    unified['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 0.5}).astype(float)
    unified['bmi'] = df['bmi'].astype(float)
    
    # Vitals
    unified['systolic_bp'] = df['systolic_bp'].astype(float)
    unified['diastolic_bp'] = df['diastolic_bp'].astype(float)
    
    # Lipid Panel - directly available
    unified['cholesterol_total'] = df['cholesterol_total'].astype(float)
    unified['hdl'] = df['hdl_cholesterol'].astype(float)
    unified['ldl'] = df['ldl_cholesterol'].astype(float)
    unified['triglycerides'] = df['triglycerides'].astype(float)
    
    # Glucose - directly available
    unified['glucose_fasting'] = df['glucose_fasting'].astype(float)
    unified['hba1c'] = df['hba1c'].astype(float)
    
    # Kidney markers - Not available
    unified['serum_creatinine'] = np.nan
    unified['bun'] = np.nan
    unified['gfr'] = np.nan
    unified['protein_urine'] = np.nan
    
    # Liver markers - Not available
    unified['alt'] = np.nan
    unified['ast'] = np.nan
    unified['alp'] = np.nan
    unified['bilirubin_total'] = np.nan
    unified['bilirubin_direct'] = np.nan
    unified['albumin'] = np.nan
    unified['total_protein'] = np.nan
    unified['ag_ratio'] = np.nan
    
    # Hematology - Not available
    unified['hemoglobin'] = np.nan
    
    # Target: Normalize risk score from 0-100 to 0-1
    unified['heart_disease_risk'] = np.nan
    unified['diabetes_risk'] = df['diabetes_risk_score'] / 100.0  # Normalize to 0-1
    unified['kidney_disease_risk'] = np.nan
    unified['liver_disease_risk'] = np.nan
    
    # Add source identifier
    unified['source_dataset'] = 'diabetes'
    
    # Data quality checks
    _validate_ranges(unified, 'diabetes')
    
    print(f"  Mapped shape: {unified.shape}")
    coverage = unified[UNIFIED_FEATURES].notna().mean() * 100
    print(f"  Feature coverage: {coverage.mean():.1f}%")
    
    return unified


# =============================================================================
# LIVER DISEASE MAPPING
# =============================================================================

def load_and_map_liver(filepath: Path) -> pd.DataFrame:
    """
    Map liver disease dataset to unified schema.
    
    Original columns (11):
    - Age of the patient
    - Gender of the patient (Male/Female)
    - Total Bilirubin, Direct Bilirubin
    - Alkphos Alkaline Phosphotase (ALP)
    - Sgpt Alamine Aminotransferase (ALT)
    - Sgot Aspartate Aminotransferase (AST)
    - Total Protiens, ALB Albumin, A/G Ratio
    - Result (target: 1=disease, 2=no disease) - NOTE: Check actual encoding
    
    Transformations:
    - gender: 'Male'/'Female' → 1/0
    - Result: Verify encoding (may need to flip)
    """
    print("\n" + "="*60)
    print("Loading LIVER DISEASE dataset...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")
    
    # Create unified dataframe
    unified = pd.DataFrame(index=df.index)
    
    # Demographics
    unified['age'] = df['Age of the patient'].astype(float)
    unified['gender'] = df['Gender of the patient'].map({'Female': 0, 'Male': 1}).astype(float)
    unified['bmi'] = np.nan  # Not available
    
    # Vitals - Not available
    unified['systolic_bp'] = np.nan
    unified['diastolic_bp'] = np.nan
    
    # Lipid Panel - Not available
    unified['cholesterol_total'] = np.nan
    unified['hdl'] = np.nan
    unified['ldl'] = np.nan
    unified['triglycerides'] = np.nan
    
    # Glucose - Not available
    unified['glucose_fasting'] = np.nan
    unified['hba1c'] = np.nan
    
    # Kidney markers - Not available
    unified['serum_creatinine'] = np.nan
    unified['bun'] = np.nan
    unified['gfr'] = np.nan
    unified['protein_urine'] = np.nan
    
    # Liver markers - ALL AVAILABLE
    unified['alt'] = df['Sgpt Alamine Aminotransferase'].astype(float)
    unified['ast'] = df['Sgot Aspartate Aminotransferase'].astype(float)
    unified['alp'] = df['Alkphos Alkaline Phosphotase'].astype(float)
    unified['bilirubin_total'] = df['Total Bilirubin'].astype(float)
    unified['bilirubin_direct'] = df['Direct Bilirubin'].astype(float)
    unified['albumin'] = df['ALB Albumin'].astype(float)
    unified['total_protein'] = df['Total Protiens'].astype(float)
    unified['ag_ratio'] = df['A/G Ratio Albumin and Globulin Ratio'].astype(float)
    
    # Hematology - Not available
    unified['hemoglobin'] = np.nan
    
    # Target: Check encoding and map to 0-1
    # Result: 1 = liver disease, 0 or 2 = no disease (verify)
    unified['heart_disease_risk'] = np.nan
    unified['diabetes_risk'] = np.nan
    unified['kidney_disease_risk'] = np.nan
    unified['liver_disease_risk'] = df['Result'].astype(float)  # Keep as is for now
    
    # Add source identifier
    unified['source_dataset'] = 'liver'
    
    # Data quality checks
    _validate_ranges(unified, 'liver')
    
    print(f"  Mapped shape: {unified.shape}")
    coverage = unified[UNIFIED_FEATURES].notna().mean() * 100
    print(f"  Feature coverage: {coverage.mean():.1f}%")
    
    return unified


# =============================================================================
# KIDNEY DISEASE MAPPING
# =============================================================================

def load_and_map_kidney(filepath: Path) -> pd.DataFrame:
    """
    Map kidney disease dataset (balanced) to unified schema.
    
    Original columns (54):
    - PatientID, Age, Gender (0/1), Ethnicity, SocioeconomicStatus, etc.
    - BMI, Smoking, AlcoholConsumption, PhysicalActivity - LIFESTYLE (NLP)
    - SystolicBP, DiastolicBP, FastingBloodSugar, HbA1c
    - SerumCreatinine, BUNLevels, GFR, ProteinInUrine, ACR
    - CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides
    - HemoglobinLevels
    - Medications, Symptoms - EXCLUDED
    - Diagnosis (target: 0=healthy, 1=CKD)
    
    Transformations:
    - Gender already 0/1
    - Most features directly mappable
    """
    print("\n" + "="*60)
    print("Loading KIDNEY DISEASE (balanced) dataset...")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"  Original shape: {df.shape}")
    
    # Create unified dataframe
    unified = pd.DataFrame(index=df.index)
    
    # Demographics
    unified['age'] = df['Age'].astype(float)
    unified['gender'] = df['Gender'].astype(float)  # Already 0/1
    unified['bmi'] = df['BMI'].astype(float)
    
    # Vitals
    unified['systolic_bp'] = df['SystolicBP'].astype(float)
    unified['diastolic_bp'] = df['DiastolicBP'].astype(float)
    
    # Lipid Panel
    unified['cholesterol_total'] = df['CholesterolTotal'].astype(float)
    unified['hdl'] = df['CholesterolHDL'].astype(float)
    unified['ldl'] = df['CholesterolLDL'].astype(float)
    unified['triglycerides'] = df['CholesterolTriglycerides'].astype(float)
    
    # Glucose
    unified['glucose_fasting'] = df['FastingBloodSugar'].astype(float)
    unified['hba1c'] = df['HbA1c'].astype(float)
    
    # Kidney markers - ALL AVAILABLE
    unified['serum_creatinine'] = df['SerumCreatinine'].astype(float)
    unified['bun'] = df['BUNLevels'].astype(float)
    unified['gfr'] = df['GFR'].astype(float)
    unified['protein_urine'] = df['ProteinInUrine'].astype(float)
    
    # Liver markers - Not available
    unified['alt'] = np.nan
    unified['ast'] = np.nan
    unified['alp'] = np.nan
    unified['bilirubin_total'] = np.nan
    unified['bilirubin_direct'] = np.nan
    unified['albumin'] = np.nan
    unified['total_protein'] = np.nan
    unified['ag_ratio'] = np.nan
    
    # Hematology
    unified['hemoglobin'] = df['HemoglobinLevels'].astype(float)
    
    # Target
    unified['heart_disease_risk'] = np.nan
    unified['diabetes_risk'] = np.nan
    unified['kidney_disease_risk'] = df['Diagnosis'].astype(float)  # 0/1
    unified['liver_disease_risk'] = np.nan
    
    # Add source identifier
    unified['source_dataset'] = 'kidney'
    
    # Data quality checks
    _validate_ranges(unified, 'kidney')
    
    print(f"  Mapped shape: {unified.shape}")
    coverage = unified[UNIFIED_FEATURES].notna().mean() * 100
    print(f"  Feature coverage: {coverage.mean():.1f}%")
    
    return unified


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def _validate_ranges(df: pd.DataFrame, source: str) -> None:
    """Validate feature values are within expected ranges."""
    
    range_checks = {
        'age': (0, 120),
        'gender': (0, 1),
        'bmi': (10, 70),
        'systolic_bp': (60, 250),
        'diastolic_bp': (30, 160),
        'cholesterol_total': (50, 500),
        'hdl': (10, 150),
        'ldl': (20, 300),
        'triglycerides': (20, 600),
        'glucose_fasting': (30, 400),
        'hba1c': (3, 16),
        'serum_creatinine': (0.1, 20),
        'bun': (2, 150),
        'gfr': (2, 150),
        'protein_urine': (0, 10),
        'alt': (1, 1000),
        'ast': (1, 1000),
        'alp': (10, 2000),
        'bilirubin_total': (0, 40),
        'bilirubin_direct': (0, 20),
        'albumin': (1, 7),
        'total_protein': (3, 12),
        'ag_ratio': (0.1, 3),
        'hemoglobin': (4, 22)
    }
    
    issues = []
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns and df[col].notna().any():
            below = (df[col] < min_val).sum()
            above = (df[col] > max_val).sum()
            if below > 0 or above > 0:
                issues.append(f"    {col}: {below} below {min_val}, {above} above {max_val}")
    
    if issues:
        print(f"  ⚠️  Range warnings for {source}:")
        for issue in issues[:5]:  # Show max 5
            print(issue)


def print_mapping_summary(datasets: Dict[str, pd.DataFrame]) -> None:
    """Print summary of all mapped datasets."""
    
    print("\n" + "="*80)
    print("MAPPING SUMMARY")
    print("="*80)
    
    total_rows = sum(len(df) for df in datasets.values())
    print(f"\nTotal samples: {total_rows:,}")
    
    print("\n┌─────────────┬──────────┬──────────────┬─────────────────┐")
    print("│ Dataset     │ Samples  │ Coverage (%) │ Target          │")
    print("├─────────────┼──────────┼──────────────┼─────────────────┤")
    
    for name, df in datasets.items():
        coverage = df[UNIFIED_FEATURES].notna().mean().mean() * 100
        target = [t for t in TARGET_COLUMNS if df[t].notna().any()][0]
        print(f"│ {name:11} │ {len(df):8,} │ {coverage:11.1f}% │ {target:15} │")
    
    print("└─────────────┴──────────┴──────────────┴─────────────────┘")
    
    # Feature availability matrix
    print("\nFeature Availability Matrix:")
    print("-" * 70)
    
    for feature in UNIFIED_FEATURES:
        avail = []
        for name, df in datasets.items():
            if df[feature].notna().any():
                avail.append(name[0].upper())  # First letter
            else:
                avail.append("-")
        print(f"  {feature:20} │ H:{avail[0]} D:{avail[1]} L:{avail[2]} K:{avail[3]}")


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================
"""
PREPROCESSING RATIONALE
-----------------------
This section handles data standardization BEFORE imputation (Week 4).

Why preprocess before imputation?
1. Ensures all datasets are on same scale for KNN-based imputation
2. Removes outliers that would skew imputation
3. Standardizes categorical encodings

What we do here (Week 3):
- Clip extreme outliers (medical impossibilities)
- Verify categorical encoding consistency
- Add metadata columns for tracking

What we do in Week 4:
- Imputation (KNN/MICE for missing values)
- Feature scaling (StandardScaler/MinMaxScaler)
- Train/test split happens in Week 5
"""

def preprocess_unified_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Preprocess a unified dataset before imputation.
    
    This function performs data cleaning and standardization:
    1. Clip physiologically impossible values
    2. Handle edge cases (negative values, extreme outliers)
    3. Add metadata for tracking data provenance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Unified dataset from mapping function
    source : str
        Source dataset name ('heart', 'diabetes', 'liver', 'kidney')
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed unified dataset
    """
    
    print(f"\n  Preprocessing {source} dataset...")
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # 1. CLIP PHYSIOLOGICALLY IMPOSSIBLE VALUES
    # -------------------------------------------------------------------------
    # These are medical maximums - values beyond these indicate data errors
    
    clip_ranges = {
        # Demographics
        'age': (0, 120),           # Max human age ~122 years
        'bmi': (10, 70),           # BMI 70+ is extreme morbid obesity
        
        # Vitals
        'systolic_bp': (60, 250),  # <60 = severe shock, >250 = hypertensive crisis
        'diastolic_bp': (30, 150), # Similar clinical bounds
        
        # Lipids (mg/dL)
        'cholesterol_total': (50, 500),   # <50 rare genetic condition
        'hdl': (5, 120),                  # Very low HDL is serious
        'ldl': (20, 350),                 # Familial hypercholesterolemia can go high
        'triglycerides': (20, 1000),      # Very high indicates pancreatitis risk
        
        # Glucose
        'glucose_fasting': (30, 500),     # <30 = severe hypoglycemia, >500 = DKA
        'hba1c': (3.5, 15),               # Normal 4-5.6%, diabetic 6.5%+
        
        # Kidney function
        'serum_creatinine': (0.2, 20),    # >15 typically needs dialysis
        'bun': (2, 150),                  # Extremely high in kidney failure
        'gfr': (2, 150),                  # GFR ~120 is normal, <15 is Stage 5
        'protein_urine': (0, 10),         # Massive proteinuria >3.5g/day
        
        # Liver function (U/L for enzymes)
        'alt': (1, 2000),                 # Very high in acute hepatitis
        'ast': (1, 2000),                 # Similar to ALT
        'alp': (10, 2000),                # Elevated in bile duct obstruction
        'bilirubin_total': (0.1, 40),     # >20 = severe jaundice
        'bilirubin_direct': (0, 20),      # Conjugated bilirubin
        'albumin': (1.0, 6.0),            # Low in liver failure, nephrotic syndrome
        'total_protein': (3.0, 12.0),     # Reflects nutritional and liver status
        'ag_ratio': (0.1, 3.0),           # Albumin/Globulin ratio
        
        # Hematology
        'hemoglobin': (4, 22),            # <7 = severe anemia, >20 = polycythemia
    }
    
    clipped_count = 0
    for col, (min_val, max_val) in clip_ranges.items():
        if col in df.columns and df[col].notna().any():
            original_outside = ((df[col] < min_val) | (df[col] > max_val)).sum()
            if original_outside > 0:
                df[col] = df[col].clip(lower=min_val, upper=max_val)
                clipped_count += original_outside
    
    if clipped_count > 0:
        print(f"    Clipped {clipped_count} values to physiological ranges")
    
    # -------------------------------------------------------------------------
    # 2. HANDLE SPECIAL CASES
    # -------------------------------------------------------------------------
    
    # Gender should be binary (0 or 1)
    if 'gender' in df.columns:
        # Replace any non-0/1 values with NaN for later imputation
        df.loc[~df['gender'].isin([0, 1, np.nan]), 'gender'] = np.nan
    
    # Negative values are impossible for most biomarkers
    non_negative_cols = [col for col in UNIFIED_FEATURES if col != 'gender']
    for col in non_negative_cols:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan
    
    # -------------------------------------------------------------------------
    # 3. ADD METADATA COLUMNS
    # -------------------------------------------------------------------------
    
    # Track data quality
    df['feature_count'] = df[UNIFIED_FEATURES].notna().sum(axis=1)
    df['missing_count'] = df[UNIFIED_FEATURES].isna().sum(axis=1)
    df['data_quality_score'] = df['feature_count'] / len(UNIFIED_FEATURES)
    
    print(f"    Average data quality score: {df['data_quality_score'].mean():.2%}")
    print(f"    Samples with >50% features: {(df['data_quality_score'] > 0.5).sum():,}")
    
    return df


def generate_feature_statistics(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate comprehensive statistics for all features across datasets.
    
    This is useful for:
    1. Understanding data distributions before imputation
    2. Identifying potential data quality issues
    3. Documentation for thesis/report
    
    Returns:
    --------
    pd.DataFrame
        Statistics table with mean, std, min, max, missing% per feature per dataset
    """
    
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    
    stats_rows = []
    
    for feature in UNIFIED_FEATURES:
        row = {'feature': feature}
        
        for name, df in datasets.items():
            if feature in df.columns and df[feature].notna().any():
                row[f'{name}_mean'] = df[feature].mean()
                row[f'{name}_std'] = df[feature].std()
                row[f'{name}_min'] = df[feature].min()
                row[f'{name}_max'] = df[feature].max()
                row[f'{name}_missing%'] = df[feature].isna().mean() * 100
            else:
                row[f'{name}_mean'] = np.nan
                row[f'{name}_std'] = np.nan
                row[f'{name}_min'] = np.nan
                row[f'{name}_max'] = np.nan
                row[f'{name}_missing%'] = 100.0
        
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    
    # Print summary
    print("\nFeature Statistics Summary:")
    print("-" * 80)
    for _, row in stats_df.iterrows():
        feature = row['feature']
        available_in = []
        for name in datasets.keys():
            if row[f'{name}_missing%'] < 100:
                available_in.append(name[0].upper())
        print(f"  {feature:20} │ Available in: {', '.join(available_in) if available_in else 'NONE'}")
    
    return stats_df


def save_unified_datasets(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    Save all unified datasets to CSV files.
    
    Output Structure:
    -----------------
    datasets/unified/
    ├── heart_unified.csv
    ├── diabetes_unified.csv
    ├── liver_unified.csv
    ├── kidney_unified.csv
    └── mapping_statistics.csv
    
    These files are intermediate outputs. The final combined dataset
    will be created in Week 4 after imputation.
    """
    
    print("\n" + "="*80)
    print("SAVING UNIFIED DATASETS")
    print("="*80)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    for name, df in datasets.items():
        output_path = output_dir / f"{name}_unified.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✅ Saved {name}: {output_path.name} ({len(df):,} rows)")
    
    # Save statistics
    stats_df = generate_feature_statistics(datasets)
    stats_path = output_dir / "mapping_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  ✅ Saved statistics: {stats_path.name}")
    
    print("\n" + "-"*80)
    print("Files ready for Week 4 imputation and fusion")
    print("-"*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to load, map, preprocess, and save all datasets.
    
    Week 3 Pipeline:
    ----------------
    1. Load raw datasets
    2. Map to unified schema (24 features + 4 targets)
    3. Preprocess (clip outliers, handle edge cases)
    4. Generate statistics
    5. Save intermediate unified files
    
    Week 4 will:
    - Load these unified files
    - Perform imputation (KNN/MICE)
    - Combine into single dataset
    - Quality assurance checks
    """
    
    print("="*80)
    print("UNIFIED DATASET CREATION - Week 3")
    print("Multi-Disease Risk Prediction Platform")
    print("IIIT Sri City BTP Project")
    print("="*80)
    
    # Load and map each dataset
    datasets = {}
    
    # Heart Disease
    heart_path = DATASETS_PATH / "heart_disease.csv"
    if heart_path.exists():
        datasets['heart'] = load_and_map_heart(heart_path)
        datasets['heart'] = preprocess_unified_data(datasets['heart'], 'heart')
    else:
        print(f"  ❌ Heart dataset not found: {heart_path}")
    
    # Diabetes
    diabetes_path = DATASETS_PATH / "diabetes_health_indicators.csv"
    if diabetes_path.exists():
        datasets['diabetes'] = load_and_map_diabetes(diabetes_path)
        datasets['diabetes'] = preprocess_unified_data(datasets['diabetes'], 'diabetes')
    else:
        print(f"  ❌ Diabetes dataset not found: {diabetes_path}")
    
    # Liver Disease
    liver_path = DATASETS_PATH / "liver_disease_30k.csv"
    if liver_path.exists():
        datasets['liver'] = load_and_map_liver(liver_path)
        datasets['liver'] = preprocess_unified_data(datasets['liver'], 'liver')
    else:
        print(f"  ❌ Liver dataset not found: {liver_path}")
    
    # Kidney Disease (balanced)
    kidney_path = DATASETS_PATH / "chronic_kidney_disease_balanced.csv"
    if kidney_path.exists():
        datasets['kidney'] = load_and_map_kidney(kidney_path)
        datasets['kidney'] = preprocess_unified_data(datasets['kidney'], 'kidney')
    else:
        print(f"  ❌ Kidney dataset not found: {kidney_path}")
    
    # Print summary
    if datasets:
        print_mapping_summary(datasets)
        
        # Save unified datasets
        output_dir = DATASETS_PATH / "unified"
        save_unified_datasets(datasets, output_dir)
    
    print("\n" + "="*80)
    print("WEEK 3 COMPLETE: Schema Mapping & Preprocessing")
    print("="*80)
    print("""
    Completed Tasks:
    ----------------
    ✅ Designed unified schema (24 features + 4 targets)
    ✅ Mapped heart dataset (age conversion, categorical expansion)
    ✅ Mapped diabetes dataset (gender encoding, risk normalization)
    ✅ Mapped liver dataset (liver panel extraction)
    ✅ Mapped kidney dataset (kidney biomarkers extraction)
    ✅ Applied preprocessing (outlier clipping, data quality scoring)
    ✅ Saved intermediate unified files
    
    Key Decisions Documented:
    -------------------------
    • Excluded insulin_level (75% missing, HbA1c more reliable)
    • Excluded lifestyle features (reserved for NLP Week 8)
    • Excluded medications (treatments, not predictors)
    • Heart cholesterol/glucose: categorical → estimated continuous values
    
    Next Steps (Week 4):
    --------------------
    → Load unified files
    → Impute missing values (KNN/MICE)
    → Combine all datasets
    → Quality assurance validation
    → Generate combined_unified.csv
    """)
    print("="*80)
    
    return datasets


if __name__ == "__main__":
    datasets = main()
