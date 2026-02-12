"""
Week 4 - Step 3A: Masked Imputation Approach
=============================================

STRATEGY: Masking Approach (48 Features)
-----------------------------------------
For each original feature with missing values, create TWO columns:
  1. feature_value: The actual value (0 if missing)
  2. feature_mask:  1 if present, 0 if missing

WHY MASKING OVER SENTINEL?
--------------------------
1. NO AMBIGUITY: Mask explicitly tells model what to ignore
   - Sentinel -1: After scaling, -1 might overlap with real scaled values
   - Masking: mask=0 clearly means "ignore this value"

2. BETTER FOR FT-TRANSFORMER:
   - Attention mechanism learns to weight mask=0 features as 0.0
   - No confusion between real low values and sentinel
   
3. INTERPRETABLE:
   - Feature importance shows which masks were used
   - Clear documentation: "Used glucose (mask=1)" vs "Ignored ALT (mask=0)"

IMPUTATION RULES:
-----------------
1. age, gender: Simple imputation (median/mode) - <1% missing (true random errors)
2. All 22 other features: value=0 (placeholder), mask=0/1

OUTPUT:
-------
- Original 24 features → 48 columns (24 values + 24 masks)
- Age, gender don't need masks (complete after imputation)
- Total: age, gender + 22 value columns + 22 mask columns = 46 columns
  (Plus 4 targets + 1 source_dataset = 51 total)

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "combined_unified.csv"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "combined_unified_masked.csv"

def main():
    print("=" * 70)
    print("WEEK 4 - STEP 3A: MASKED IMPUTATION")
    print("=" * 70)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n[1] Loading combined dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"    ✓ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    
    # Define column groups
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    # Features that need masking (22 features with structural missingness)
    features_to_mask = [
        'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_total', 
        'glucose_fasting', 'hdl', 'ldl', 'triglycerides', 'hba1c',
        'alt', 'ast', 'alp', 'albumin', 'total_protein', 
        'bilirubin_total', 'bilirubin_direct', 'ag_ratio',
        'serum_creatinine', 'bun', 'gfr', 'protein_urine', 'hemoglobin'
    ]
    
    # Features to simply impute (complete features, <1% random missing)
    features_simple_impute = ['age', 'gender']
    
    # =========================================================================
    # 2. SIMPLE IMPUTATION FOR AGE & GENDER
    # =========================================================================
    print("\n[2] Simple imputation for age and gender...")
    
    # Age: median imputation
    age_missing = df['age'].isna().sum()
    age_median = df['age'].median()
    df['age'] = df['age'].fillna(age_median)
    print(f"    age: {age_missing} missing → filled with median {age_median:.1f}")
    
    # Gender: mode imputation
    gender_missing = df['gender'].isna().sum()
    gender_mode = df['gender'].mode()[0]
    df['gender'] = df['gender'].fillna(gender_mode)
    print(f"    gender: {gender_missing} missing → filled with mode {gender_mode:.0f}")
    
    # =========================================================================
    # 3. CREATE MASKED COLUMNS FOR 22 FEATURES
    # =========================================================================
    print("\n[3] Creating masked columns (value + mask for each feature)...")
    
    masked_df = pd.DataFrame()
    
    # Keep age and gender (already imputed, no mask needed)
    masked_df['age'] = df['age']
    masked_df['gender'] = df['gender']
    
    # Create value and mask columns for each feature
    for feature in features_to_mask:
        # Value column: original value if present, 0 if missing
        masked_df[f'{feature}'] = df[feature].fillna(0)
        
        # Mask column: 1 if present, 0 if missing
        masked_df[f'{feature}_mask'] = (~df[feature].isna()).astype(int)
        
        # Report
        present = masked_df[f'{feature}_mask'].sum()
        missing = len(df) - present
        pct_missing = missing / len(df) * 100
        print(f"    {feature}: {present:,} present, {missing:,} missing ({pct_missing:.1f}%)")
    
    # Add targets and metadata
    for col in targets + metadata:
        masked_df[col] = df[col]
    
    # =========================================================================
    # 4. VERIFY STRUCTURE
    # =========================================================================
    print("\n[4] Verifying masked dataset structure...")
    
    total_cols = len(masked_df.columns)
    feature_cols = 2 + len(features_to_mask) * 2  # age, gender + 22 values + 22 masks
    target_cols = len(targets)
    meta_cols = len(metadata)
    
    print(f"    Total columns: {total_cols}")
    print(f"    - Simple features (no mask): 2 (age, gender)")
    print(f"    - Value columns: {len(features_to_mask)}")
    print(f"    - Mask columns: {len(features_to_mask)}")
    print(f"    - Target columns: {target_cols}")
    print(f"    - Metadata columns: {meta_cols}")
    
    # Check no NaN in feature columns
    feature_columns = ['age', 'gender'] + features_to_mask + [f'{f}_mask' for f in features_to_mask]
    nan_check = masked_df[feature_columns].isna().sum().sum()
    print(f"\n    NaN remaining in features: {nan_check}")
    if nan_check == 0:
        print("    ✓ All features are complete (no NaN)")
    
    # =========================================================================
    # 5. COLUMN ORDER
    # =========================================================================
    print("\n[5] Organizing column order...")
    
    # Order: simple features → value/mask pairs → targets → metadata
    ordered_columns = ['age', 'gender']
    for feature in features_to_mask:
        ordered_columns.append(feature)
        ordered_columns.append(f'{feature}_mask')
    ordered_columns.extend(targets)
    ordered_columns.extend(metadata)
    
    masked_df = masked_df[ordered_columns]
    
    print(f"    Column order: age, gender, [feature, feature_mask] × 22, targets, metadata")
    
    # =========================================================================
    # 6. SAVE OUTPUT
    # =========================================================================
    print("\n[6] Saving masked dataset...")
    masked_df.to_csv(OUTPUT_PATH, index=False)
    
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {OUTPUT_PATH}")
    print(f"    ✓ Size: {file_size:.1f} MB")
    print(f"    ✓ Shape: {masked_df.shape[0]:,} rows × {masked_df.shape[1]} cols")
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3A COMPLETE: MASKED IMPUTATION")
    print("=" * 70)
    
    print("""
    Strategy: Masking Approach
    --------------------------
    
    Input:  24 features (60% NaN structural missingness)
    Output: 46 feature columns + 4 targets + 1 metadata = 51 columns
    
    Structure:
    - age, gender: Simple imputation (no mask needed)
    - 22 features × 2 = 44 columns:
        - feature_value: Real value or 0 (placeholder)
        - feature_mask: 1 (present) or 0 (missing)
    
    Advantages:
    ✓ NO AMBIGUITY: mask=0 explicitly means "ignore"
    ✓ SCALING SAFE: placeholder 0 won't matter (masked out)
    ✓ MODEL FRIENDLY: FT-Transformer learns attention from masks
    ✓ INTERPRETABLE: Clear which features were used per sample
    
    Next: Step 4A - Scale masked version (exclude mask columns from scaling)
    """)


if __name__ == "__main__":
    main()
