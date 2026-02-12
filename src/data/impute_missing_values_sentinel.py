"""
Week 4 - Step 3B: Sentinel Imputation Approach
==============================================

STRATEGY: Sentinel Value Approach (24 Features)
------------------------------------------------
Keep original 24 features, replace NaN with sentinel value -999.

WHY -999 (NOT -1)?
------------------
1. CLEAR SEPARATION FROM SCALED VALUES:
   - After StandardScaler, real values typically fall in [-3, +3]
   - Sentinel -1 would overlap with scaled low values
   - Sentinel -999 is clearly outside any realistic range
   
2. EXAMPLE AMBIGUITY WITH -1:
   glucose = [70, 85, ..., 200] → scaled: [-1.38, -1.09, ..., +2.31]
   Sentinel -1 after scaling? Could be confused with real value
   
   glucose = [70, 85, ..., 200, -999] → scaled: [-1.38, ..., +2.31, -999 (kept)]
   Sentinel -999 clearly means "not applicable"

IMPUTATION RULES:
-----------------
1. age, gender: Simple imputation (median/mode) - <1% missing (true random errors)
2. All 22 other features: -999 sentinel value

SCALING STRATEGY (FOR LATER):
-----------------------------
When scaling, EXCLUDE -999 from mean/std calculation:
  1. Mask out -999 values
  2. Calculate mean/std from real values only
  3. Apply StandardScaler to real values
  4. Keep -999 unchanged (don't scale it)

OUTPUT:
-------
- 24 features + 4 targets + 1 metadata = 29 columns (same as input)
- NaN replaced with -999
- Ready for comparison with masked approach

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "datasets" / "combined_unified.csv"
OUTPUT_PATH = PROJECT_ROOT / "datasets" / "combined_unified_sentinel.csv"

# Sentinel value
SENTINEL_VALUE = -999

def main():
    print("=" * 70)
    print("WEEK 4 - STEP 3B: SENTINEL IMPUTATION")
    print("=" * 70)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print("\n[1] Loading combined dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"    ✓ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    
    # Original missing count
    original_nan = df.isna().sum().sum()
    print(f"    Original NaN cells: {original_nan:,}")
    
    # Define column groups
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    # Features with structural missingness (22 features)
    features_with_missing = [
        'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_total', 
        'glucose_fasting', 'hdl', 'ldl', 'triglycerides', 'hba1c',
        'alt', 'ast', 'alp', 'albumin', 'total_protein', 
        'bilirubin_total', 'bilirubin_direct', 'ag_ratio',
        'serum_creatinine', 'bun', 'gfr', 'protein_urine', 'hemoglobin'
    ]
    
    # Features to simply impute (complete, <1% random missing)
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
    # 3. SENTINEL IMPUTATION FOR 22 FEATURES
    # =========================================================================
    print(f"\n[3] Sentinel imputation ({SENTINEL_VALUE}) for 22 features...")
    
    for feature in features_with_missing:
        missing_count = df[feature].isna().sum()
        present_count = len(df) - missing_count
        pct_missing = missing_count / len(df) * 100
        
        # Replace NaN with sentinel
        df[feature] = df[feature].fillna(SENTINEL_VALUE)
        
        print(f"    {feature}: {present_count:,} present, {missing_count:,} → {SENTINEL_VALUE} ({pct_missing:.1f}%)")
    
    # =========================================================================
    # 4. VERIFY NO NaN REMAINING
    # =========================================================================
    print("\n[4] Verifying dataset completeness...")
    
    remaining_nan = df.isna().sum().sum()
    print(f"    NaN remaining: {remaining_nan}")
    
    if remaining_nan == 0:
        print("    ✓ All NaN values replaced")
    else:
        print(f"    ⚠ Warning: {remaining_nan} NaN values remain")
        nan_by_col = df.isna().sum()
        for col, count in nan_by_col.items():
            if count > 0:
                print(f"      - {col}: {count}")
    
    # =========================================================================
    # 5. DOCUMENT SENTINEL STATISTICS
    # =========================================================================
    print("\n[5] Sentinel value statistics...")
    
    print(f"\n    Feature                    Real Values  Sentinel({SENTINEL_VALUE})")
    print("    " + "-" * 55)
    
    for feature in features_with_missing:
        real_count = (df[feature] != SENTINEL_VALUE).sum()
        sentinel_count = (df[feature] == SENTINEL_VALUE).sum()
        print(f"    {feature:25} {real_count:>10,}  {sentinel_count:>10,}")
    
    # =========================================================================
    # 6. SAVE OUTPUT
    # =========================================================================
    print("\n[6] Saving sentinel dataset...")
    df.to_csv(OUTPUT_PATH, index=False)
    
    file_size = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {OUTPUT_PATH}")
    print(f"    ✓ Size: {file_size:.1f} MB")
    print(f"    ✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # =========================================================================
    # 7. SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3B COMPLETE: SENTINEL IMPUTATION")
    print("=" * 70)
    
    print(f"""
    Strategy: Sentinel Value Approach
    ----------------------------------
    
    Sentinel Value: {SENTINEL_VALUE}
    
    Input:  24 features + 4 targets + 1 metadata = 29 columns
    Output: Same structure, NaN replaced with {SENTINEL_VALUE}
    
    Why {SENTINEL_VALUE} instead of -1?
    - After StandardScaler, real values fall in [-3, +3]
    - -1 would overlap with scaled low values → AMBIGUITY
    - {SENTINEL_VALUE} is clearly outside any realistic range → NO AMBIGUITY
    
    Scaling Strategy (for Step 4B):
    - Calculate mean/std ONLY from real values (exclude {SENTINEL_VALUE})
    - Apply scaling to real values
    - Keep {SENTINEL_VALUE} unchanged
    
    Use Case:
    - Comparison with masked approach (ablation study)
    - Tree-based models (XGBoost, Random Forest) 
    - Simpler structure (24 vs 46 features)
    
    Next: Step 4B - Scale sentinel version (protect {SENTINEL_VALUE})
    """)


if __name__ == "__main__":
    main()
