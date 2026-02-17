"""
Week 4 - Step 5B: Fix Data Issues
==================================

Issues found during QA:
1. liver_disease_risk uses 1/2 instead of 0/1 (1=disease, 2=no disease)
2. diabetes_risk is continuous (0.027-0.672), not binary
3. Liver dataset has 11,323 duplicates (37%)

FIXES:
------
1. Convert liver_disease_risk: 1→1, 2→0
2. Binarize diabetes_risk: >0.5 = 1 (high risk), <=0.5 = 0 (low risk)
   (Or keep continuous for regression tasks)
3. Remove exact duplicates from liver dataset

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def fix_liver_target(df):
    """Convert liver_disease_risk from 1/2 to 0/1."""
    print("\n[1] Fixing liver_disease_risk (1/2 → 0/1)...")
    
    # Current values
    before = df['liver_disease_risk'].value_counts(dropna=False)
    print(f"    Before: {dict(before)}")
    
    # Convert: 1 stays 1 (disease), 2 becomes 0 (no disease)
    df.loc[df['liver_disease_risk'] == 2, 'liver_disease_risk'] = 0
    
    # Verify
    after = df['liver_disease_risk'].value_counts(dropna=False)
    print(f"    After:  {dict(after)}")
    
    return df


def fix_diabetes_target(df, threshold=0.5):
    """Binarize diabetes_risk at threshold."""
    print(f"\n[2] Binarizing diabetes_risk (threshold={threshold})...")
    
    # Only binarize non-NaN values
    mask = df['diabetes_risk'].notna()
    
    # Before
    before_mean = df.loc[mask, 'diabetes_risk'].mean()
    print(f"    Before: mean={before_mean:.3f}, min={df.loc[mask, 'diabetes_risk'].min():.3f}, max={df.loc[mask, 'diabetes_risk'].max():.3f}")
    
    # Binarize
    df.loc[mask, 'diabetes_risk'] = (df.loc[mask, 'diabetes_risk'] > threshold).astype(float)
    
    # After
    after_counts = df['diabetes_risk'].value_counts(dropna=False)
    print(f"    After:  {dict(after_counts)}")
    
    return df


def remove_duplicates(df):
    """Remove duplicate rows."""
    print("\n[3] Removing duplicates...")
    
    # Count by source before
    print("    Before:")
    for source in df['source_dataset'].unique():
        subset = df[df['source_dataset'] == source]
        dups = subset.duplicated().sum()
        print(f"      {source}: {len(subset):,} rows, {dups:,} duplicates")
    
    # Remove duplicates
    before_len = len(df)
    df = df.drop_duplicates()
    after_len = len(df)
    removed = before_len - after_len
    
    print(f"\n    Removed: {removed:,} duplicate rows")
    print(f"    Final: {after_len:,} rows")
    
    # Count by source after
    print("\n    After:")
    for source in df['source_dataset'].unique():
        subset = df[df['source_dataset'] == source]
        print(f"      {source}: {len(subset):,} rows")
    
    return df


def fix_dataset(input_path, output_path, name):
    """Fix a single dataset."""
    print(f"\n{'='*60}")
    print(f"FIXING: {name}")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    print(f"\nLoaded: {input_path.name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Apply fixes
    df = fix_liver_target(df)
    df = fix_diabetes_target(df, threshold=0.5)
    df = remove_duplicates(df)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path.name}")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    return df


def main():
    print("=" * 70)
    print("WEEK 4 - STEP 5B: FIX DATA ISSUES")
    print("=" * 70)
    
    # Fix original combined dataset first
    print("\n>>> Fixing ORIGINAL combined dataset...")
    df_original = fix_dataset(
        DATASETS_DIR / "combined_unified.csv",
        DATASETS_DIR / "combined_unified.csv",  # Overwrite
        "ORIGINAL"
    )
    
    # Now recreate masked and sentinel from fixed original
    print("\n>>> Recreating MASKED version from fixed data...")
    # We need to re-run imputation scripts
    print("    (Re-running imputation will happen next)")
    
    # Summary
    print("\n" + "=" * 70)
    print("FIXES APPLIED")
    print("=" * 70)
    
    print("""
    1. liver_disease_risk: Converted 1/2 → 1/0 (1=disease, 0=no disease)
    
    2. diabetes_risk: Binarized at 0.5 threshold
       - >0.5  → 1 (high risk)
       - <=0.5 → 0 (low risk)
    
    3. Duplicates: Removed ~11,360 duplicate rows
       - Final dataset: ~191,363 rows (was 202,723)
    
    NEXT: Re-run imputation scripts to update masked/sentinel versions
    """)


if __name__ == "__main__":
    main()
