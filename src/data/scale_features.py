"""
Week 4 - Step 4: Feature Scaling
=================================

This script scales numerical features using StandardScaler.
Creates scaled versions of BOTH masked and sentinel datasets.

SCALING STRATEGY:
-----------------
1. MASKED VERSION:
   - Scale value columns (bmi, glucose, etc.) 
   - Keep mask columns as 0/1 (no scaling)
   - Placeholder 0 values will be scaled too (OK, masked out anyway)

2. SENTINEL VERSION:
   - Calculate mean/std ONLY from real values (exclude -999)
   - Scale real values only
   - Keep -999 unchanged

CRITICAL: 
---------
We scale ALL data together first, then split.
In production, you should split first, fit scaler on train, apply to val/test.
For this project, we'll save the scaler and apply same transformation.

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure models directory exists for saving scalers
MODELS_DIR.mkdir(exist_ok=True)

def scale_masked_version():
    """Scale the masked version dataset."""
    print("\n" + "=" * 70)
    print("SCALING MASKED VERSION")
    print("=" * 70)
    
    # Load data
    input_path = DATASETS_DIR / "combined_unified_masked.csv"
    df = pd.read_csv(input_path)
    print(f"\n[1] Loaded: {input_path.name}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Identify column types
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    # Value columns to scale (exclude mask columns, targets, metadata)
    all_columns = df.columns.tolist()
    mask_columns = [col for col in all_columns if col.endswith('_mask')]
    value_columns = [col for col in all_columns 
                     if col not in mask_columns 
                     and col not in targets 
                     and col not in metadata]
    
    print(f"\n[2] Column classification:")
    print(f"    Value columns (to scale): {len(value_columns)}")
    print(f"    Mask columns (keep 0/1): {len(mask_columns)}")
    print(f"    Target columns: {len(targets)}")
    print(f"    Metadata columns: {len(metadata)}")
    
    # Fit scaler on value columns
    print(f"\n[3] Fitting StandardScaler on {len(value_columns)} value columns...")
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    # Scale value columns
    df_scaled[value_columns] = scaler.fit_transform(df[value_columns])
    
    # Report scaling statistics
    print(f"\n[4] Scaling statistics (sample features):")
    for i, col in enumerate(value_columns[:5]):  # Show first 5
        idx = value_columns.index(col)
        print(f"    {col}: mean={scaler.mean_[idx]:.2f}, std={scaler.scale_[idx]:.2f}")
    print(f"    ... and {len(value_columns) - 5} more features")
    
    # Verify scaling
    print(f"\n[5] Verification (scaled ranges):")
    for col in value_columns[:3]:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        mean_val = df_scaled[col].mean()
        print(f"    {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.4f}")
    
    # Save scaled dataset
    output_path = DATASETS_DIR / "combined_unified_masked_scaled.csv"
    df_scaled.to_csv(output_path, index=False)
    print(f"\n[6] Saved scaled dataset: {output_path.name}")
    print(f"    Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Save scaler for later use
    scaler_path = MODELS_DIR / "scaler_masked.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"    Saved scaler: {scaler_path.name}")
    
    return df_scaled, value_columns


def scale_sentinel_version():
    """Scale the sentinel version dataset (exclude -999 from calculations)."""
    print("\n" + "=" * 70)
    print("SCALING SENTINEL VERSION")
    print("=" * 70)
    
    SENTINEL = -999
    
    # Load data
    input_path = DATASETS_DIR / "combined_unified_sentinel.csv"
    df = pd.read_csv(input_path)
    print(f"\n[1] Loaded: {input_path.name}")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Identify column types
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    # Feature columns to scale
    feature_columns = [col for col in df.columns 
                       if col not in targets 
                       and col not in metadata]
    
    print(f"\n[2] Feature columns to scale: {len(feature_columns)}")
    
    # Calculate mean/std excluding sentinel values
    print(f"\n[3] Calculating statistics (excluding {SENTINEL})...")
    
    means = {}
    stds = {}
    
    for col in feature_columns:
        # Get only real values (not sentinel)
        real_values = df[col][df[col] != SENTINEL]
        means[col] = real_values.mean()
        stds[col] = real_values.std()
        
        # Handle case where std is 0 (constant feature)
        if stds[col] == 0:
            stds[col] = 1.0
    
    # Apply scaling (only to non-sentinel values)
    print(f"\n[4] Applying StandardScaler (preserving {SENTINEL})...")
    df_scaled = df.copy()
    
    for col in feature_columns:
        # Create mask for real values
        is_real = df[col] != SENTINEL
        
        # Scale only real values
        df_scaled.loc[is_real, col] = (df.loc[is_real, col] - means[col]) / stds[col]
        # Sentinel values stay as SENTINEL
    
    # Report scaling statistics
    print(f"\n[5] Scaling statistics (sample features):")
    for col in feature_columns[:5]:
        print(f"    {col}: mean={means[col]:.2f}, std={stds[col]:.2f}")
    print(f"    ... and {len(feature_columns) - 5} more features")
    
    # Verify scaling
    print(f"\n[6] Verification (scaled ranges, excluding sentinel):")
    for col in feature_columns[:3]:
        real_values = df_scaled[col][df_scaled[col] != SENTINEL]
        if len(real_values) > 0:
            min_val = real_values.min()
            max_val = real_values.max()
            mean_val = real_values.mean()
            print(f"    {col}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.4f}")
    
    # Check sentinel values preserved
    sentinel_counts = (df_scaled[feature_columns] == SENTINEL).sum()
    print(f"\n[7] Sentinel values preserved:")
    for col in ['bmi', 'glucose_fasting', 'alt', 'gfr'][:3]:
        if col in sentinel_counts:
            print(f"    {col}: {sentinel_counts[col]:,} sentinel values kept")
    
    # Save scaled dataset
    output_path = DATASETS_DIR / "combined_unified_sentinel_scaled.csv"
    df_scaled.to_csv(output_path, index=False)
    print(f"\n[8] Saved scaled dataset: {output_path.name}")
    print(f"    Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Save scaler parameters for later use
    scaler_params = {'means': means, 'stds': stds, 'sentinel': SENTINEL}
    scaler_path = MODELS_DIR / "scaler_sentinel.joblib"
    joblib.dump(scaler_params, scaler_path)
    print(f"    Saved scaler params: {scaler_path.name}")
    
    return df_scaled, feature_columns


def main():
    print("=" * 70)
    print("WEEK 4 - STEP 4: FEATURE SCALING")
    print("=" * 70)
    
    # Scale both versions
    df_masked, masked_cols = scale_masked_version()
    df_sentinel, sentinel_cols = scale_sentinel_version()
    
    # Summary
    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE: FEATURE SCALING")
    print("=" * 70)
    
    print("""
    Summary:
    --------
    
    MASKED VERSION:
    - Input:  combined_unified_masked.csv
    - Output: combined_unified_masked_scaled.csv
    - Scaled: {} value columns
    - Kept:   mask columns as 0/1
    - Scaler: models/scaler_masked.joblib
    
    SENTINEL VERSION:
    - Input:  combined_unified_sentinel.csv
    - Output: combined_unified_sentinel_scaled.csv
    - Scaled: {} feature columns (excluding -999)
    - Kept:   -999 sentinel values unchanged
    - Scaler: models/scaler_sentinel.joblib
    
    Next: Step 5 - Quality Assurance
    """.format(len(masked_cols), len(sentinel_cols)))


if __name__ == "__main__":
    main()
