"""
Week 4 - Step 6: Train/Val/Test Splits
=======================================

This script creates stratified train/validation/test splits for both
masked and sentinel versions of the dataset.

SPLIT RATIOS:
-------------
- Train: 70% (~133,954 samples)
- Validation: 15% (~28,705 samples)
- Test: 15% (~28,704 samples)

STRATIFICATION:
---------------
Stratified by source_dataset to ensure proportional representation:
- Diabetes: 52.3%
- Heart: 36.6%
- Liver: 10.1%
- Kidney: 1.1%

Each split will have approximately these same ratios.

CRITICAL - DATA LEAKAGE PREVENTION:
-----------------------------------
1. Split happens AFTER scaling (using training set scaler params)
2. Same split indices used for both masked and sentinel versions
3. Random seed fixed for reproducibility

OUTPUT FILES:
-------------
For each version (masked/sentinel):
- datasets/splits/masked/train.csv
- datasets/splits/masked/val.csv
- datasets/splits/masked/test.csv
- datasets/splits/sentinel/train.csv
- datasets/splits/sentinel/val.csv
- datasets/splits/sentinel/test.csv

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
SPLITS_DIR = DATASETS_DIR / "splits"

# Create split directories
(SPLITS_DIR / "masked").mkdir(parents=True, exist_ok=True)
(SPLITS_DIR / "sentinel").mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_stratified_splits(df, stratify_col='source_dataset'):
    """Create stratified train/val/test splits."""
    
    # First split: train+val / test (85% / 15%)
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_RATIO,
        stratify=df[stratify_col],
        random_state=RANDOM_SEED
    )
    
    # Second split: train / val (70% / 15% of original = 82.4% / 17.6% of train_val)
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=df.iloc[train_val_idx][stratify_col],
        random_state=RANDOM_SEED
    )
    
    return train_idx, val_idx, test_idx


def verify_split_distribution(df_train, df_val, df_test, stratify_col='source_dataset'):
    """Verify stratification is preserved."""
    print("\n    Split distribution by source_dataset:")
    print(f"    {'Source':<12} {'Train':<15} {'Val':<15} {'Test':<15}")
    print(f"    {'-'*12} {'-'*15} {'-'*15} {'-'*15}")
    
    for source in df_train[stratify_col].unique():
        train_pct = (df_train[stratify_col] == source).mean() * 100
        val_pct = (df_val[stratify_col] == source).mean() * 100
        test_pct = (df_test[stratify_col] == source).mean() * 100
        print(f"    {source:<12} {train_pct:.1f}%          {val_pct:.1f}%          {test_pct:.1f}%")


def verify_target_distribution(df_train, df_val, df_test):
    """Verify target distributions are similar across splits."""
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    
    print("\n    Target positive rates:")
    print(f"    {'Target':<25} {'Train':<10} {'Val':<10} {'Test':<10}")
    print(f"    {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    
    for target in targets:
        train_rate = (df_train[target] == 1).sum() / df_train[target].notna().sum() * 100 if df_train[target].notna().sum() > 0 else 0
        val_rate = (df_val[target] == 1).sum() / df_val[target].notna().sum() * 100 if df_val[target].notna().sum() > 0 else 0
        test_rate = (df_test[target] == 1).sum() / df_test[target].notna().sum() * 100 if df_test[target].notna().sum() > 0 else 0
        print(f"    {target:<25} {train_rate:.1f}%      {val_rate:.1f}%      {test_rate:.1f}%")


def split_and_save(input_path, output_dir, name, split_indices):
    """Split dataset using pre-computed indices and save."""
    print(f"\n{'='*60}")
    print(f"SPLITTING: {name}")
    print(f"{'='*60}")
    
    train_idx, val_idx, test_idx = split_indices
    
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"\nLoaded: {input_path.name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Split using indices
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    print(f"\n[1] Split sizes:")
    print(f"    Train: {len(df_train):,} ({len(df_train)/len(df)*100:.1f}%)")
    print(f"    Val:   {len(df_val):,} ({len(df_val)/len(df)*100:.1f}%)")
    print(f"    Test:  {len(df_test):,} ({len(df_test)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print(f"\n[2] Verifying stratification...")
    verify_split_distribution(df_train, df_val, df_test)
    
    # Verify target distributions
    print(f"\n[3] Verifying target distributions...")
    verify_target_distribution(df_train, df_val, df_test)
    
    # Save splits
    print(f"\n[4] Saving splits...")
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"    ✓ Saved: {train_path.name} ({train_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"    ✓ Saved: {val_path.name} ({val_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"    ✓ Saved: {test_path.name} ({test_path.stat().st_size / (1024*1024):.1f} MB)")
    
    return df_train, df_val, df_test


def main():
    print("=" * 70)
    print("WEEK 4 - STEP 6: TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    # Load one dataset to compute indices (use sentinel as reference)
    df_reference = pd.read_csv(DATASETS_DIR / "combined_unified_sentinel_scaled.csv")
    print(f"\nReference dataset: combined_unified_sentinel_scaled.csv")
    print(f"Total samples: {len(df_reference):,}")
    
    # Compute split indices (same for both versions)
    print(f"\n[1] Computing stratified split indices...")
    print(f"    Stratification: source_dataset")
    print(f"    Random seed: {RANDOM_SEED}")
    
    train_idx, val_idx, test_idx = create_stratified_splits(df_reference)
    split_indices = (train_idx, val_idx, test_idx)
    
    print(f"    Train indices: {len(train_idx):,}")
    print(f"    Val indices: {len(val_idx):,}")
    print(f"    Test indices: {len(test_idx):,}")
    
    # Split and save masked version
    split_and_save(
        DATASETS_DIR / "combined_unified_masked_scaled.csv",
        SPLITS_DIR / "masked",
        "MASKED SCALED",
        split_indices
    )
    
    # Split and save sentinel version
    split_and_save(
        DATASETS_DIR / "combined_unified_sentinel_scaled.csv",
        SPLITS_DIR / "sentinel",
        "SENTINEL SCALED",
        split_indices
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("STEP 6 COMPLETE: TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    
    print("""
    Summary:
    --------
    Split Ratios: 70% train / 15% val / 15% test
    Stratified by: source_dataset (disease type)
    Random Seed: 42 (reproducible)
    
    Files Created:
    --------------
    MASKED VERSION:
    - datasets/splits/masked/train.csv
    - datasets/splits/masked/val.csv
    - datasets/splits/masked/test.csv
    
    SENTINEL VERSION:
    - datasets/splits/sentinel/train.csv
    - datasets/splits/sentinel/val.csv
    - datasets/splits/sentinel/test.csv
    
    PREPROCESSING COMPLETE!
    -----------------------
    Both datasets are now ready for model training:
    - Week 6: ML Baselines (XGBoost, Random Forest) using sentinel version
    - Week 7: FT-Transformer using masked version
    """)
    
    # Save split indices for reproducibility
    import joblib
    indices_path = SPLITS_DIR / "split_indices.joblib"
    joblib.dump({
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'random_seed': RANDOM_SEED
    }, indices_path)
    print(f"    ✓ Saved split indices: {indices_path.name}")


if __name__ == "__main__":
    main()
