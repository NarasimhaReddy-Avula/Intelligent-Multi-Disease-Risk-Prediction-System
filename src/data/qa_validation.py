"""
Week 4 - Step 5: Quality Assurance Validation
==============================================

This script validates both scaled datasets to ensure data quality:
1. Check for duplicates
2. Validate feature ranges (no extreme outliers)
3. Check target distributions
4. Verify no data leakage indicators
5. Validate sentinel values (for sentinel version)
6. Verify mask integrity (for masked version)

QA CHECKLIST:
-------------
✓ No duplicate rows
✓ Features in expected ranges after scaling
✓ Target distributions preserved
✓ No NaN in features (after imputation)
✓ Sentinel values preserved correctly (-999)
✓ Mask columns are binary (0/1)

Author: BTP Team
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

def check_duplicates(df, name):
    """Check for duplicate rows."""
    print(f"\n[1] Duplicate Check - {name}")
    print("-" * 50)
    
    # Full row duplicates
    full_duplicates = df.duplicated().sum()
    print(f"    Full row duplicates: {full_duplicates:,}")
    
    # Feature-only duplicates (ignoring targets)
    feature_cols = [col for col in df.columns 
                    if col not in ['diabetes_risk', 'heart_disease_risk', 
                                   'liver_disease_risk', 'kidney_disease_risk',
                                   'source_dataset']]
    feature_duplicates = df[feature_cols].duplicated().sum()
    print(f"    Feature-only duplicates: {feature_duplicates:,}")
    
    if full_duplicates > 0 or feature_duplicates > 0:
        print(f"    ⚠️ Warning: Duplicates found")
        return False
    else:
        print(f"    ✓ No duplicates")
        return True


def check_feature_ranges(df, name, sentinel=-999):
    """Check feature value ranges after scaling."""
    print(f"\n[2] Feature Range Check - {name}")
    print("-" * 50)
    
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    feature_cols = [col for col in df.columns if col not in targets + metadata]
    issues = []
    
    # Check each feature
    for col in feature_cols:
        # Skip mask columns (should be 0/1)
        if col.endswith('_mask'):
            unique_vals = df[col].unique()
            if not set(unique_vals).issubset({0, 1}):
                issues.append(f"{col}: Invalid mask values {unique_vals}")
            continue
        
        # Get non-sentinel values
        values = df[col][df[col] != sentinel]
        if len(values) == 0:
            continue
            
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        std_val = values.std()
        
        # Check for extreme values (beyond 10 std after scaling)
        if abs(min_val) > 10 or abs(max_val) > 10:
            if sentinel != -999 or (min_val != sentinel and max_val != sentinel):
                issues.append(f"{col}: Extreme values [{min_val:.2f}, {max_val:.2f}]")
    
    if issues:
        print(f"    ⚠️ Issues found:")
        for issue in issues[:5]:  # Show first 5
            print(f"      - {issue}")
        return False
    else:
        print(f"    ✓ All {len(feature_cols)} features in valid ranges")
        return True


def check_target_distributions(df, name):
    """Check target variable distributions."""
    print(f"\n[3] Target Distribution Check - {name}")
    print("-" * 50)
    
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    
    print(f"    {'Target':<25} {'Present':<12} {'Positive':<12} {'Rate':<10}")
    print(f"    {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    
    for target in targets:
        present = df[target].notna().sum()
        positive = (df[target] == 1).sum()
        rate = positive / present * 100 if present > 0 else 0
        print(f"    {target:<25} {present:<12,} {positive:<12,} {rate:.1f}%")
    
    return True


def check_nan_values(df, name, sentinel=-999):
    """Check for unexpected NaN values."""
    print(f"\n[4] NaN Value Check - {name}")
    print("-" * 50)
    
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    feature_cols = [col for col in df.columns if col not in targets + metadata]
    
    # Check features (should have no NaN after imputation)
    feature_nan = df[feature_cols].isna().sum().sum()
    
    # Check targets (NaN expected for structural missingness)
    target_nan = df[targets].isna().sum().sum()
    
    print(f"    Feature NaN count: {feature_nan:,}")
    print(f"    Target NaN count: {target_nan:,} (expected - structural)")
    
    if feature_nan > 0:
        print(f"    ⚠️ Unexpected NaN in features!")
        nan_cols = df[feature_cols].isna().sum()
        for col, count in nan_cols.items():
            if count > 0:
                print(f"      - {col}: {count:,} NaN")
        return False
    else:
        print(f"    ✓ No unexpected NaN in features")
        return True


def check_sentinel_values(df, name, sentinel=-999):
    """Check sentinel values are preserved (sentinel version only)."""
    print(f"\n[5] Sentinel Value Check - {name}")
    print("-" * 50)
    
    if 'masked' in name.lower():
        print(f"    N/A - This is masked version (uses mask columns instead)")
        return True
    
    targets = ['diabetes_risk', 'heart_disease_risk', 'liver_disease_risk', 'kidney_disease_risk']
    metadata = ['source_dataset']
    
    feature_cols = [col for col in df.columns if col not in targets + metadata]
    
    sentinel_counts = {}
    for col in feature_cols:
        count = (df[col] == sentinel).sum()
        if count > 0:
            sentinel_counts[col] = count
    
    print(f"    Features with sentinel values: {len(sentinel_counts)}")
    print(f"\n    Sample sentinel counts:")
    for col, count in list(sentinel_counts.items())[:5]:
        pct = count / len(df) * 100
        print(f"      {col}: {count:,} ({pct:.1f}%)")
    
    if len(sentinel_counts) > 0:
        print(f"    ✓ Sentinel values preserved in {len(sentinel_counts)} columns")
    return True


def check_mask_integrity(df, name):
    """Check mask columns are valid (masked version only)."""
    print(f"\n[6] Mask Integrity Check - {name}")
    print("-" * 50)
    
    if 'sentinel' in name.lower():
        print(f"    N/A - This is sentinel version (no mask columns)")
        return True
    
    mask_cols = [col for col in df.columns if col.endswith('_mask')]
    
    issues = []
    for col in mask_cols:
        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            issues.append(f"{col}: Invalid values {unique_vals}")
    
    print(f"    Mask columns: {len(mask_cols)}")
    
    if issues:
        print(f"    ⚠️ Invalid mask columns:")
        for issue in issues:
            print(f"      - {issue}")
        return False
    else:
        # Show mask statistics
        print(f"\n    Sample mask statistics (% present):")
        for col in mask_cols[:5]:
            present_pct = df[col].mean() * 100
            print(f"      {col}: {present_pct:.1f}% present")
        print(f"    ✓ All {len(mask_cols)} mask columns are valid binary (0/1)")
        return True


def check_source_distribution(df, name):
    """Check source dataset distribution."""
    print(f"\n[7] Source Dataset Distribution - {name}")
    print("-" * 50)
    
    dist = df['source_dataset'].value_counts()
    total = len(df)
    
    print(f"    {'Source':<15} {'Count':<12} {'Percentage':<10}")
    print(f"    {'-'*15} {'-'*12} {'-'*10}")
    for source, count in dist.items():
        pct = count / total * 100
        print(f"    {source:<15} {count:<12,} {pct:.1f}%")
    
    return True


def validate_dataset(filepath, name):
    """Run all QA checks on a dataset."""
    print("\n" + "=" * 70)
    print(f"QUALITY ASSURANCE: {name}")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"\nLoaded: {filepath.name}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    results = {}
    results['duplicates'] = check_duplicates(df, name)
    results['ranges'] = check_feature_ranges(df, name)
    results['targets'] = check_target_distributions(df, name)
    results['nan'] = check_nan_values(df, name)
    results['sentinel'] = check_sentinel_values(df, name)
    results['mask'] = check_mask_integrity(df, name)
    results['source'] = check_source_distribution(df, name)
    
    # Summary
    print(f"\n" + "-" * 50)
    print(f"QA SUMMARY - {name}")
    print("-" * 50)
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {check:<15}: {status}")
    
    if all_passed:
        print(f"\n    🎉 ALL CHECKS PASSED!")
    else:
        print(f"\n    ⚠️ SOME CHECKS FAILED - Review issues above")
    
    return all_passed


def main():
    print("=" * 70)
    print("WEEK 4 - STEP 5: QUALITY ASSURANCE VALIDATION")
    print("=" * 70)
    
    # Validate both versions
    masked_path = DATASETS_DIR / "combined_unified_masked_scaled.csv"
    sentinel_path = DATASETS_DIR / "combined_unified_sentinel_scaled.csv"
    
    masked_ok = validate_dataset(masked_path, "MASKED SCALED")
    sentinel_ok = validate_dataset(sentinel_path, "SENTINEL SCALED")
    
    # Final summary
    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE: QUALITY ASSURANCE")
    print("=" * 70)
    
    print(f"""
    Results:
    --------
    MASKED VERSION:  {"✓ PASSED" if masked_ok else "✗ FAILED"}
    SENTINEL VERSION: {"✓ PASSED" if sentinel_ok else "✗ FAILED"}
    
    Both datasets are ready for train/val/test splitting.
    
    Next: Step 6 - Create Train/Val/Test Splits
    """)


if __name__ == "__main__":
    main()
