"""
Week 4 - Step 1: Combine Unified Datasets
==========================================

This script combines the 4 unified disease datasets into a single combined dataset.

Input Files:
- datasets/unified/heart_unified.csv (70,000 rows)
- datasets/unified/diabetes_unified.csv (100,000 rows)
- datasets/unified/liver_unified.csv (30,691 rows)
- datasets/unified/kidney_unified.csv (2,032 rows)

Output File:
- datasets/combined_unified.csv (202,723 rows)

Purpose:
- Creates a single dataset for multi-disease model training
- All 4 disease datasets now have identical structure (29 columns)
- Enables training a single FT-Transformer on all diseases simultaneously
"""

import pandas as pd
import numpy as np
from pathlib import Path

def combine_unified_datasets():
    """
    Combine all 4 unified datasets into a single dataset.
    
    Returns:
        pd.DataFrame: Combined dataset with all samples
    """
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    UNIFIED_DIR = PROJECT_ROOT / "datasets" / "unified"
    OUTPUT_PATH = PROJECT_ROOT / "datasets" / "combined_unified.csv"
    
    print("=" * 70)
    print("WEEK 4 - STEP 1: COMBINING UNIFIED DATASETS")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load all unified datasets
    # =========================================================================
    print("\n[1] Loading unified datasets...")
    
    datasets = {}
    dataset_files = [
        "heart_unified.csv",
        "diabetes_unified.csv",
        "liver_unified.csv",
        "kidney_unified.csv"
    ]
    
    for filename in dataset_files:
        filepath = UNIFIED_DIR / filename
        disease_name = filename.replace("_unified.csv", "")
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            datasets[disease_name] = df
            print(f"    ✓ {disease_name}: {len(df):,} rows × {len(df.columns)} cols")
        else:
            print(f"    ✗ {filename}: FILE NOT FOUND")
            return None
    
    # =========================================================================
    # 2. Verify identical structure (same columns, may be different order)
    # =========================================================================
    print("\n[2] Verifying identical column structure...")
    
    reference_cols = set(datasets["heart"].columns)
    all_match = True
    
    for disease, df in datasets.items():
        current_cols = set(df.columns)
        if current_cols != reference_cols:
            missing = reference_cols - current_cols
            extra = current_cols - reference_cols
            print(f"    ✗ {disease}: Column mismatch!")
            if missing:
                print(f"        Missing: {missing}")
            if extra:
                print(f"        Extra: {extra}")
            all_match = False
        else:
            print(f"    ✓ {disease}: {len(df.columns)} columns match")
    
    if not all_match:
        print("\n[ERROR] Column structures don't match. Cannot combine.")
        return None
    
    # Standardize column order across all datasets
    standard_order = list(datasets["heart"].columns)
    for disease in datasets:
        datasets[disease] = datasets[disease][standard_order]
    
    print(f"\n    Common columns: {len(standard_order)}")
    print(f"    ✓ Standardized column order across all datasets")
    
    # =========================================================================
    # 3. Combine datasets (vertical stack)
    # =========================================================================
    print("\n[3] Combining datasets...")
    
    combined_df = pd.concat(
        [datasets["heart"], datasets["diabetes"], datasets["liver"], datasets["kidney"]],
        axis=0,
        ignore_index=True
    )
    
    print(f"    Combined shape: {combined_df.shape[0]:,} rows × {combined_df.shape[1]} cols")
    
    # =========================================================================
    # 4. Verify source distribution
    # =========================================================================
    print("\n[4] Source dataset distribution:")
    
    source_counts = combined_df["source_dataset"].value_counts()
    for source, count in source_counts.items():
        pct = count / len(combined_df) * 100
        print(f"    {source:12s}: {count:>8,} samples ({pct:5.1f}%)")
    
    # =========================================================================
    # 5. Shuffle the data
    # =========================================================================
    print("\n[5] Shuffling data (random seed=42 for reproducibility)...")
    
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"    ✓ Shuffled {len(combined_df):,} rows")
    
    # =========================================================================
    # 6. Data type summary
    # =========================================================================
    print("\n[6] Data types summary:")
    
    dtype_counts = combined_df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    {str(dtype):15s}: {count} columns")
    
    # =========================================================================
    # 7. Missing value overview (preview for Step 2)
    # =========================================================================
    print("\n[7] Missing value overview (preview):")
    
    features = [col for col in combined_df.columns if col not in 
                ['heart_disease_risk', 'diabetes_risk', 'kidney_disease_risk', 
                 'liver_disease_risk', 'source_dataset']]
    
    missing_summary = []
    for col in features:
        missing_count = combined_df[col].isna().sum()
        missing_pct = missing_count / len(combined_df) * 100
        missing_summary.append({
            'feature': col,
            'missing_count': missing_count,
            'missing_pct': missing_pct
        })
    
    # Sort by missing percentage
    missing_summary = sorted(missing_summary, key=lambda x: x['missing_pct'], reverse=True)
    
    print(f"\n    {'Feature':<25s} {'Missing Count':>15s} {'Missing %':>10s}")
    print("    " + "-" * 52)
    
    for item in missing_summary:
        print(f"    {item['feature']:<25s} {item['missing_count']:>15,} {item['missing_pct']:>9.1f}%")
    
    # =========================================================================
    # 8. Target distribution
    # =========================================================================
    print("\n[8] Target variable distribution:")
    
    target_cols = ['heart_disease_risk', 'diabetes_risk', 'kidney_disease_risk', 'liver_disease_risk']
    
    for target in target_cols:
        valid_count = combined_df[target].notna().sum()
        if valid_count > 0:
            mean_val = combined_df[target].mean()
            # For binary targets, show class distribution
            if combined_df[target].dropna().isin([0, 1]).all():
                positive = (combined_df[target] == 1).sum()
                negative = (combined_df[target] == 0).sum()
                ratio = positive / negative if negative > 0 else float('inf')
                print(f"    {target:<25s}: {valid_count:>8,} valid | Pos: {positive:,} | Neg: {negative:,} | Ratio: {ratio:.2f}:1")
            else:
                std_val = combined_df[target].std()
                print(f"    {target:<25s}: {valid_count:>8,} valid | Mean: {mean_val:.3f} | Std: {std_val:.3f}")
        else:
            print(f"    {target:<25s}: All NaN (expected for other disease datasets)")
    
    # =========================================================================
    # 9. Save combined dataset
    # =========================================================================
    print("\n[9] Saving combined dataset...")
    
    combined_df.to_csv(OUTPUT_PATH, index=False)
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    
    print(f"    ✓ Saved to: {OUTPUT_PATH}")
    print(f"    ✓ File size: {file_size_mb:.1f} MB")
    
    # =========================================================================
    # 10. Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE: DATASET COMBINATION")
    print("=" * 70)
    print(f"""
    Combined Dataset Summary:
    -------------------------
    Total Samples:  {len(combined_df):,}
    Total Columns:  {len(combined_df.columns)}
    
    Features:       24
    Targets:        4
    Metadata:       1 (source_dataset)
    
    Source Distribution:
    - Heart:    {source_counts.get('heart', 0):>8,} ({source_counts.get('heart', 0)/len(combined_df)*100:.1f}%)
    - Diabetes: {source_counts.get('diabetes', 0):>8,} ({source_counts.get('diabetes', 0)/len(combined_df)*100:.1f}%)
    - Liver:    {source_counts.get('liver', 0):>8,} ({source_counts.get('liver', 0)/len(combined_df)*100:.1f}%)
    - Kidney:   {source_counts.get('kidney', 0):>8,} ({source_counts.get('kidney', 0)/len(combined_df)*100:.1f}%)
    
    Output File: datasets/combined_unified.csv
    """)
    
    return combined_df


if __name__ == "__main__":
    combined_df = combine_unified_datasets()
