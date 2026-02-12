"""
Week 4 - Step 2: Missing Value Analysis
========================================

This script analyzes missing value patterns in the combined unified dataset.

Input:
- datasets/combined_unified.csv (202,723 rows)

Output:
- Missing value report
- Visualizations (heatmap, bar charts)
- Imputation strategy recommendations

Purpose:
- Understand which features need imputation
- Identify missing data patterns (MCAR, MAR, MNAR)
- Plan imputation strategy for Step 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_missing_values():
    """
    Comprehensive missing value analysis.
    """
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "datasets" / "combined_unified.csv"
    OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "eda" / "figures" / "week4"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("WEEK 4 - STEP 2: MISSING VALUE ANALYSIS")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load combined dataset
    # =========================================================================
    print("\n[1] Loading combined dataset...")
    
    df = pd.read_csv(DATA_PATH)
    print(f"    ✓ Loaded: {len(df):,} rows × {len(df.columns)} cols")
    
    # Separate features, targets, metadata
    target_cols = ['heart_disease_risk', 'diabetes_risk', 'kidney_disease_risk', 'liver_disease_risk']
    metadata_cols = ['source_dataset']
    feature_cols = [col for col in df.columns if col not in target_cols + metadata_cols]
    
    print(f"    Features: {len(feature_cols)}")
    print(f"    Targets:  {len(target_cols)}")
    
    # =========================================================================
    # 2. Overall missing statistics
    # =========================================================================
    print("\n[2] Overall missing statistics...")
    
    total_cells = len(df) * len(feature_cols)
    total_missing = df[feature_cols].isna().sum().sum()
    overall_missing_pct = (total_missing / total_cells) * 100
    
    print(f"    Total cells (features only): {total_cells:,}")
    print(f"    Missing cells: {total_missing:,}")
    print(f"    Overall missing %: {overall_missing_pct:.2f}%")
    
    # =========================================================================
    # 3. Per-feature missing analysis
    # =========================================================================
    print("\n[3] Per-feature missing analysis...")
    
    missing_summary = []
    for col in feature_cols:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        present_count = len(df) - missing_count
        
        missing_summary.append({
            'feature': col,
            'missing_count': missing_count,
            'present_count': present_count,
            'missing_pct': missing_pct
        })
    
    # Sort by missing percentage (descending)
    missing_df = pd.DataFrame(missing_summary).sort_values('missing_pct', ascending=False)
    
    print(f"\n    {'Feature':<25s} {'Present':>12s} {'Missing':>12s} {'Missing %':>10s}")
    print("    " + "-" * 61)
    
    for _, row in missing_df.iterrows():
        print(f"    {row['feature']:<25s} {row['present_count']:>12,} {row['missing_count']:>12,} {row['missing_pct']:>9.1f}%")
    
    # =========================================================================
    # 4. Categorize features by missingness
    # =========================================================================
    print("\n[4] Categorizing features by missingness level...")
    
    # Define categories
    categories = {
        'Complete (0-5%)': missing_df[missing_df['missing_pct'] < 5],
        'Low Missing (5-25%)': missing_df[(missing_df['missing_pct'] >= 5) & (missing_df['missing_pct'] < 25)],
        'Medium Missing (25-50%)': missing_df[(missing_df['missing_pct'] >= 25) & (missing_df['missing_pct'] < 50)],
        'High Missing (50-90%)': missing_df[(missing_df['missing_pct'] >= 50) & (missing_df['missing_pct'] < 90)],
        'Very High Missing (90%+)': missing_df[missing_df['missing_pct'] >= 90]
    }
    
    for category, cat_df in categories.items():
        if len(cat_df) > 0:
            print(f"\n    {category}: {len(cat_df)} features")
            for _, row in cat_df.iterrows():
                print(f"      - {row['feature']:<25s} ({row['missing_pct']:5.1f}%)")
    
    # =========================================================================
    # 5. Missing by source dataset
    # =========================================================================
    print("\n[5] Missing patterns by source dataset...")
    
    for source in df['source_dataset'].unique():
        source_df = df[df['source_dataset'] == source]
        print(f"\n    {source.upper()} dataset ({len(source_df):,} samples):")
        
        # Calculate coverage for this source
        source_missing = source_df[feature_cols].isna().mean() * 100
        source_missing_sorted = source_missing.sort_values()
        
        # Show features with data (< 50% missing)
        features_with_data = source_missing_sorted[source_missing_sorted < 50]
        if len(features_with_data) > 0:
            print(f"      Features available ({len(features_with_data)}):")
            for feat, miss_pct in features_with_data.items():
                coverage_pct = 100 - miss_pct
                print(f"        {feat:<25s}: {coverage_pct:5.1f}% coverage")
    
    # =========================================================================
    # 6. Imputation strategy recommendations
    # =========================================================================
    print("\n[6] Imputation strategy recommendations...")
    
    print("\n    COMPLETE (0-5% missing) - Simple imputation:")
    complete_features = missing_df[missing_df['missing_pct'] < 5]
    for _, row in complete_features.iterrows():
        print(f"      {row['feature']:<25s} → Median imputation")
    
    print("\n    LOW-MEDIUM (5-50% missing) - KNN imputation:")
    impute_features = missing_df[(missing_df['missing_pct'] >= 5) & (missing_df['missing_pct'] < 50)]
    for _, row in impute_features.iterrows():
        print(f"      {row['feature']:<25s} → KNN (k=5)")
    
    print("\n    HIGH (50%+ missing) - Keep as NaN (disease-specific markers):")
    keep_nan_features = missing_df[missing_df['missing_pct'] >= 50]
    for _, row in keep_nan_features.iterrows():
        print(f"      {row['feature']:<25s} → Keep NaN (FT-Transformer handles)")
    
    # =========================================================================
    # 7. Visualization: Missing value heatmap
    # =========================================================================
    print("\n[7] Creating missing value visualizations...")
    
    # Sample for visualization (too many rows to plot all)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)[feature_cols]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create binary missing indicator
    missing_matrix = df_sample.isna().astype(int)
    
    # Plot
    sns.heatmap(missing_matrix.T, 
                cbar=False, 
                cmap=['lightblue', 'darkred'],
                xticklabels=False,
                yticklabels=True,
                ax=ax)
    
    ax.set_title(f'Missing Value Pattern (Sample of {sample_size:,} rows)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Present'),
        Patch(facecolor='darkred', label='Missing')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "01_missing_value_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved heatmap: {heatmap_path}")
    plt.close()
    
    # =========================================================================
    # 8. Visualization: Missing percentage bar chart
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color code by category
    colors = []
    for pct in missing_df['missing_pct']:
        if pct < 5:
            colors.append('green')
        elif pct < 25:
            colors.append('yellowgreen')
        elif pct < 50:
            colors.append('orange')
        elif pct < 90:
            colors.append('orangered')
        else:
            colors.append('darkred')
    
    ax.barh(range(len(missing_df)), missing_df['missing_pct'], color=colors)
    ax.set_yticks(range(len(missing_df)))
    ax.set_yticklabels(missing_df['feature'])
    ax.set_xlabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
    ax.axvline(x=50, color='red', linestyle='--', linewidth=1, alpha=0.7, label='50% threshold')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()
    
    # Add percentage labels
    for i, (_, row) in enumerate(missing_df.iterrows()):
        ax.text(row['missing_pct'] + 1, i, f"{row['missing_pct']:.1f}%", 
                va='center', fontsize=8)
    
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "02_missing_percentages.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved bar chart: {bar_path}")
    plt.close()
    
    # =========================================================================
    # 9. Visualization: Missing by source dataset
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sources = df['source_dataset'].unique()
    x = np.arange(len(feature_cols))
    width = 0.2
    
    for i, source in enumerate(sources):
        source_df = df[df['source_dataset'] == source]
        source_missing_pct = source_df[feature_cols].isna().mean() * 100
        
        ax.bar(x + i*width, source_missing_pct, width, label=source.capitalize())
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Missing Percentage (%)', fontsize=12)
    ax.set_title('Missing Patterns by Source Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(feature_cols, rotation=90, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout()
    source_path = OUTPUT_DIR / "03_missing_by_source.png"
    plt.savefig(source_path, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved source comparison: {source_path}")
    plt.close()
    
    # =========================================================================
    # 10. Save summary report
    # =========================================================================
    print("\n[8] Saving analysis report...")
    
    report_path = OUTPUT_DIR / "missing_value_analysis.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MISSING VALUE ANALYSIS REPORT\n")
        f.write("Week 4 - Step 2\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: combined_unified.csv\n")
        f.write(f"Total samples: {len(df):,}\n")
        f.write(f"Total features: {len(feature_cols)}\n")
        f.write(f"Overall missing: {overall_missing_pct:.2f}%\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("IMPUTATION STRATEGY\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("COMPLETE (0-5%) - Median imputation:\n")
        for _, row in complete_features.iterrows():
            f.write(f"  - {row['feature']} ({row['missing_pct']:.1f}%)\n")
        
        f.write("\nLOW-MEDIUM (5-50%) - KNN imputation (k=5):\n")
        for _, row in impute_features.iterrows():
            f.write(f"  - {row['feature']} ({row['missing_pct']:.1f}%)\n")
        
        f.write("\nHIGH (50%+) - Keep as NaN (FT-Transformer handles):\n")
        for _, row in keep_nan_features.iterrows():
            f.write(f"  - {row['feature']} ({row['missing_pct']:.1f}%)\n")
        
        f.write("\n" + "-" * 70 + "\n")
        f.write("RATIONALE\n")
        f.write("-" * 70 + "\n\n")
        f.write("Disease-specific markers (>50% missing) are kept as NaN because:\n")
        f.write("1. They are only present in respective disease datasets\n")
        f.write("2. FT-Transformer's attention mechanism handles sparse features naturally\n")
        f.write("3. Imputing 70%+ missing values would create artificial patterns\n")
        f.write("4. Model learns when markers are present vs absent\n")
    
    print(f"    ✓ Saved report: {report_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE: MISSING VALUE ANALYSIS")
    print("=" * 70)
    print(f"""
    Analysis Summary:
    -----------------
    Features to impute (median):     {len(complete_features)}
    Features to impute (KNN):        {len(impute_features)}
    Features to keep as NaN:         {len(keep_nan_features)}
    
    Outputs:
    --------
    - Heatmap visualization
    - Bar chart (missing %)
    - Source comparison plot
    - Strategy report (TXT)
    
    Files saved to: {OUTPUT_DIR}
    
    Next: Step 3 will apply the imputation strategy.
    """)
    
    return missing_df


if __name__ == "__main__":
    missing_analysis = analyze_missing_values()
