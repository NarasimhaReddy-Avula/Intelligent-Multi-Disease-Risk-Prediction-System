"""
EDA Utilities Module for Multi-Disease AI Healthcare Platform
==============================================================

This module provides reusable functions for Exploratory Data Analysis (EDA)
across all disease datasets in the project. It ensures consistent analysis
methodology and visualization standards.

Author: BTP Project Team
Created: January 27, 2026
Version: 1.0.0

Functions:
    - load_dataset(): Load and display basic dataset info
    - missing_value_analysis(): Analyze and visualize missing values
    - statistical_summary(): Generate comprehensive statistics
    - class_distribution_analysis(): Analyze target class balance
    - distribution_plots(): Plot feature distributions
    - outlier_analysis(): Detect and visualize outliers
    - correlation_analysis(): Generate correlation heatmaps
    - feature_type_analysis(): Categorize features by type
    - generate_eda_report(): Create comprehensive EDA summary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import warnings
from pathlib import Path

# Set visualization defaults for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Color scheme for the project
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Pink
    'success': '#28A745',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'info': '#17A2B8',         # Cyan
    'heart': '#E63946',        # Red for heart disease
    'diabetes': '#457B9D',     # Blue for diabetes
    'kidney': '#2A9D8F',       # Teal for kidney
    'liver': '#E9C46A'         # Yellow for liver
}


def load_dataset(filepath: str, display_info: bool = True) -> pd.DataFrame:
    """
    Load a dataset and display basic information.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    display_info : bool
        Whether to print basic info (default: True)
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
        
    Example:
    --------
    >>> df = load_dataset('datasets/heart_disease.csv')
    """
    df = pd.read_csv(filepath)
    
    if display_info:
        print("=" * 60)
        print("📊 DATASET OVERVIEW")
        print("=" * 60)
        print(f"📁 File: {Path(filepath).name}")
        print(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"🔢 Numeric Columns: {len(df.select_dtypes(include=[np.number]).columns)}")
        print(f"📝 Categorical Columns: {len(df.select_dtypes(include=['object', 'category']).columns)}")
        print("=" * 60)
        
    return df


def missing_value_analysis(df: pd.DataFrame, plot: bool = True, 
                           figsize: Tuple[int, int] = (12, 6)) -> pd.DataFrame:
    """
    Comprehensive missing value analysis with visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    plot : bool
        Whether to generate visualization (default: True)
    figsize : tuple
        Figure size for plots
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics per column
    """
    # Calculate missing value statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2),
        'Data_Type': df.dtypes.values,
        'Unique_Values': df.nunique().values
    })
    
    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
    missing_stats = missing_stats.reset_index(drop=True)
    
    # Summary
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    overall_missing_pct = (total_missing / total_cells * 100)
    
    print("\n" + "=" * 60)
    print("🔍 MISSING VALUE ANALYSIS")
    print("=" * 60)
    print(f"📊 Total Cells: {total_cells:,}")
    print(f"❌ Total Missing: {total_missing:,}")
    print(f"📈 Overall Missing Rate: {overall_missing_pct:.2f}%")
    print(f"📋 Columns with Missing Values: {(missing_stats['Missing_Count'] > 0).sum()}/{len(df.columns)}")
    print("=" * 60)
    
    if plot and missing_stats['Missing_Count'].sum() > 0:
        # Only show columns with missing values
        missing_cols = missing_stats[missing_stats['Missing_Count'] > 0]
        
        if len(missing_cols) > 0:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Bar plot of missing percentages
            ax1 = axes[0]
            colors = [COLORS['danger'] if pct > 30 else COLORS['warning'] if pct > 10 
                      else COLORS['success'] for pct in missing_cols['Missing_Percentage']]
            bars = ax1.barh(missing_cols['Column'], missing_cols['Missing_Percentage'], color=colors)
            ax1.set_xlabel('Missing Percentage (%)')
            ax1.set_title('Missing Values by Column', fontsize=12, fontweight='bold')
            ax1.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Critical (30%)')
            ax1.axvline(x=10, color='orange', linestyle='--', alpha=0.7, label='Warning (10%)')
            ax1.legend()
            
            # Heatmap of missing pattern (sample if too large)
            ax2 = axes[1]
            sample_size = min(100, len(df))
            sample_df = df[missing_cols['Column'].tolist()].sample(n=sample_size, random_state=42)
            sns.heatmap(sample_df.isnull(), cbar=True, yticklabels=False, ax=ax2,
                       cmap='RdYlGn_r', cbar_kws={'label': 'Missing'})
            ax2.set_title(f'Missing Value Pattern (Sample: {sample_size} rows)', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
    elif plot:
        print("\n✅ No missing values found in this dataset!")
        
    return missing_stats


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive statistical summary for numeric columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    pd.DataFrame
        Statistical summary including additional metrics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) == 0:
        print("⚠️ No numeric columns found!")
        return pd.DataFrame()
    
    stats = pd.DataFrame({
        'Count': numeric_df.count(),
        'Mean': numeric_df.mean().round(3),
        'Std': numeric_df.std().round(3),
        'Min': numeric_df.min().round(3),
        'Q1 (25%)': numeric_df.quantile(0.25).round(3),
        'Median': numeric_df.median().round(3),
        'Q3 (75%)': numeric_df.quantile(0.75).round(3),
        'Max': numeric_df.max().round(3),
        'Skewness': numeric_df.skew().round(3),
        'Kurtosis': numeric_df.kurtosis().round(3),
        'IQR': (numeric_df.quantile(0.75) - numeric_df.quantile(0.25)).round(3)
    })
    
    print("\n" + "=" * 60)
    print("📈 STATISTICAL SUMMARY")
    print("=" * 60)
    print(f"📊 Numeric Features: {len(numeric_df.columns)}")
    
    # Highlight potential issues
    highly_skewed = stats[abs(stats['Skewness']) > 1].index.tolist()
    if highly_skewed:
        print(f"⚠️  Highly Skewed Features (|skew| > 1): {len(highly_skewed)}")
        
    print("=" * 60)
    
    return stats


def class_distribution_analysis(df: pd.DataFrame, target_col: str,
                                 figsize: Tuple[int, int] = (10, 4)) -> Dict:
    """
    Analyze target class distribution and check for imbalance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of target column
    figsize : tuple
        Figure size for plots
        
    Returns:
    --------
    dict
        Dictionary with class distribution statistics
    """
    if target_col not in df.columns:
        print(f"⚠️ Column '{target_col}' not found!")
        return {}
    
    # Calculate distribution
    class_counts = df[target_col].value_counts()
    class_pcts = df[target_col].value_counts(normalize=True) * 100
    
    # Determine imbalance ratio
    majority_class = class_counts.max()
    minority_class = class_counts.min()
    imbalance_ratio = majority_class / minority_class
    
    print("\n" + "=" * 60)
    print("⚖️ CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"🎯 Target Column: {target_col}")
    print(f"📊 Number of Classes: {len(class_counts)}")
    print(f"\n📈 Class Distribution:")
    for cls, count in class_counts.items():
        pct = class_pcts[cls]
        print(f"   • Class {cls}: {count:,} samples ({pct:.2f}%)")
    
    print(f"\n⚖️ Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print(f"⚠️  SEVERE IMBALANCE DETECTED! Consider SMOTE or class weights.")
    elif imbalance_ratio > 1.5:
        print(f"⚡ MODERATE IMBALANCE. May need handling during training.")
    else:
        print(f"✅ Classes are relatively balanced.")
    print("=" * 60)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    colors = [COLORS['success'] if i == 0 else COLORS['danger'] for i in range(len(class_counts))]
    axes[0].bar(class_counts.index.astype(str), class_counts.values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Class Distribution: {target_col}', fontsize=12, fontweight='bold')
    
    # Add count labels on bars
    for i, (count, pct) in enumerate(zip(class_counts.values, class_pcts.values)):
        axes[0].text(i, count + count*0.02, f'{count:,}\n({pct:.1f}%)', 
                    ha='center', va='bottom', fontsize=10)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=[f'Class {c}' for c in class_counts.index],
                autopct='%1.1f%%', colors=colors, explode=[0.02]*len(class_counts),
                shadow=True, startangle=90)
    axes[1].set_title('Class Proportion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'class_counts': class_counts.to_dict(),
        'class_percentages': class_pcts.to_dict(),
        'imbalance_ratio': imbalance_ratio,
        'is_imbalanced': imbalance_ratio > 1.5
    }


def distribution_plots(df: pd.DataFrame, columns: Optional[List[str]] = None,
                       n_cols: int = 3, figsize: Tuple[int, int] = (15, 12)) -> None:
    """
    Plot distribution histograms for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    columns : list, optional
        Specific columns to plot. If None, plots all numeric columns.
    n_cols : int
        Number of columns in subplot grid
    figsize : tuple
        Figure size
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(columns) == 0:
        print("⚠️ No numeric columns to plot!")
        return
    
    n_features = len(columns)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    print("\n" + "=" * 60)
    print("📊 FEATURE DISTRIBUTIONS")
    print("=" * 60)
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Plot histogram with KDE
        data = df[col].dropna()
        sns.histplot(data, kde=True, ax=ax, color=COLORS['primary'], edgecolor='white')
        
        # Add vertical lines for mean and median
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color=COLORS['success'], linestyle='-', linewidth=2, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'{col}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlabel('')
        
    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Feature Distributions with Mean & Median', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def outlier_analysis(df: pd.DataFrame, columns: Optional[List[str]] = None,
                     method: str = 'iqr', threshold: float = 1.5,
                     figsize: Tuple[int, int] = (15, 10)) -> pd.DataFrame:
    """
    Detect and visualize outliers using IQR or Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    columns : list, optional
        Specific columns to analyze
    method : str
        Detection method - 'iqr' or 'zscore'
    threshold : float
        IQR multiplier (default 1.5) or Z-score threshold (default 3)
    figsize : tuple
        Figure size
        
    Returns:
    --------
    pd.DataFrame
        Outlier statistics per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_stats = []
    
    print("\n" + "=" * 60)
    print(f"🔎 OUTLIER ANALYSIS (Method: {method.upper()})")
    print("=" * 60)
    
    for col in columns:
        data = df[col].dropna()
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        else:  # zscore
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > threshold]
            lower_bound = data.mean() - threshold * data.std()
            upper_bound = data.mean() + threshold * data.std()
        
        outlier_pct = len(outliers) / len(data) * 100
        
        outlier_stats.append({
            'Column': col,
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': round(outlier_pct, 2),
            'Lower_Bound': round(lower_bound, 3),
            'Upper_Bound': round(upper_bound, 3),
            'Min': data.min(),
            'Max': data.max()
        })
    
    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df = outlier_df.sort_values('Outlier_Percentage', ascending=False)
    
    total_outliers = outlier_df['Outlier_Count'].sum()
    high_outlier_cols = outlier_df[outlier_df['Outlier_Percentage'] > 5]
    
    print(f"📊 Columns Analyzed: {len(columns)}")
    print(f"⚠️  Total Outliers Detected: {total_outliers:,}")
    print(f"🔴 Columns with >5% Outliers: {len(high_outlier_cols)}")
    print("=" * 60)
    
    # Box plots
    n_cols_plot = min(4, len(columns))
    n_rows_plot = (len(columns) + n_cols_plot - 1) // n_cols_plot
    
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=figsize)
    axes = axes.flatten() if len(columns) > 1 else [axes]
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()
        
        # Color based on outlier percentage
        outlier_pct = outlier_df[outlier_df['Column'] == col]['Outlier_Percentage'].values[0]
        color = COLORS['danger'] if outlier_pct > 5 else COLORS['warning'] if outlier_pct > 2 else COLORS['success']
        
        bp = ax.boxplot(data, patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
        
        ax.set_title(f'{col}\n({outlier_pct:.1f}% outliers)', fontsize=9, fontweight='bold')
        ax.set_ylabel('Value')
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Outlier Detection (Box Plots)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
    return outlier_df


def correlation_analysis(df: pd.DataFrame, target_col: Optional[str] = None,
                         method: str = 'pearson', figsize: Tuple[int, int] = (12, 10)) -> pd.DataFrame:
    """
    Generate correlation heatmap and target correlations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    target_col : str, optional
        Target column for specific correlation analysis
    method : str
        Correlation method - 'pearson', 'spearman', or 'kendall'
    figsize : tuple
        Figure size
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        print("⚠️ Need at least 2 numeric columns for correlation analysis!")
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr(method=method)
    
    print("\n" + "=" * 60)
    print(f"🔗 CORRELATION ANALYSIS (Method: {method.upper()})")
    print("=" * 60)
    
    # Find highly correlated pairs (excluding self-correlation)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    round(corr_val, 3)
                ))
    
    if high_corr_pairs:
        print(f"⚠️  Highly Correlated Pairs (|r| > 0.7): {len(high_corr_pairs)}")
        for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
            print(f"   • {col1} ↔ {col2}: {corr}")
    else:
        print("✅ No highly correlated feature pairs found.")
    
    # Heatmap
    fig, axes = plt.subplots(1, 2 if target_col else 1, figsize=figsize)
    
    if target_col:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # Full correlation heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=len(corr_matrix) <= 15, 
                fmt='.2f', cmap='RdBu_r', center=0, ax=ax1,
                square=True, linewidths=0.5,
                annot_kws={'size': 8} if len(corr_matrix) <= 15 else {})
    ax1.set_title(f'Feature Correlation Heatmap ({method.capitalize()})', 
                  fontsize=12, fontweight='bold')
    
    # Target correlation bar plot
    if target_col and target_col in numeric_df.columns:
        target_corr = corr_matrix[target_col].drop(target_col).sort_values()
        colors = [COLORS['danger'] if c < 0 else COLORS['success'] for c in target_corr.values]
        target_corr.plot(kind='barh', ax=ax2, color=colors, edgecolor='black')
        ax2.set_xlabel('Correlation Coefficient')
        ax2.set_title(f'Correlation with Target: {target_col}', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        print(f"\n🎯 Top 5 Features Correlated with {target_col}:")
        top_corr = abs(corr_matrix[target_col].drop(target_col)).sort_values(ascending=False).head(5)
        for feat, corr_val in top_corr.items():
            actual_corr = corr_matrix[target_col][feat]
            sign = "+" if actual_corr > 0 else ""
            print(f"   • {feat}: {sign}{actual_corr:.3f}")
    
    print("=" * 60)
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def feature_type_analysis(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize features by type (numeric continuous, numeric discrete, categorical).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
        
    Returns:
    --------
    dict
        Dictionary with feature categories
    """
    feature_types = {
        'continuous': [],
        'discrete': [],
        'binary': [],
        'categorical': [],
        'high_cardinality': []
    }
    
    print("\n" + "=" * 60)
    print("📋 FEATURE TYPE ANALYSIS")
    print("=" * 60)
    
    for col in df.columns:
        n_unique = df[col].nunique()
        dtype = df[col].dtype
        
        if dtype in ['object', 'category']:
            if n_unique > 20:
                feature_types['high_cardinality'].append(col)
            else:
                feature_types['categorical'].append(col)
        elif np.issubdtype(dtype, np.number):
            if n_unique == 2:
                feature_types['binary'].append(col)
            elif n_unique <= 20:
                feature_types['discrete'].append(col)
            else:
                feature_types['continuous'].append(col)
    
    for ftype, features in feature_types.items():
        if features:
            print(f"\n📌 {ftype.upper()} ({len(features)} features):")
            print(f"   {', '.join(features[:10])}" + ("..." if len(features) > 10 else ""))
    
    print("\n" + "=" * 60)
    
    return feature_types


def generate_eda_report(df: pd.DataFrame, dataset_name: str, target_col: str,
                        disease_color: str = 'primary') -> Dict:
    """
    Generate a comprehensive EDA report with all analyses.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    dataset_name : str
        Name of the dataset (e.g., 'Heart Disease')
    target_col : str
        Target column name
    disease_color : str
        Color key from COLORS dict
        
    Returns:
    --------
    dict
        Complete EDA report with all statistics
    """
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + f"  📊 EDA REPORT: {dataset_name.upper()}".ljust(57) + "║")
    print("╚" + "═" * 58 + "╝")
    
    report = {
        'dataset_name': dataset_name,
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # 1. Basic Info
    load_dataset.__doc__  # Just to reference
    print(f"\n📁 Dataset: {dataset_name}")
    print(f"📐 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"💾 Memory: {report['memory_mb']:.2f} MB")
    
    # 2. Missing Values
    report['missing_stats'] = missing_value_analysis(df)
    
    # 3. Statistical Summary
    report['statistics'] = statistical_summary(df)
    
    # 4. Class Distribution
    report['class_distribution'] = class_distribution_analysis(df, target_col)
    
    # 5. Feature Types
    report['feature_types'] = feature_type_analysis(df)
    
    # 6. Distributions
    distribution_plots(df)
    
    # 7. Outliers
    report['outlier_stats'] = outlier_analysis(df)
    
    # 8. Correlations
    report['correlations'] = correlation_analysis(df, target_col)
    
    # Final Summary
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  📋 EDA SUMMARY & RECOMMENDATIONS".ljust(57) + "║")
    print("╚" + "═" * 58 + "╝")
    
    recommendations = []
    
    # Missing value recommendations
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 0:
        recommendations.append(f"⚠️ Handle {missing_pct:.2f}% missing values")
    
    # Class imbalance recommendations
    if report['class_distribution'].get('is_imbalanced', False):
        ratio = report['class_distribution']['imbalance_ratio']
        recommendations.append(f"⚠️ Address class imbalance (ratio: {ratio:.1f}:1)")
    
    # Outlier recommendations
    high_outlier_cols = report['outlier_stats'][report['outlier_stats']['Outlier_Percentage'] > 5]
    if len(high_outlier_cols) > 0:
        recommendations.append(f"⚠️ Treat outliers in {len(high_outlier_cols)} columns")
    
    if recommendations:
        print("\n🔧 PREPROCESSING RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("\n✅ Dataset is in good shape!")
    
    print("\n" + "=" * 60)
    
    return report


def save_figure(fig: plt.Figure, filename: str, output_dir: str = 'notebooks/eda/figures') -> str:
    """
    Save a matplotlib figure to the specified directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name of the file (without extension)
    output_dir : str
        Output directory path
        
    Returns:
    --------
    str
        Full path to saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / f"{filename}.png"
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📁 Figure saved: {filepath}")
    
    return str(filepath)


# ============================================================================
# CROSS-DISEASE COMPARISON UTILITIES
# ============================================================================

def compare_datasets(datasets: Dict[str, pd.DataFrame], target_cols: Dict[str, str]) -> pd.DataFrame:
    """
    Compare multiple datasets for schema alignment planning.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary of {name: dataframe}
    target_cols : dict
        Dictionary of {name: target_column_name}
        
    Returns:
    --------
    pd.DataFrame
        Comparison summary
    """
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🔄 CROSS-DATASET COMPARISON".ljust(57) + "║")
    print("╚" + "═" * 58 + "╝")
    
    comparison = []
    
    for name, df in datasets.items():
        target = target_cols.get(name, 'Unknown')
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        
        # Get class imbalance
        if target in df.columns:
            class_counts = df[target].value_counts()
            imbalance = class_counts.max() / class_counts.min()
        else:
            imbalance = np.nan
        
        comparison.append({
            'Dataset': name,
            'Rows': df.shape[0],
            'Columns': df.shape[1],
            'Missing_%': round(missing_pct, 2),
            'Target': target,
            'Imbalance_Ratio': round(imbalance, 2) if not np.isnan(imbalance) else 'N/A',
            'Numeric_Cols': len(df.select_dtypes(include=[np.number]).columns),
            'Categorical_Cols': len(df.select_dtypes(include=['object', 'category']).columns)
        })
    
    comparison_df = pd.DataFrame(comparison)
    
    print("\n📊 Dataset Comparison Summary:")
    print(comparison_df.to_string(index=False))
    
    # Find common features (by similar names)
    print("\n🔗 Feature Mapping Hints:")
    all_cols = {}
    for name, df in datasets.items():
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            if col_lower not in all_cols:
                all_cols[col_lower] = []
            all_cols[col_lower].append((name, col))
    
    common_features = {k: v for k, v in all_cols.items() if len(v) > 1}
    if common_features:
        print("   Potentially mappable features across datasets:")
        for feat_key, occurrences in list(common_features.items())[:10]:
            print(f"   • {[f'{name}: {col}' for name, col in occurrences]}")
    
    return comparison_df


if __name__ == "__main__":
    print("EDA Utilities Module - Ready for use!")
    print("Import with: from src.utils.eda_utils import *")
