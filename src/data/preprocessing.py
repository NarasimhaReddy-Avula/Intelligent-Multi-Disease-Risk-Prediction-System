"""
Data Preprocessing Module

This module provides functions for preprocessing disease datasets:
- Missing value handling
- Outlier detection and treatment
- Feature scaling
- Train/validation/test splitting

Author: BTP Project 2025-2026
Last Updated: February 12, 2026
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
SCALER_DIR = PROJECT_ROOT / 'models' / 'scalers'

# Create directories if they don't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'auto',
                          disease_type: Optional[str] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        strategy: 'auto', 'median', 'mean', 'mode', 'drop'
        disease_type: Optional disease type for specific strategies
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    logger.info(f"Missing values before: {df_clean.isnull().sum().sum()}")
    
    if strategy == 'auto' or strategy == 'median':
        # For numeric columns, fill with median
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} with median: {median_val:.2f}")
        
        # For categorical columns, fill with mode
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                logger.info(f"  Filled {col} with mode: {mode_val}")
    
    elif strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                df_clean[col].fillna(mean_val, inplace=True)
    
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
        logger.info(f"  Dropped rows with missing values")
    
    logger.info(f"Missing values after: {df_clean.isnull().sum().sum()}")
    logger.info(f"Rows remaining: {len(df_clean)} / {len(df)}")
    
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, 
                        column: str,
                        multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        multiplier: IQR multiplier (default 1.5)
    
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    if outliers.any():
        logger.info(f"  {column}: {outliers.sum()} outliers detected "
                   f"(bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
    
    return outliers


def handle_outliers(df: pd.DataFrame, 
                    method: str = 'cap',
                    multiplier: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in numeric columns.
    
    Args:
        df: Input DataFrame
        method: 'cap' (winsorize), 'remove', or 'keep'
        multiplier: IQR multiplier for outlier detection
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    logger.info(f"Handling outliers using method: {method}")
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        if method == 'cap':
            # Cap outliers at bounds (winsorization)
            original_outliers = ((df_clean[col] < lower_bound) | 
                               (df_clean[col] > upper_bound)).sum()
            
            if original_outliers > 0:
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"  Capped {original_outliers} outliers in {col}")
        
        elif method == 'remove':
            # Remove rows with outliers
            mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
            removed = len(df_clean) - mask.sum()
            if removed > 0:
                df_clean = df_clean[mask]
                logger.info(f"  Removed {removed} outlier rows for {col}")
    
    logger.info(f"Final dataset size: {len(df_clean)} rows")
    return df_clean


def scale_features(df: pd.DataFrame,
                   scaler_type: str = 'standard',
                   exclude_cols: Optional[list] = None) -> Tuple[pd.DataFrame, object]:
    """
    Scale numeric features using StandardScaler or MinMaxScaler.
    
    Args:
        df: Input DataFrame
        scaler_type: 'standard' or 'minmax'
        exclude_cols: Columns to exclude from scaling (e.g., target variable)
    
    Returns:
        Tuple[pd.DataFrame, scaler]: Scaled DataFrame and fitted scaler object
    """
    df_scaled = df.copy()
    
    # Get numeric columns
    numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclude specified columns
    if exclude_cols:
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    logger.info(f"Scaling {len(numeric_cols)} numeric features using {scaler_type}")
    
    # Initialize scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Fit and transform
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    logger.info(f"✅ Features scaled successfully")
    logger.info(f"  Mean of scaled features: {df_scaled[numeric_cols].mean().mean():.4f}")
    logger.info(f"  Std of scaled features: {df_scaled[numeric_cols].std().mean():.4f}")
    
    return df_scaled, scaler


def create_splits(df: pd.DataFrame,
                 target_col: str,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42) -> Tuple:
    """
    Create stratified train/validation/test splits.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion for test set (default 0.2)
        val_size: Proportion for validation set (default 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Creating train/val/test splits...")
    logger.info(f"  Test size: {test_size:.1%}")
    logger.info(f"  Val size: {val_size:.1%}")
    logger.info(f"  Train size: {1 - test_size - val_size:.1%}")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Log split sizes
    logger.info(f"\n{'='*50}")
    logger.info(f"SPLIT SIZES:")
    logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(df):.1%})")
    logger.info(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(df):.1%})")
    logger.info(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(df):.1%})")
    logger.info(f"  Total: {len(df):,} samples")
    
    # Log class distribution
    logger.info(f"\nCLASS DISTRIBUTION:")
    logger.info(f"  Train: {y_train.value_counts().to_dict()}")
    logger.info(f"  Val:   {y_val.value_counts().to_dict()}")
    logger.info(f"  Test:  {y_test.value_counts().to_dict()}")
    logger.info(f"{'='*50}\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train, X_val, X_test, y_train, y_val, y_test,
               disease_name: str,
               output_dir: Optional[Path] = None):
    """
    Save train/val/test splits to CSV files.
    
    Args:
        X_train, X_val, X_test: Feature DataFrames
        y_train, y_val, y_test: Target Series
        disease_name: Name of disease (e.g., 'heart', 'diabetes')
        output_dir: Directory to save files (default: data/processed/)
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV
    train_path = output_dir / f"{disease_name}_train.csv"
    val_path = output_dir / f"{disease_name}_val.csv"
    test_path = output_dir / f"{disease_name}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"✅ Saved splits for {disease_name}:")
    logger.info(f"  {train_path}")
    logger.info(f"  {val_path}")
    logger.info(f"  {test_path}")


def save_scaler(scaler, disease_name: str, output_dir: Optional[Path] = None):
    """
    Save fitted scaler object for inference.
    
    Args:
        scaler: Fitted scaler object
        disease_name: Name of disease
        output_dir: Directory to save (default: models/scalers/)
    """
    if output_dir is None:
        output_dir = SCALER_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scaler_path = output_dir / f"{disease_name}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"✅ Saved scaler: {scaler_path}")


if __name__ == "__main__":
    """
    Test the preprocessing functions
    """
    print("\n" + "="*70)
    print("TESTING PREPROCESSING MODULE")
    print("="*70 + "\n")
    
    print("Functions available:")
    print("  - handle_missing_values(df)")
    print("  - handle_outliers(df)")
    print("  - scale_features(df)")
    print("  - create_splits(df, target_col)")
    print("  - save_splits(...)")
    print("  - save_scaler(...)")
    
    print("\nReady to use! Import with:")
    print("  from src.data.preprocessing import *")
