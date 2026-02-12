"""
Dataset Loading Module

This module provides functions to load all disease datasets
for the Multi-Disease Risk Prediction System.

Author: BTP Project 2025-2026
Last Updated: February 12, 2026
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / 'datasets'


def load_heart_dataset() -> pd.DataFrame:
    """
    Load the Cardiovascular Disease dataset.
    
    Returns:
        pd.DataFrame: Heart disease dataset with 70,000 samples
    """
    try:
        filepath = DATASETS_DIR / 'heart.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded heart dataset: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading heart dataset: {e}")
        raise


def load_diabetes_dataset() -> pd.DataFrame:
    """
    Load the Diabetes dataset.
    
    Returns:
        pd.DataFrame: Diabetes dataset with 100,000 samples
    """
    try:
        filepath = DATASETS_DIR / 'diabetes.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded diabetes dataset: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading diabetes dataset: {e}")
        raise


def load_kidney_dataset() -> pd.DataFrame:
    """
    Load the Chronic Kidney Disease dataset.
    
    Returns:
        pd.DataFrame: Kidney disease dataset with 1,659 samples
    """
    try:
        filepath = DATASETS_DIR / 'Chronic_Kidney_Dsease_data.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded kidney dataset: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading kidney dataset: {e}")
        raise


def load_liver_dataset() -> pd.DataFrame:
    """
    Load the Liver Disease dataset.
    
    Returns:
        pd.DataFrame: Liver disease dataset with 30,691 samples
    """
    try:
        filepath = DATASETS_DIR / 'liver.csv'
        df = pd.read_csv(filepath)
        logger.info(f"✅ Loaded liver dataset: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading liver dataset: {e}")
        raise


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load all disease datasets and return as a dictionary.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with disease names as keys
                                 and DataFrames as values
    
    Example:
        >>> datasets = load_all_datasets()
        >>> print(datasets['heart'].shape)
        (70000, X)
    """
    logger.info("Loading all datasets...")
    
    datasets = {
        'heart': load_heart_dataset(),
        'diabetes': load_diabetes_dataset(),
        'kidney': load_kidney_dataset(),
        'liver': load_liver_dataset()
    }
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DATASET SUMMARY")
    logger.info("="*60)
    
    total_samples = 0
    for name, df in datasets.items():
        missing = df.isnull().sum().sum()
        total_samples += len(df)
        logger.info(f"{name.upper():12} | Shape: {str(df.shape):15} | Missing: {missing:6}")
    
    logger.info("="*60)
    logger.info(f"TOTAL SAMPLES: {total_samples:,}")
    logger.info("="*60 + "\n")
    
    return datasets


def get_dataset_info(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Get comprehensive information about a dataset.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset
    
    Returns:
        Dict: Dictionary containing dataset statistics
    """
    info = {
        'name': dataset_name,
        'shape': df.shape,
        'n_samples': len(df),
        'n_features': len(df.columns),
        'features': df.columns.tolist(),
        'missing_values': df.isnull().sum().to_dict(),
        'total_missing': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / df.size * 100),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return info


def validate_datasets(datasets: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate that all datasets are loaded correctly.
    
    Args:
        datasets: Dictionary of DataFrames
    
    Returns:
        bool: True if all validations pass
    """
    logger.info("Validating datasets...")
    
    expected_datasets = ['heart', 'diabetes', 'kidney', 'liver']
    all_valid = True
    
    # Check all datasets are present
    for name in expected_datasets:
        if name not in datasets:
            logger.error(f"❌ Missing dataset: {name}")
            all_valid = False
        elif datasets[name] is None or datasets[name].empty:
            logger.error(f"❌ Dataset {name} is empty")
            all_valid = False
        else:
            logger.info(f"✅ Dataset {name} validated")
    
    if all_valid:
        logger.info("✅ All datasets validated successfully!")
    else:
        logger.error("❌ Some datasets failed validation")
    
    return all_valid


if __name__ == "__main__":
    """
    Test the data loading functions
    """
    print("\n" + "="*70)
    print("TESTING DATA LOADING MODULE")
    print("="*70 + "\n")
    
    # Load all datasets
    datasets = load_all_datasets()
    
    # Validate
    valid = validate_datasets(datasets)
    
    if valid:
        print("\n✅ All datasets loaded successfully!")
        print("\nTo use in your code:")
        print("  from src.data.load_datasets import load_all_datasets")
        print("  datasets = load_all_datasets()")
        print("  heart_df = datasets['heart']")
    else:
        print("\n❌ Some datasets failed to load")
