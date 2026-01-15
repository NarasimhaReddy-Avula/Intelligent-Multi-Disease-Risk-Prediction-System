"""
Data utilities for the Multi-Disease Risk Prediction System
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_sample_data(n_samples=1000, random_state=42):
    """
    Generate sample health data for demonstration
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame: Generated health data
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
        'bmi': np.random.uniform(18.5, 40, n_samples),
        'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
        'cholesterol': np.random.uniform(150, 300, n_samples),
        'glucose': np.random.uniform(70, 200, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'alcohol': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'physical_activity': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3]),  # 0: Low, 1: Medium, 2: High
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'stress_level': np.random.randint(1, 11, n_samples),  # 1-10 scale
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable (disease risk) based on features
    # Higher risk with: older age, high BMI, high BP, high cholesterol, smoking, etc.
    risk_score = (
        (df['age'] > 50).astype(int) * 0.2 +
        (df['bmi'] > 30).astype(int) * 0.15 +
        (df['blood_pressure_systolic'] > 140).astype(int) * 0.15 +
        (df['cholesterol'] > 240).astype(int) * 0.1 +
        (df['glucose'] > 126).astype(int) * 0.15 +
        df['smoking'] * 0.1 +
        df['family_history'] * 0.1 +
        (df['stress_level'] > 7).astype(int) * 0.05
    )
    
    # Add some randomness
    risk_score += np.random.uniform(-0.1, 0.1, n_samples)
    
    # Convert to binary classification (0: Low Risk, 1: High Risk)
    df['disease_risk'] = (risk_score > 0.5).astype(int)
    
    return df


def preprocess_data(df, target_column='disease_risk'):
    """
    Preprocess the data for model training
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def load_data(filepath):
    """
    Load data from a CSV file
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame: Loaded data
    """
    return pd.read_csv(filepath)


def save_data(df, filepath):
    """
    Save data to a CSV file
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file
    """
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def get_feature_descriptions():
    """
    Get descriptions of all features
    
    Returns:
        dict: Dictionary of feature descriptions
    """
    descriptions = {
        'age': 'Age in years (20-80)',
        'gender': 'Gender (0: Female, 1: Male)',
        'bmi': 'Body Mass Index (18.5-40)',
        'blood_pressure_systolic': 'Systolic Blood Pressure (mmHg)',
        'blood_pressure_diastolic': 'Diastolic Blood Pressure (mmHg)',
        'cholesterol': 'Total Cholesterol (mg/dL)',
        'glucose': 'Blood Glucose Level (mg/dL)',
        'smoking': 'Smoking Status (0: No, 1: Yes)',
        'alcohol': 'Alcohol Consumption (0: No, 1: Yes)',
        'physical_activity': 'Physical Activity Level (0: Low, 1: Medium, 2: High)',
        'family_history': 'Family History of Disease (0: No, 1: Yes)',
        'stress_level': 'Stress Level (1-10 scale)',
    }
    return descriptions


def validate_input(input_data):
    """
    Validate user input data
    
    Args:
        input_data: Dictionary of input features
        
    Returns:
        bool: True if valid, raises ValueError if invalid
    """
    # Define valid ranges
    ranges = {
        'age': (20, 80),
        'gender': (0, 1),
        'bmi': (15, 50),
        'blood_pressure_systolic': (80, 200),
        'blood_pressure_diastolic': (50, 130),
        'cholesterol': (100, 400),
        'glucose': (50, 300),
        'smoking': (0, 1),
        'alcohol': (0, 1),
        'physical_activity': (0, 2),
        'family_history': (0, 1),
        'stress_level': (1, 10),
    }
    
    for feature, value in input_data.items():
        if feature in ranges:
            min_val, max_val = ranges[feature]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature} must be between {min_val} and {max_val}")
    
    return True
