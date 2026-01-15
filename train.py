"""
Training script for the Multi-Disease Risk Prediction System
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictor import MultiDiseasePredictor
from src.utils.data_utils import generate_sample_data, preprocess_data, split_data, save_data


def train_model(model_type='ensemble', save_path='models/disease_predictor.pkl'):
    """
    Train the disease risk prediction model
    
    Args:
        model_type: Type of model to train ('random_forest', 'xgboost', 'neural_network', 'ensemble')
        save_path: Path to save the trained model
    """
    print("=" * 60)
    print("Multi-Disease Risk Prediction System - Training")
    print("=" * 60)
    
    # Generate sample data
    print("\n1. Generating sample health data...")
    df = generate_sample_data(n_samples=2000)
    print(f"Generated {len(df)} samples")
    print(f"Class distribution:\n{df['disease_risk'].value_counts()}")
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    save_data(df, 'data/sample_health_data.csv')
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, y = preprocess_data(df)
    print(f"Features: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data
    print("\n3. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print(f"\n4. Training {model_type} model...")
    predictor = MultiDiseasePredictor(model_type=model_type)
    predictor.train(X_train, y_train)
    
    # Evaluate model
    print("\n5. Evaluating model on test set...")
    scores = predictor.evaluate(X_test, y_test)
    
    # Display feature importance
    print("\n6. Feature importance:")
    importance_dict = predictor.get_feature_importance()
    for model_name, importance in importance_dict.items():
        print(f"\n{model_name}:")
        feature_importance = sorted(
            zip(X.columns, importance), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        for feature, imp in feature_importance[:5]:
            print(f"  {feature}: {imp:.4f}")
    
    # Save model
    print(f"\n7. Saving model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    predictor.save(save_path)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    
    return predictor


if __name__ == "__main__":
    # Train ensemble model
    train_model(model_type='ensemble', save_path='models/disease_predictor.pkl')
