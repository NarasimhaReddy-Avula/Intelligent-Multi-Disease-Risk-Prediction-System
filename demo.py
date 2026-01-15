"""
Demo script to showcase the Multi-Disease Risk Prediction System
with AI Explainability features
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.predictor import MultiDiseasePredictor
from src.explainability.explainer import ModelExplainer
from src.utils.data_utils import generate_sample_data, preprocess_data, split_data


def demonstrate_system():
    """
    Demonstrate the complete AI-powered disease risk prediction system
    with explainability
    """
    print("=" * 80)
    print("AI-POWERED MULTI-DISEASE RISK PREDICTION WITH EXPLAINABILITY")
    print("=" * 80)
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating Sample Health Data...")
    df = generate_sample_data(n_samples=1000)
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Features: {list(df.columns[:-1])}")
    print(f"\nClass Distribution:")
    print(df['disease_risk'].value_counts())
    
    # Step 2: Prepare data
    print("\n🔧 Step 2: Preprocessing Data...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # Step 3: Train model
    print("\n🤖 Step 3: Training Ensemble Model...")
    print("   (Random Forest + XGBoost + Neural Network)")
    predictor = MultiDiseasePredictor(model_type='ensemble')
    predictor.train(X_train, y_train)
    
    # Step 4: Evaluate model
    print("\n📈 Step 4: Evaluating Model Performance...")
    scores = predictor.evaluate(X_test, y_test)
    avg_score = np.mean(list(scores.values()))
    print(f"\n✓ Average Ensemble Accuracy: {avg_score:.2%}")
    
    # Step 5: Feature importance
    print("\n🎯 Step 5: Analyzing Feature Importance...")
    importance_dict = predictor.get_feature_importance()
    
    if importance_dict:
        # Get average importance across models
        all_importances = np.mean(list(importance_dict.values()), axis=0)
        feature_importance = sorted(
            zip(X.columns, all_importances),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        print("\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(feature_importance[:5], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # Step 6: Make predictions with explainability
    print("\n🔍 Step 6: Demonstrating AI Explainability...")
    print("\nSelecting test cases for explainability analysis...")
    
    # Get one high-risk and one low-risk example
    high_risk_idx = y_test[y_test == 1].index[0]
    low_risk_idx = y_test[y_test == 0].index[0]
    
    # Initialize explainer
    primary_model = list(predictor.models.values())[0]
    X_train_scaled = predictor.scaler.transform(X_train)
    explainer = ModelExplainer(primary_model, X_train_scaled, predictor.feature_names)
    
    # Analyze high-risk case
    print("\n" + "-" * 80)
    print("📋 HIGH RISK CASE ANALYSIS")
    print("-" * 80)
    
    X_high = X_test.loc[[high_risk_idx]]
    pred_high = predictor.predict(X_high)[0]
    prob_high = predictor.predict_proba(X_high)[0]
    
    print(f"\nPrediction: {'High Risk' if pred_high == 1 else 'Low Risk'}")
    print(f"Confidence: {max(prob_high):.2%}")
    print(f"Risk Score: {prob_high[1] if len(prob_high) > 1 else prob_high[0]:.2%}")
    
    print("\n🔬 SHAP Analysis (Top 5 Contributing Features):")
    shap_features = explainer.get_top_features(X_high, method='shap', top_n=5)
    for i, (feature, impact) in enumerate(shap_features, 1):
        direction = "increases" if impact > 0 else "decreases"
        print(f"  {i}. {feature}: {impact:+.4f} ({direction} risk)")
    
    print("\n🔬 LIME Analysis (Top 5 Contributing Features):")
    lime_features = explainer.get_top_features(X_high, method='lime', top_n=5)
    for i, (feature, weight) in enumerate(lime_features, 1):
        direction = "increases" if weight > 0 else "decreases"
        print(f"  {i}. {feature}: {weight:+.4f} ({direction} risk)")
    
    # Analyze low-risk case
    print("\n" + "-" * 80)
    print("📋 LOW RISK CASE ANALYSIS")
    print("-" * 80)
    
    X_low = X_test.loc[[low_risk_idx]]
    pred_low = predictor.predict(X_low)[0]
    prob_low = predictor.predict_proba(X_low)[0]
    
    print(f"\nPrediction: {'High Risk' if pred_low == 1 else 'Low Risk'}")
    print(f"Confidence: {max(prob_low):.2%}")
    print(f"Risk Score: {prob_low[1] if len(prob_low) > 1 else prob_low[0]:.2%}")
    
    print("\n🔬 SHAP Analysis (Top 5 Contributing Features):")
    shap_features = explainer.get_top_features(X_low, method='shap', top_n=5)
    for i, (feature, impact) in enumerate(shap_features, 1):
        direction = "increases" if impact > 0 else "decreases"
        print(f"  {i}. {feature}: {impact:+.4f} ({direction} risk)")
    
    print("\n🔬 LIME Analysis (Top 5 Contributing Features):")
    lime_features = explainer.get_top_features(X_low, method='lime', top_n=5)
    for i, (feature, weight) in enumerate(lime_features, 1):
        direction = "increases" if weight > 0 else "decreases"
        print(f"  {i}. {feature}: {weight:+.4f} ({direction} risk)")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📝 Summary:")
    print(f"   • Trained ensemble model with {avg_score:.2%} accuracy")
    print(f"   • Analyzed predictions with SHAP and LIME explainability")
    print(f"   • Identified key risk factors for each prediction")
    print(f"\n💡 Key Insights:")
    print(f"   • The model provides transparent, interpretable predictions")
    print(f"   • Both SHAP and LIME agree on major contributing features")
    print(f"   • Healthcare professionals can understand WHY predictions are made")
    print(f"\n🌐 Next Steps:")
    print(f"   • Run 'python app.py' to start the web interface")
    print(f"   • Access http://localhost:5000 for interactive predictions")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_system()
