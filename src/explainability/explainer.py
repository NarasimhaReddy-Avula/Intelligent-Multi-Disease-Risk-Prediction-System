"""
Explainability Module for Multi-Disease Risk Prediction
Implements SHAP and LIME for model interpretability
"""

import numpy as np
import shap
from lime import lime_tabular
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64


class ModelExplainer:
    """
    Provides explainability for disease risk predictions using SHAP and LIME
    """
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize the explainer
        
        Args:
            model: Trained model
            X_train: Training data for background distribution
            feature_names: List of feature names
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        
        if feature_names is None and hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # Initialize SHAP explainer
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
        
        # Use TreeExplainer for tree-based models, otherwise use KernelExplainer
        try:
            # Try to use TreeExplainer (faster for tree-based models)
            self.shap_explainer = shap.TreeExplainer(model)
        except:
            # Fall back to KernelExplainer for other models
            self.shap_explainer = shap.KernelExplainer(
                model.predict_proba, 
                shap.sample(X_train_array, 100)
            )
        
        # Initialize LIME explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train_array,
            feature_names=self.feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification'
        )
    
    def explain_shap(self, X_instance):
        """
        Generate SHAP explanation for a single instance
        
        Args:
            X_instance: Single instance to explain
            
        Returns:
            shap_values: SHAP values for the instance
        """
        if hasattr(X_instance, 'values'):
            X_instance = X_instance.values
        
        if len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.shap_explainer.shap_values(X_instance)
        
        return shap_values
    
    def explain_lime(self, X_instance, num_features=10):
        """
        Generate LIME explanation for a single instance
        
        Args:
            X_instance: Single instance to explain
            num_features: Number of top features to show
            
        Returns:
            explanation: LIME explanation object
        """
        if hasattr(X_instance, 'values'):
            X_instance = X_instance.values
        
        if len(X_instance.shape) > 1:
            X_instance = X_instance.flatten()
        
        explanation = self.lime_explainer.explain_instance(
            X_instance,
            self.model.predict_proba,
            num_features=num_features
        )
        
        return explanation
    
    def plot_shap_waterfall(self, X_instance, max_display=10):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X_instance: Single instance to explain
            max_display: Maximum number of features to display
            
        Returns:
            plot_base64: Base64 encoded plot image
        """
        shap_values = self.explain_shap(X_instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification, use positive class
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        if hasattr(X_instance, 'values'):
            X_instance = X_instance.values
        
        if len(X_instance.shape) > 1:
            X_instance = X_instance.flatten()
        
        # Create waterfall plot
        plt.figure(figsize=(10, 6))
        
        # Sort features by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[::-1][:max_display]
        
        # Create the waterfall data
        features = [self.feature_names[i] if self.feature_names else f"Feature {i}" for i in indices]
        values = shap_values[indices]
        
        # Plot
        colors = ['red' if v > 0 else 'blue' for v in values]
        plt.barh(features, values, color=colors)
        plt.xlabel('SHAP Value (Impact on Prediction)')
        plt.title('Feature Impact on Disease Risk Prediction')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return plot_base64
    
    def plot_lime_explanation(self, X_instance, num_features=10):
        """
        Create LIME explanation plot
        
        Args:
            X_instance: Single instance to explain
            num_features: Number of features to show
            
        Returns:
            plot_base64: Base64 encoded plot image
        """
        explanation = self.explain_lime(X_instance, num_features)
        
        # Get feature weights
        exp_list = explanation.as_list()
        
        features = [item[0] for item in exp_list]
        weights = [item[1] for item in exp_list]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        colors = ['red' if w > 0 else 'blue' for w in weights]
        plt.barh(features, weights, color=colors)
        plt.xlabel('Feature Weight')
        plt.title('LIME: Feature Contributions to Prediction')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return plot_base64
    
    def get_top_features(self, X_instance, method='shap', top_n=5):
        """
        Get top N features affecting the prediction
        
        Args:
            X_instance: Single instance to explain
            method: 'shap' or 'lime'
            top_n: Number of top features to return
            
        Returns:
            top_features: List of (feature_name, impact_score) tuples
        """
        if method == 'shap':
            shap_values = self.explain_shap(X_instance)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Ensure shap_values is a 1D array
            shap_values = np.asarray(shap_values).flatten()
            
            # Get top features by absolute SHAP value
            indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
            
            top_features = []
            for i in range(len(indices)):
                idx = indices[i]
                feature_name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
                impact = shap_values[idx]
                top_features.append((feature_name, float(impact)))
            
            return top_features
        
        elif method == 'lime':
            explanation = self.explain_lime(X_instance, num_features=top_n)
            exp_list = explanation.as_list()
            
            # Sort by absolute weight
            exp_list = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)[:top_n]
            
            return [(item[0], item[1]) for item in exp_list]
        
        else:
            raise ValueError("Method must be 'shap' or 'lime'")
    
    def generate_explanation_report(self, X_instance, prediction, probability):
        """
        Generate a comprehensive explanation report
        
        Args:
            X_instance: Single instance to explain
            prediction: Model prediction
            probability: Prediction probability
            
        Returns:
            report: Dictionary containing explanation information
        """
        # Get SHAP explanation
        shap_features = self.get_top_features(X_instance, method='shap', top_n=5)
        shap_plot = self.plot_shap_waterfall(X_instance, max_display=10)
        
        # Get LIME explanation
        lime_features = self.get_top_features(X_instance, method='lime', top_n=5)
        lime_plot = self.plot_lime_explanation(X_instance, num_features=10)
        
        report = {
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'probability': float(probability),
            'shap_features': shap_features,
            'lime_features': lime_features,
            'shap_plot': shap_plot,
            'lime_plot': lime_plot
        }
        
        return report
