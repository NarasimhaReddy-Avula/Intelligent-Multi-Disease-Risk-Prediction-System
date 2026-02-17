"""
Flask Web Application for Multi-Disease Risk Prediction with Explainability
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.predictor import MultiDiseasePredictor
from src.explainability.explainer import ModelExplainer
from src.utils.data_utils import validate_input, get_feature_descriptions, generate_sample_data, preprocess_data

app = Flask(__name__)

# Global variables for model and explainer
model = None
explainer = None
X_train = None


def initialize_model():
    """Initialize or load the trained model"""
    global model, explainer, X_train
    
    model_path = 'models/disease_predictor.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        model = MultiDiseasePredictor.load(model_path)
        print("Model loaded successfully!")
    else:
        # Train a new model
        print("No existing model found. Training new model...")
        from train import train_model
        model = train_model(model_type='ensemble', save_path=model_path)
    
    # Generate training data for explainer
    df = generate_sample_data(n_samples=1000)
    X_train, _ = preprocess_data(df)
    
    # Create a wrapper class that handles scaling internally
    class ScaledModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            return self.model.predict_proba(X_scaled)
    
    # Get the first model from ensemble and wrap it
    primary_model = list(model.models.values())[0]
    wrapped_model = ScaledModelWrapper(primary_model, model.scaler)
    
    # Initialize explainer with unscaled data
    explainer = ModelExplainer(wrapped_model, X_train.values, model.feature_names)
    
    print("Model and explainer initialized!")


@app.route('/')
def index():
    """Render the home page"""
    feature_descriptions = get_feature_descriptions()
    return render_template('index.html', features=feature_descriptions)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get input data from request
        data = request.get_json()
        
        # Validate input
        validate_input(data)
        
        # Prepare features in correct order
        feature_order = [
            'age', 'gender', 'bmi', 'blood_pressure_systolic', 
            'blood_pressure_diastolic', 'cholesterol', 'glucose', 
            'smoking', 'alcohol', 'physical_activity', 
            'family_history', 'stress_level'
        ]
        
        features = [float(data.get(f)) for f in feature_order if f in data]
        
        # Validate all required features are present
        if len(features) != len(feature_order):
            missing_features = [f for f in feature_order if f not in data]
            return jsonify({
                'success': False,
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
        
        X_input = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]
        
        # Get explanation
        X_input_df = pd.DataFrame([features], columns=feature_order)
        explanation_report = explainer.generate_explanation_report(
            X_input_df, 
            prediction, 
            probabilities[1] if len(probabilities) > 1 else probabilities[0]
        )
        
        # Prepare response
        response = {
            'success': True,
            'prediction': 'High Risk' if prediction == 1 else 'Low Risk',
            'risk_score': float(probabilities[1] if len(probabilities) > 1 else probabilities[0]) * 100,
            'confidence': float(max(probabilities)) * 100,
            'explanation': explanation_report
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("Error in prediction:")
        print(error_trace)
        # Don't expose detailed traces in production
        return jsonify({
            'success': False,
            'error': 'An error occurred during prediction. Please check your input values.'
        }), 400


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    # Initialize model before starting the app
    initialize_model()
    
    # Run the Flask app (use debug=False and localhost in production)
    # For production, use a WSGI server like Gunicorn
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='127.0.0.1', port=5000)
