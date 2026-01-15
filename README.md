# 🏥 Intelligent Multi-Disease Risk Prediction System

An AI-powered healthcare solution that predicts disease risk using ensemble machine learning models with comprehensive explainability features through SHAP and LIME.

## 🌟 Features

- **Multi-Model Ensemble**: Combines Random Forest, XGBoost, and Neural Networks for robust predictions
- **AI Explainability**: Implements both SHAP and LIME for transparent, interpretable predictions
- **Interactive Web Interface**: User-friendly Flask web application for easy interaction
- **Comprehensive Analysis**: Evaluates multiple health parameters including:
  - Age, Gender, BMI
  - Blood Pressure (Systolic & Diastolic)
  - Cholesterol & Glucose levels
  - Lifestyle factors (Smoking, Alcohol, Physical Activity)
  - Family History & Stress Levels
- **Visual Explanations**: Generates detailed visualizations showing feature importance and contribution

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NarasimhaReddy-Avula/Intelligent-Multi-Disease-Risk-Prediction-System.git
cd Intelligent-Multi-Disease-Risk-Prediction-System
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

Train the ensemble model with sample data:

```bash
python train.py
```

This will:
- Generate sample health data
- Train Random Forest, XGBoost, and Neural Network models
- Save the trained model to `models/disease_predictor.pkl`
- Display training and evaluation metrics

### Running the Web Application

Start the Flask web application:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

## 📊 Model Architecture

### Ensemble Learning
The system uses three complementary models:

1. **Random Forest Classifier**
   - 100 estimators
   - Max depth: 10
   - Handles non-linear relationships

2. **XGBoost Classifier**
   - Gradient boosting algorithm
   - Optimized for performance
   - Handles feature interactions

3. **Neural Network (MLP)**
   - Hidden layers: (100, 50)
   - ReLU activation
   - Captures complex patterns

### Explainability Methods

#### SHAP (SHapley Additive exPlanations)
- Based on game theory
- Provides consistent feature attributions
- Shows exact contribution of each feature
- Global and local interpretability

#### LIME (Local Interpretable Model-agnostic Explanations)
- Model-agnostic approach
- Creates local linear approximations
- Easy to understand explanations
- Works with any model type

## 🎯 Use Cases

- **Personal Health Assessment**: Individuals can assess their disease risk
- **Healthcare Screening**: Clinics can use for preliminary risk assessment
- **Research**: Study feature importance in disease prediction
- **Education**: Learn about AI explainability in healthcare

## 📁 Project Structure

```
Intelligent-Multi-Disease-Risk-Prediction-System/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py          # ML models implementation
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── explainer.py          # SHAP & LIME implementation
│   └── utils/
│       ├── __init__.py
│       └── data_utils.py         # Data handling utilities
├── templates/
│   └── index.html                # Web interface
├── static/
│   ├── css/
│   │   └── style.css             # Styling
│   └── js/
│       └── main.js               # Frontend logic
├── data/                         # Generated/stored data
├── models/                       # Trained models
├── app.py                        # Flask application
├── train.py                      # Model training script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔬 Technical Details

### Data Features

The system analyzes 12 key health parameters:

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| Age | Numeric | 20-80 | Age in years |
| Gender | Categorical | 0-1 | 0: Female, 1: Male |
| BMI | Numeric | 15-50 | Body Mass Index |
| BP Systolic | Numeric | 80-200 | Systolic blood pressure (mmHg) |
| BP Diastolic | Numeric | 50-130 | Diastolic blood pressure (mmHg) |
| Cholesterol | Numeric | 100-400 | Total cholesterol (mg/dL) |
| Glucose | Numeric | 50-300 | Blood glucose level (mg/dL) |
| Smoking | Binary | 0-1 | Smoking status |
| Alcohol | Binary | 0-1 | Alcohol consumption |
| Physical Activity | Categorical | 0-2 | Activity level (Low/Med/High) |
| Family History | Binary | 0-1 | Family disease history |
| Stress Level | Numeric | 1-10 | Self-reported stress |

### Model Performance

The ensemble model achieves:
- **Accuracy**: ~85-90% on test data
- **Explainability**: Full transparency through SHAP and LIME
- **Interpretability**: Clear feature importance rankings

## 🛠️ API Endpoints

### Health Check
```
GET /health
```
Returns the system status and model availability.

### Prediction
```
POST /predict
Content-Type: application/json

{
  "age": 45,
  "gender": 1,
  "bmi": 28.5,
  "blood_pressure_systolic": 140,
  "blood_pressure_diastolic": 90,
  "cholesterol": 240,
  "glucose": 120,
  "smoking": 1,
  "alcohol": 0,
  "physical_activity": 1,
  "family_history": 1,
  "stress_level": 7
}
```

Returns prediction with explainability:
```json
{
  "success": true,
  "prediction": "High Risk",
  "risk_score": 72.5,
  "confidence": 85.3,
  "explanation": {
    "shap_features": [...],
    "lime_features": [...],
    "shap_plot": "base64_image",
    "lime_plot": "base64_image"
  }
}
```

## ⚠️ Disclaimer

This system is designed for **educational and demonstration purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🙏 Acknowledgments

- SHAP library by Scott Lundberg
- LIME library by Marco Tulio Ribeiro
- scikit-learn, XGBoost, and Flask communities
- Healthcare AI research community