"""
Simple test to verify the Flask app can start and respond to requests
"""

import sys
import os
import time
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_app():
    """Test the Flask application using Flask test client"""
    print("=" * 80)
    print("Testing Flask Web Application")
    print("=" * 80)
    
    # Import and configure app for testing
    from app import app, initialize_model
    
    print("\n1. Initializing model...")
    initialize_model()
    
    # Configure app for testing
    app.config['TESTING'] = True
    client = app.test_client()
    
    try:
        print("\n2. Testing health endpoint...")
        response = client.get('/health')
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.get_json()}")
        
        if response.status_code == 200:
            print("   ✓ Health check passed!")
        else:
            print("   ✗ Health check failed!")
            return False
        
        print("\n3. Testing home page...")
        response = client.get('/')
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200 and b'Multi-Disease' in response.data:
            print("   ✓ Home page loaded successfully!")
        else:
            print("   ✗ Home page failed to load!")
            return False
        
        print("\n4. Testing prediction endpoint...")
        test_data = {
            'age': 55,
            'gender': 1,
            'bmi': 30.5,
            'blood_pressure_systolic': 150,
            'blood_pressure_diastolic': 95,
            'cholesterol': 250,
            'glucose': 130,
            'smoking': 1,
            'alcohol': 1,
            'physical_activity': 0,
            'family_history': 1,
            'stress_level': 8
        }
        
        response = client.post('/predict', json=test_data)
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.get_json()
            print(f"   ✓ Prediction successful!")
            print(f"   Risk Level: {result.get('prediction')}")
            print(f"   Risk Score: {result.get('risk_score'):.2f}%")
            print(f"   Confidence: {result.get('confidence'):.2f}%")
            
            # Check explainability data
            if 'explanation' in result:
                print(f"   ✓ Explainability data included!")
                explanation = result['explanation']
                print(f"   - SHAP features: {len(explanation.get('shap_features', []))} features")
                print(f"   - LIME features: {len(explanation.get('lime_features', []))} features")
                print(f"   - SHAP plot: {'Present' if explanation.get('shap_plot') else 'Missing'}")
                print(f"   - LIME plot: {'Present' if explanation.get('lime_plot') else 'Missing'}")
            else:
                print("   ✗ Explainability data missing!")
                return False
        else:
            print("   ✗ Prediction failed!")
            print(f"   Error: {response.get_json()}")
            return False
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_app()
    sys.exit(0 if success else 1)
