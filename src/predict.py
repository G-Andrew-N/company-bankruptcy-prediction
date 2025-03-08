import joblib
import pandas as pd
import numpy as np

def load_model(model_path='../models/rf_model.pkl', scaler_path='../models/scaler.pkl'):
    """
    Load the trained model and scaler
    Returns:
        model, scaler
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully!")
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_bankruptcy(model, scaler, data):
    """
    Make bankruptcy predictions on new data
    Args:
        model: Trained model
        scaler: Fitted scaler
        data: DataFrame with features
    Returns:
        predictions, probabilities
    """
    try:
        # Scale the input data
        scaled_data = scaler.transform(data)
        
        # Make predictions
        predictions = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)[:, 1]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Prediction': predictions,
            'Probability': probabilities
        })
        
        return results
    
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None
