from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning
    Args:
        X_train: Training features
        y_train: Training labels
    Returns:
        Trained model
    """
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    # Initialize the base model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Fit the model
    print("Training model...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def save_model(model, scaler, model_path='models/rf_model.pkl', scaler_path='models/scaler.pkl'):
    """
    Save the trained model and scaler
    """
    try:
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print("Model and scaler saved successfully")
    except Exception as e:
        print(f"Error saving model: {e}")
