import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path='data/data.csv'):
    """
    Load and prepare the bankruptcy dataset
    Args:
        file_path: Path to the CSV file
    Returns:
        DataFrame containing the dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the data including scaling and handling class imbalance
    Args:
        df: Input DataFrame
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop('Bankrupt?', axis=1)
    y = df['Bankrupt?']
    
    # Split data first to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to handle class imbalance (only on training data)
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler

def plot_boxplots(df):
    for col in df.columns:
        if col != 'Bankrupt?':
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()
