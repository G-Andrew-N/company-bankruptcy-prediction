# Company Bankruptcy Prediction

## Project Overview
This project implements a machine learning model to predict company bankruptcy using financial indicators. 

## Dataset Information
- **Source**: [Kaggle Dataset](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction)
- **Size**: 6819 companies
  - Non-bankrupt cases: 6599
  - Bankrupt cases: 220
- **Features**: 95 financial indicators
- **Target Variable**: Bankruptcy status (1: Bankrupt, 0: Non-bankrupt)


## Setup and Installation

### Prerequisites
- Python latest version
- pip (Python package installer)

### Installation
1. Clone the repository:
```bash
git clone [repository-url]
cd company-bankruptcy-prediction
```

2. Install required packages:
```bash
pip pip install pandas numpy scikit-learn imbalanced-learn joblib matplotlib seaborn
```



## Usage

### 1. Data Preprocessing
- Run `notebooks/data_preprocessing.ipynb`
- Handles data loading, cleaning, and initial exploration
- Performs feature scaling and handles class imbalance

### 2. Model Training
- Run `notebooks/model_training.ipynb`
- Trains a Random Forest Classifier
- Performs hyperparameter tuning
- Saves the trained model and scaler

### 3. Model Evaluation
- Run `notebooks/model_evaluation.ipynb`
- Evaluates model performance using various metrics
- Generates visualization of results
- Analyzes feature importance

### 4. Making Predictions
- Run `notebooks/prediction.ipynb`
- Demonstrates how to use the trained model for predictions
- Includes interpretation of results



## Model Details

### Algorithm
- **Model**: Random Forest Classifier
- **Features**: 95 financial indicators
- **Handling Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Evaluation Metrics**: 
  - ROC-AUC Score
  - Precision-Recall
  - F1-Score
  - Confusion Matrix

### Performance
- Model performance metrics will be updated after training





## Source Code Description

### src/preprocess.py
- Data loading and preprocessing functions
- Feature scaling
- SMOTE implementation for handling class imbalance

### src/train.py
- Model training implementation
- Hyperparameter tuning
- Model saving functionality

### src/utils.py
- Evaluation metrics
- Visualization functions
- Helper utilities

### src/predict.py
- Model loading
- Prediction functionality
- Results formatting

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

