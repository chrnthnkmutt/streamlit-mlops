import joblib
import pandas as pd
import numpy as np
import os

def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        model = joblib.load("./models/RF_model_20250531_161059.pkl")
        scaler = joblib.load("./models/scaler_20250531_161059.pkl")
        feature_names = joblib.load("./models/feature_names_20250531_161059.pkl")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        # If specific timestamp files don't exist, try to find the latest files
        model_files = [f for f in os.listdir("./models") if f.startswith("RF_model_") and f.endswith(".pkl")]
        if model_files:
            latest_model = sorted(model_files)[-1]
            timestamp = latest_model.replace("RF_model_", "").replace(".pkl", "")
            
            model = joblib.load(f"./models/RF_model_{timestamp}.pkl")
            scaler = joblib.load(f"./models/scaler_{timestamp}.pkl")
            feature_names = joblib.load(f"./models/feature_names_{timestamp}.pkl")
            return model, scaler, feature_names
        else:
            raise FileNotFoundError("No model files found in ./models directory")

def create_engineered_features(input_data):
    """Create the same engineered features used in training"""
    data = input_data.copy()
    
    # Age groups
    age = data['age']
    if age <= 25:
        data['age_group_adult'] = 0
        data['age_group_middle'] = 0
        data['age_group_senior'] = 0
        data['age_group_elderly'] = 0
    elif age <= 35:
        data['age_group_adult'] = 1
        data['age_group_middle'] = 0
        data['age_group_senior'] = 0
        data['age_group_elderly'] = 0
    elif age <= 45:
        data['age_group_adult'] = 0
        data['age_group_middle'] = 1
        data['age_group_senior'] = 0
        data['age_group_elderly'] = 0
    elif age <= 55:
        data['age_group_adult'] = 0
        data['age_group_middle'] = 0
        data['age_group_senior'] = 1
        data['age_group_elderly'] = 0
    else:
        data['age_group_adult'] = 0
        data['age_group_middle'] = 0
        data['age_group_senior'] = 0
        data['age_group_elderly'] = 1
    
    # Hours categories
    hours = data['hours.per.week']
    if hours <= 20:
        data['hours_category_full_time'] = 0
        data['hours_category_overtime'] = 0
        data['hours_category_workaholic'] = 0
    elif hours <= 40:
        data['hours_category_full_time'] = 1
        data['hours_category_overtime'] = 0
        data['hours_category_workaholic'] = 0
    elif hours <= 60:
        data['hours_category_full_time'] = 0
        data['hours_category_overtime'] = 1
        data['hours_category_workaholic'] = 0
    else:
        data['hours_category_full_time'] = 0
        data['hours_category_overtime'] = 0
        data['hours_category_workaholic'] = 1
    
    # Capital features
    data['capital_net'] = data['capital.gain'] - data['capital.loss']
    data['has_capital_gain'] = 1 if data['capital.gain'] > 0 else 0
    data['has_capital_loss'] = 1 if data['capital.loss'] > 0 else 0
    
    return data

def make_prediction(model, scaler, feature_names, input_data):
    """Make prediction based on input data with feature engineering"""
    # Add engineered features
    enriched_data = create_engineered_features(input_data)
    
    # Define numerical columns (including new ones)
    numeric_cols = ['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'capital_net']
    categorical_cols = [col for col in feature_names if col not in numeric_cols]

    # Numeric DataFrame
    num_df = pd.DataFrame([[enriched_data[col] for col in numeric_cols]], columns=numeric_cols)
    num_scaled = scaler.transform(num_df)

    # Create dummy categorical DataFrame
    cat_df = pd.DataFrame([enriched_data])
    cat_df = pd.get_dummies(cat_df)

    # Add missing columns
    for col in categorical_cols:
        if col not in cat_df.columns:
            cat_df[col] = 0
    cat_df = cat_df[categorical_cols]  # Reorder

    # Combine
    final_input = np.hstack([cat_df.values, num_scaled])

    # Predict
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0]
    
    result = "<=50K" if pred == 1 else ">50K"
    confidence = max(prob)
    
    return result, confidence

# Keep the old functions for backward compatibility
def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def predict_income(model, scaler, input_data):
    return make_prediction(model, scaler, None, input_data)