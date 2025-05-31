import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import xgboost as xgb

def load_data():
    df = pd.read_csv("data/adult.csv")
    df.drop(columns=['fnlwgt'], inplace=True)
    
    # Better data cleaning
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].str.strip()
        # Handle missing values represented as '?'
        df[col] = df[col].replace('?', df[col].mode()[0])
    
    df['income'] = df['income'].map({'<=50K': 1, '>50K': 0})
    df = pd.get_dummies(df, drop_first=True)  # Avoid multicollinearity
    return df

def create_features(df):
    """Create additional features for better prediction"""
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                            labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    # Hours per week categories
    df['hours_category'] = pd.cut(df['hours.per.week'], 
                                 bins=[0, 20, 40, 60, 100],
                                 labels=['part_time', 'full_time', 'overtime', 'workaholic'])
    
    # Capital net (gain - loss)
    df['capital_net'] = df['capital.gain'] - df['capital.loss']
    
    # Has capital gains/losses
    df['has_capital_gain'] = (df['capital.gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital.loss'] > 0).astype(int)
    
    # Convert new categorical features to dummies
    df = pd.get_dummies(df, columns=['age_group', 'hours_category'], drop_first=True)
    
    return df

def train_multiple_models():
    """Train and compare multiple models"""
    df = load_data()
    df = create_features(df)
    
    X = df.drop('income', axis=1)
    y = df['income']
    feature_names = X.columns.tolist()
    
    # Scale numerical features
    num_cols = ['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'capital_net']
    scaler = StandardScaler()  # Often better than MinMaxScaler
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    print("Model Comparison:")
    print("=" * 80)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print("-" * 50)
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate accuracy scores
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        
        # Calculate additional metrics for test set
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Print all metrics
        print(f"  Train Accuracy:     {train_score:.4f}")
        print(f"  Test Accuracy:      {test_score:.4f}")
        print(f"  Overfitting:        {train_score - test_score:.4f}")
        print(f"  Precision:          {precision:.4f}")
        print(f"  Recall:             {recall:.4f}")
        print(f"  F1-Score:           {f1:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    True Neg (<=50K):  {tn}")
        print(f"    False Pos:         {fp}")
        print(f"    False Neg:         {fn}")
        print(f"    True Pos (>50K):   {tp}")
        
        # Class-specific metrics
        print(f"  Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['<=50K', '>50K'], digits=4))
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print("=" * 80)
    print(f"Best Model: {best_name} with Test Accuracy: {best_score:.4f}")
    
    # Save the best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/RF_model_{timestamp}.pkl"
    scaler_path = f"models/scaler_{timestamp}.pkl"
    feature_path = f"models/feature_names_{timestamp}.pkl"
    metrics_path = f"models/metrics_{timestamp}.pkl"
    
    # Calculate final metrics for the best model
    y_test_pred_best = best_model.predict(X_test)
    final_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_best),
        'classification_report': classification_report(y_test, y_test_pred_best, 
                                                     target_names=['<=50K', '>50K'], 
                                                     output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred_best).tolist(),
        'model_name': best_name
    }
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_path)
    joblib.dump(final_metrics, metrics_path)
    
    print(f"Best model ({best_name}) saved to {model_path}")
    return model_path, scaler_path, feature_names

def train_optimized_model():
    """Train a hyperparameter-optimized model"""
    df = load_data()
    df = create_features(df)
    
    X = df.drop('income', axis=1)
    y = df['income']
    feature_names = X.columns.tolist()
    
    num_cols = ['age', 'capital.gain', 'capital.loss', 'hours.per.week', 'capital_net']
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    
    print(f"Optimized Random Forest Accuracy: {test_score:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Save the optimized model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/RF_model_{timestamp}.pkl"
    scaler_path = f"models/scaler_{timestamp}.pkl"
    feature_path = f"models/feature_names_{timestamp}.pkl"
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_path)
    
    print(f"Optimized model saved to {model_path}")
    return model_path, scaler_path, feature_names

if __name__ == "__main__":
    print("Training multiple models...")
    train_multiple_models()
    print("\nTraining optimized model...")
    train_optimized_model()