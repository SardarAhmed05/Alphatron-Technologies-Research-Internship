"""
Trains baseline classification models (Logistic Regression, Random Forest, XGBoost).
"""

import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def run_model_training(data_dir="TransformedData", models_dir="Models", metrics_dir="ModelTraining"):
    print("\n" + "="*50)
    print(" [STEP 6] BASELINE MODEL TRAINING")
    print("="*50)
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    # 1. Logistic Regression Baseline
    print("Training Logistic Regression Baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(models_dir, "baseline_logistic_regression.pkl"))
    
    # 2. Random Forest Baseline
    print("Training Random Forest Baseline...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(models_dir, "baseline_random_forest.pkl"))
    
    # 3. XGBoost Baseline
    print("Training XGBoost Baseline...")
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss", n_jobs=1)
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, os.path.join(models_dir, "baseline_xgboost.pkl"))
    
    print(f"Model Training Complete! Baseline models saved in: {models_dir}/")

if __name__ == "__main__":
    run_model_training()
