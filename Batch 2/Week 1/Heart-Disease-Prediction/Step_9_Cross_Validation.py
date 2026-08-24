"""
Executes 5-Fold Stratified Cross-Validation across candidate algorithms to verify generalization.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def run_cross_validation(data_dir="TransformedData"):
    print("\n" + "="*50)
    print(" [STEP 9] 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*50)
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_sampled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_sampled.csv")).values.ravel()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss", n_jobs=1)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        print(f"{name:22} -> 5-Fold CV Mean Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

if __name__ == "__main__":
    run_cross_validation()
