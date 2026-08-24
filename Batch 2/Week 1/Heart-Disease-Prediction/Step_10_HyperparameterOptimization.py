"""
Uses GridSearchCV to fine-tune XGBoost and Random Forest hyperparameters.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

def run_hyperparameter_optimization(data_dir="TransformedData", models_dir="Models"):
    print("\n" + "="*50)
    print(" [STEP 10] HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
    print("="*50)
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_sampled.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_sampled.csv")).values.ravel()
    
    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=1)
    
    param_grid = {
        "n_estimators": [100, 150],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=1, verbose=0)
    
    print("Executing GridSearchCV on XGBoost...")
    grid.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid.best_params_}")
    print(f"Best CV Accuracy: {grid.best_score_*100:.2f}%")
    
    best_xgb = grid.best_estimator_
    joblib.dump(best_xgb, os.path.join(models_dir, "tuned_xgboost.pkl"))
    print(f"Saved tuned model to: {models_dir}/tuned_xgboost.pkl")
    return best_xgb

if __name__ == "__main__":
    run_hyperparameter_optimization()
