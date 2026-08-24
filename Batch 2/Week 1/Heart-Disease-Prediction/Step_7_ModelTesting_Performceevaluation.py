"""
Evaluates baseline models on unseen test data and reports classification metrics.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

def run_model_evaluation(data_dir="TransformedData", models_dir="Models"):
    print("\n" + "="*50)
    print(" [STEP 7] BASELINE MODEL EVALUATION")
    print("="*50)
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    models = {
        "Logistic Regression": "baseline_logistic_regression.pkl",
        "Random Forest": "baseline_random_forest.pkl",
        "XGBoost": "baseline_xgboost.pkl"
    }
    
    results = {}
    for name, filename in models.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1_w = f1_score(y_test, y_pred, average="weighted")
            results[name] = {"Accuracy": acc, "Weighted_F1": f1_w}
            print(f"\n--- {name} Results ---")
            print(f"Accuracy: {acc*100:.2f}% | Weighted F1: {f1_w:.4f}")
            print(classification_report(y_test, y_pred, zero_division=0))
            
    return results

if __name__ == "__main__":
    run_model_evaluation()
