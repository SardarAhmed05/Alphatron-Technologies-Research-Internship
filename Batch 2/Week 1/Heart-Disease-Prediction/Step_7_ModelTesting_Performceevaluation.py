"""
Class-based diagnostic evaluation of baseline models on unseen test data.
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


class ModelEvaluator:
    """
    Loads trained baseline models and evaluates performance metrics
    (Accuracy, Weighted F1, Classification Report) on unseen test data.
    """

    def __init__(self, data_dir: str = "TransformedData", models_dir: str = "Models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.X_test = None
        self.y_test = None

    def load_test_data(self):
        """Loads held-out test data."""
        self.X_test = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
        self.y_test = pd.read_csv(os.path.join(self.data_dir, "y_test.csv")).values.ravel()

    def evaluate_model(self, model_name: str, filename: str) -> dict:
        """Evaluates an individual serialized model."""
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            return {}
        model = joblib.load(path)
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1_w = f1_score(self.y_test, y_pred, average="weighted")
        
        print(f"\n--- {model_name} Results ---")
        print(f"Accuracy: {acc*100:.2f}% | Weighted F1: {f1_w:.4f}")
        print(classification_report(self.y_test, y_pred, zero_division=0))
        return {"Accuracy": acc, "Weighted_F1": f1_w}

    def run(self) -> dict:
        """Evaluates all candidate baseline models."""
        print("\n" + "="*50)
        print(" [STEP 7] BASELINE MODEL EVALUATION")
        print("="*50)
        self.load_test_data()
        
        candidates = {
            "Logistic Regression": "baseline_logistic_regression.pkl",
            "Random Forest": "baseline_random_forest.pkl",
            "XGBoost": "baseline_xgboost.pkl"
        }
        
        results = {}
        for name, fname in candidates.items():
            results[name] = self.evaluate_model(name, fname)
        return results


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.run()
