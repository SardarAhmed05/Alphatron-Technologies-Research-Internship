"""
Class-based training and serialization of baseline candidate models.
"""

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class ModelTrainer:
    """
    Fits baseline Logistic Regression, Random Forest, and XGBoost models
    and serializes them to the Models/ folder.
    """

    def __init__(self, data_dir: str = "TransformedData", models_dir: str = "Models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.X_train = None
        self.y_train = None
        self.models = {}

    def load_data(self):
        """Loads training data from disk."""
        self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train.csv")).values.ravel()

    def train_logistic_regression(self) -> LogisticRegression:
        """Trains Logistic Regression baseline."""
        print("Training Logistic Regression Baseline...")
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(self.X_train, self.y_train)
        self.models["baseline_logistic_regression.pkl"] = lr
        return lr

    def train_random_forest(self) -> RandomForestClassifier:
        """Trains Random Forest baseline."""
        print("Training Random Forest Baseline...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models["baseline_random_forest.pkl"] = rf
        return rf

    def train_xgboost(self) -> XGBClassifier:
        """Trains XGBoost baseline."""
        print("Training XGBoost Baseline...")
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="mlogloss", n_jobs=1)
        xgb.fit(self.X_train, self.y_train)
        self.models["baseline_xgboost.pkl"] = xgb
        return xgb

    def save_models(self):
        """Serializes all trained models to disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        for fname, model in self.models.items():
            joblib.dump(model, os.path.join(self.models_dir, fname))
        print(f"Model Training Complete! Baseline models saved in: {self.models_dir}/")

    def run(self):
        """Executes model training workflow."""
        print("\n" + "="*50)
        print(" [STEP 6] BASELINE MODEL TRAINING")
        print("="*50)
        self.load_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.save_models()
        return self.models


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
