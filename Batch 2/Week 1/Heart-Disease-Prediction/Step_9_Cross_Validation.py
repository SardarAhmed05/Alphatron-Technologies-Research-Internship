"""
Class-based Stratified 5-Fold Cross-Validation across candidate models.
"""

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


class CrossValidator:
    """
    Conducts Stratified 5-Fold Cross-Validation on training data to assess generalization stability.
    """

    def __init__(self, data_dir: str = "TransformedData", n_splits: int = 5, random_state: int = 42):
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.X_train = None
        self.y_train = None

    def load_data(self):
        """Loads resampled training data."""
        self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train_sampled.csv"))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train_sampled.csv")).values.ravel()

    def evaluate_models(self) -> dict:
        """Runs Stratified 5-Fold CV on each model."""
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=self.random_state),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=self.random_state, eval_metric="mlogloss", n_jobs=1)
        }
        
        cv_scores = {}
        for name, model in models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring="accuracy", n_jobs=1)
            cv_scores[name] = scores
            print(f"{name:22} -> 5-Fold CV Mean Accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")
        return cv_scores

    def run(self) -> dict:
        """Executes cross-validation."""
        print("\n" + "="*50)
        print(" [STEP 9] 5-FOLD STRATIFIED CROSS-VALIDATION")
        print("="*50)
        self.load_data()
        return self.evaluate_models()


if __name__ == "__main__":
    validator = CrossValidator()
    validator.run()
