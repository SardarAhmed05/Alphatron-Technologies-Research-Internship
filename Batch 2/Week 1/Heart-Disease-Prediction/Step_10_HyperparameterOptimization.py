"""
Class-based GridSearchCV hyperparameter tuning for gradient boosted trees.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier


class HyperparameterOptimizer:
    """
    Executes systematic Grid Search cross-validation to discover optimal
    hyperparameters for XGBoost on the resampled training partition.
    """

    def __init__(self, data_dir: str = "TransformedData", models_dir: str = "Models", random_state: int = 42):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.best_model = None

    def load_data(self):
        """Loads resampled training partition."""
        self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train_sampled.csv"))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train_sampled.csv")).values.ravel()

    def optimize_xgboost(self) -> XGBClassifier:
        """Performs GridSearchCV on XGBoost."""
        xgb = XGBClassifier(random_state=self.random_state, eval_metric="mlogloss", n_jobs=1)
        param_grid = {
            "n_estimators": [100, 150],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.8, 1.0]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=1, verbose=0)
        
        print("Executing GridSearchCV on XGBoost...")
        grid.fit(self.X_train, self.y_train)
        
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Accuracy: {grid.best_score_*100:.2f}%")
        self.best_model = grid.best_estimator_
        return self.best_model

    def save_tuned_model(self):
        """Serializes tuned model to disk."""
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(self.models_dir, "tuned_xgboost.pkl"))
        print(f"Saved tuned model to: {self.models_dir}/tuned_xgboost.pkl")

    def run(self) -> XGBClassifier:
        """Executes hyperparameter optimization."""
        print("\n" + "="*50)
        print(" [STEP 10] HYPERPARAMETER OPTIMIZATION (GRID SEARCH)")
        print("="*50)
        self.load_data()
        self.optimize_xgboost()
        self.save_tuned_model()
        return self.best_model


if __name__ == "__main__":
    optimizer = HyperparameterOptimizer()
    optimizer.run()
