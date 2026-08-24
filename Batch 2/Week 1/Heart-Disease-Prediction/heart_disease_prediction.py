"""
Heart Disease Prediction - Machine Learning Pipeline
=====================================================
A clean, object-oriented machine learning pipeline for UCI Heart Disease classification.
Applies proper separation of concerns (Data Loading, Preprocessing, Model Training, Evaluation).
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier


class DataLoader:
    """
    Handles data ingestion, structural validation, and initial biological cleaning.
    """

    def __init__(self, data_path: str = "heart_disease_uci.csv"):
        self.data_path = data_path

    def load_and_clean(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Loads the dataset, handles invalid zeroes/values, drops non-predictive/high-missing columns,
        and returns separated feature matrix X and target y.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        df = pd.read_csv(self.data_path)
        print(f"[DataLoader] Successfully loaded dataset with shape: {df.shape}")

        # 1. Handle biological impossibilities: Replace invalid 0s with NaN
        df["trestbps"] = df["trestbps"].replace(0, np.nan)
        df["chol"] = df["chol"].replace(0, np.nan)

        # 2. Handle invalid negative values in oldpeak
        df.loc[df["oldpeak"] < 0, "oldpeak"] = np.nan

        # 3. Drop columns with excessive missingness (>30%) and arbitrary identifiers
        drop_cols = ["id", "slope", "ca", "thal"]
        df = df.drop(columns=[col for col in drop_cols if col in df.columns])
        print(f"[DataLoader] Dropped non-predictive/high-missingness columns: {drop_cols}")

        # 4. Standardize binary columns to clean boolean/string format for encoding
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype(str)
        if "fbs" in df.columns:
            df["fbs"] = df["fbs"].astype(str)
        if "exang" in df.columns:
            df["exang"] = df["exang"].astype(str)

        # Separate features (X) and target (y)
        X = df.drop(columns=["num"])
        y = df["num"].astype(int)

        print(f"[DataLoader] Features shape: {X.shape}, Target distribution:\n{y.value_counts().sort_index().to_dict()}")
        return X, y


class DataPreprocessor:
    """
    Constructs leakage-free scikit-learn ColumnTransformers for imputation, scaling, and encoding.
    """

    def __init__(self, numeric_features: list, categorical_features: list):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def build_transformer(self) -> ColumnTransformer:
        """
        Builds and returns a ColumnTransformer encapsulating all preprocessing logic.
        """
        # Numerical pipeline: Median Imputation followed by Feature Scaling
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical pipeline: Most Frequent Imputation followed by One-Hot Encoding
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features)
            ],
            remainder="drop"
        )
        return preprocessor


class ModelTrainer:
    """
    Handles model pipeline assembly, cross-validation, hyperparameter tuning, and serialization.
    """

    def __init__(self, preprocessor: ColumnTransformer, random_state: int = 42):
        self.preprocessor = preprocessor
        self.random_state = random_state

    def create_pipeline(self, classifier) -> Pipeline:
        """
        Combines preprocessing and classification into an end-to-end pipeline.
        """
        return Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", classifier)
        ])

    def tune_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5
    ) -> Pipeline:
        """
        Performs Stratified 5-Fold Grid Search to tune XGBoost hyperparameters.
        """
        xgb = XGBClassifier(
            random_state=self.random_state,
            eval_metric="mlogloss",
            n_jobs=1
        )

        pipeline = self.create_pipeline(xgb)

        if param_grid is None:
            param_grid = {
                "classifier__n_estimators": [100, 150],
                "classifier__max_depth": [3, 4],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0]
            }

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        print(f"[ModelTrainer] Launching GridSearchCV with {cv_folds}-Fold Stratified Cross-Validation...", flush=True)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_weighted",
            n_jobs=1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        print(f"[ModelTrainer] Best Parameters: {grid.best_params_}")
        print(f"[ModelTrainer] Best CV Weighted F1-Score: {grid.best_score_:.4f}")

        return grid.best_estimator_

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Trains a baseline Random Forest model for performance comparison.
        """
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            random_state=self.random_state,
            class_weight="balanced"
        )
        pipeline = self.create_pipeline(rf)
        pipeline.fit(X_train, y_train)
        return pipeline

    @staticmethod
    def save_model(model: Pipeline, filepath: str = "xgboost_model.pkl") -> None:
        """
        Serializes the trained pipeline object to disk using joblib.
        """
        joblib.dump(model, filepath)
        print(f"[ModelTrainer] Successfully saved trained pipeline to: {filepath}")


class ModelEvaluator:
    """
    Computes comprehensive evaluation metrics and visualizations.
    """

    @staticmethod
    def evaluate(
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Model",
        save_cm_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates model predictions against ground truth and returns metric dictionary.
        """
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        print(f"\n{'='*55}")
        print(f"       Evaluation Results: {model_name}")
        print(f"{'='*55}")
        print(f"Accuracy:            {acc * 100:.2f}%")
        print(f"Macro F1-Score:      {f1_macro:.4f}")
        print(f"Weighted F1-Score:   {f1_weighted:.4f}")
        print(f"\nClassification Report:\n{report}")

        if save_cm_path:
            plt.figure(figsize=(7, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"Confusion Matrix - {model_name}")
            plt.xlabel("Predicted Class")
            plt.ylabel("True Class")
            plt.tight_layout()
            plt.savefig(save_cm_path, dpi=300)
            plt.close()
            print(f"[ModelEvaluator] Saved confusion matrix plot to {save_cm_path}")

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report
        }


class HeartDiseasePipeline:
    """
    Main pipeline orchestrator coordinating data loading, preprocessing, training, and evaluation.
    """

    def __init__(self, data_path: str = "heart_disease_uci.csv", output_model_path: str = "xgboost_model.pkl"):
        self.data_path = data_path
        self.output_model_path = output_model_path
        self.data_loader = DataLoader(self.data_path)
        self.evaluator = ModelEvaluator()

    def run(self) -> None:
        """
        Executes the full machine learning lifecycle.
        """
        print("\n" + "#"*60)
        print("   STARTING HEART DISEASE MACHINE LEARNING PIPELINE")
        print("#"*60 + "\n")

        # Step 1: Load and Clean Data
        X, y = self.data_loader.load_and_clean()

        # Step 2: Identify Feature Types
        numeric_features = ["age", "trestbps", "chol", "thalch", "oldpeak"]
        categorical_features = ["sex", "dataset", "cp", "fbs", "restecg", "exang"]

        # Step 3: Stratified Train/Test Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        print(f"[Pipeline] Stratified Split -> Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

        # Step 4: Build Preprocessing Pipeline
        preprocessor = DataPreprocessor(numeric_features, categorical_features).build_transformer()

        # Step 5: Train & Tune Models
        trainer = ModelTrainer(preprocessor=preprocessor, random_state=42)

        # Baseline: Random Forest
        print("\n--- Training Random Forest Baseline ---")
        rf_model = trainer.train_random_forest(X_train, y_train)
        self.evaluator.evaluate(rf_model, X_test, y_test, model_name="Random Forest Baseline")

        # Tuned Model: XGBoost
        print("\n--- Hyperparameter Tuning: XGBoost Classifier ---")
        best_xgb_model = trainer.tune_xgboost(X_train, y_train)
        self.evaluator.evaluate(
            best_xgb_model,
            X_test,
            y_test,
            model_name="Tuned XGBoost Classifier",
            save_cm_path="eda_charts/confusion_matrix_xgb.png"
        )

        # Step 6: Serialize Best Model
        trainer.save_model(best_xgb_model, self.output_model_path)

        print("\n" + "#"*60)
        print("   PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("#"*60 + "\n")


def main():
    pipeline = HeartDiseasePipeline(
        data_path="heart_disease_uci.csv",
        output_model_path="xgboost_model.pkl"
    )
    pipeline.run()


if __name__ == "__main__":
    main()
