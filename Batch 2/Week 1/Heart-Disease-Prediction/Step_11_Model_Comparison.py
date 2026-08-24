"""
Class-based benchmarking across all models, comparative visualization, and champion export.
"""

import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")


class ModelComparator:
    """
    Evaluates all trained and tuned models on the unseen test set, generates
    comparative bar charts, and exports the top-performing model as best_model.pkl.
    """

    def __init__(self, data_dir: str = "TransformedData", models_dir: str = "Models", output_dir: str = "ModelComparison"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.X_test = None
        self.y_test = None
        self.results_df = None
        self.best_model_name = None
        self.best_model_obj = None

    def load_data(self):
        """Loads held-out test data."""
        self.X_test = pd.read_csv(os.path.join(self.data_dir, "X_test.csv"))
        self.y_test = pd.read_csv(os.path.join(self.data_dir, "y_test.csv")).values.ravel()

    def compare_models(self) -> pd.DataFrame:
        """Evaluates all candidate models."""
        models = {
            "Logistic Regression (Baseline)": "baseline_logistic_regression.pkl",
            "Random Forest (Baseline)": "baseline_random_forest.pkl",
            "XGBoost (Baseline)": "baseline_xgboost.pkl",
            "XGBoost (Tuned & SMOTE)": "tuned_xgboost.pkl"
        }
        
        results = []
        best_acc = -1
        
        for name, filename in models.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                m = joblib.load(path)
                y_pred = m.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                f1_macro = f1_score(self.y_test, y_pred, average="macro")
                f1_weighted = f1_score(self.y_test, y_pred, average="weighted")
                
                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Macro_F1": f1_macro,
                    "Weighted_F1": f1_weighted
                })
                
                if acc > best_acc:
                    best_acc = acc
                    self.best_model_name = name
                    self.best_model_obj = m
                    
        self.results_df = pd.DataFrame(results)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_df.to_csv(os.path.join(self.output_dir, "model_comparison_results.csv"), index=False)
        print("\n--- Model Comparison Summary ---")
        print(self.results_df.to_string(index=False))
        return self.results_df

    def plot_comparison(self):
        """Generates comparative accuracy bar plot."""
        plt.figure(figsize=(10, 5))
        sns.barplot(data=self.results_df, x="Model", y="Accuracy", hue="Model", palette="Blues_r", legend=False, edgecolor="black")
        plt.title("Model Accuracy Comparison on Test Data")
        plt.ylabel("Accuracy Score")
        plt.ylim(0, 1.0)
        for p in plt.gca().patches:
            h = p.get_height()
            plt.gca().text(p.get_x() + p.get_width()/2, h + 0.02, f"{h*100:.2f}%", ha="center", fontweight="bold")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "model_accuracy_comparison.png"), dpi=300)
        plt.close()

    def export_champion_model(self):
        """Saves overall best model to best_model.pkl."""
        if self.best_model_obj:
            joblib.dump(self.best_model_obj, os.path.join(self.models_dir, "best_model.pkl"))
            print(f"\nBest Selected Model: {self.best_model_name}")
            print(f"Exported to: {self.models_dir}/best_model.pkl")

    def run(self) -> pd.DataFrame:
        """Executes model comparison."""
        print("\n" + "="*50)
        print(" [STEP 11] MODEL COMPARISON & FINAL SELECTION")
        print("="*50)
        self.load_data()
        self.compare_models()
        self.plot_comparison()
        self.export_champion_model()
        return self.results_df


if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run()
