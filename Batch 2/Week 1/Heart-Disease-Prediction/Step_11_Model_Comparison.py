"""
Benchmarks all trained models on test data, generates comparative plots, and saves best_model.pkl.
"""

import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

def run_model_comparison(data_dir="TransformedData", models_dir="Models", output_dir="ModelComparison"):
    print("\n" + "="*50)
    print(" [STEP 11] MODEL COMPARISON & FINAL SELECTION")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    
    models = {
        "Logistic Regression (Baseline)": "baseline_logistic_regression.pkl",
        "Random Forest (Baseline)": "baseline_random_forest.pkl",
        "XGBoost (Baseline)": "baseline_xgboost.pkl",
        "XGBoost (Tuned & SMOTE)": "tuned_xgboost.pkl"
    }
    
    results = []
    best_acc = -1
    best_name = None
    best_model_obj = None
    
    for name, filename in models.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            m = joblib.load(path)
            y_pred = m.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Macro_F1": f1_macro,
                "Weighted_F1": f1_weighted
            })
            
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_model_obj = m
                
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
    print("\n--- Model Comparison Summary ---")
    print(results_df.to_string(index=False))
    
    # Comparative Plot
    plt.figure(figsize=(10, 5))
    sns.barplot(data=results_df, x="Model", y="Accuracy", hue="Model", palette="Blues_r", legend=False, edgecolor="black")
    plt.title("Model Accuracy Comparison on Test Data")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1.0)
    for p in plt.gca().patches:
        h = p.get_height()
        plt.gca().text(p.get_x() + p.get_width()/2, h + 0.02, f"{h*100:.2f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # Save Best Model
    if best_model_obj:
        joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))
        print(f"\nBest Selected Model: {best_name} (Accuracy: {best_acc*100:.2f}%)")
        print(f"Exported to: {models_dir}/best_model.pkl")

if __name__ == "__main__":
    run_model_comparison()
