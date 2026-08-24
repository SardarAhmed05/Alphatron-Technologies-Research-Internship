"""
Step 2: Exploratory Data Analysis (EDA)
=======================================
Performs comprehensive statistical profiling and exports 9 high-resolution
publication-grade exploratory data analysis figures into the EDA/ folder.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

def run_eda(input_path="Datapreparation/cleaned_heart_disease.csv", output_dir="EDA"):
    print("\n" + "="*50)
    print(" [STEP 2] EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    
    # Modern Seaborn & Matplotlib styling
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 15,
        "figure.titleweight": "bold"
    })

    # Summary stats
    stats_df = df.describe(include="all").transpose()
    stats_path = os.path.join(output_dir, "eda_summary_statistics.csv")
    stats_df.to_csv(stats_path)
    print(f"Saved statistical summary table to: {stats_path}")

    # Add binary indicator for EDA
    df["disease_present"] = (df["num"] > 0).astype(int)

    # 1. Missing Values Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Feature": missing_pct.index, "Missing_Percent": missing_pct.values}).sort_values(by="Missing_Percent", ascending=False)
    colors = ["#d9534f" if p > 30 else "#f0ad4e" if p > 5 else "#5cb85c" for p in missing_df["Missing_Percent"]]
    bars = axes[0].barh(missing_df["Feature"], missing_df["Missing_Percent"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Missing Percentage (%)")
    axes[0].set_title("Missing Value Percentage by Feature")
    axes[0].axvline(30, color="red", linestyle="--", alpha=0.7, label="30% High-Missing Threshold")
    axes[0].legend(loc="lower right")
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            axes[0].text(w + 0.8, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xlim(0, 75)

    missing_matrix = df.drop(columns=["id", "disease_present"], errors="ignore").isnull().astype(int)
    sns.heatmap(missing_matrix.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Missingness Pattern Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_missing_values_analysis.png"), dpi=300)
    plt.close()

    # 2. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    class_counts = df["num"].value_counts().sort_index()
    class_labels = [f"Class {i}\n(Stage {i})" if i > 0 else "Class 0\n(Healthy)" for i in class_counts.index]
    palette_multi = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
    bars1 = axes[0].bar(class_labels, class_counts.values, color=palette_multi, edgecolor="black", alpha=0.85)
    axes[0].set_title("Multiclass Disease Severity Distribution (0 to 4)")
    axes[0].set_ylabel("Patient Count")
    for bar in bars1:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 6, f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_ylim(0, 480)

    bin_counts = df["disease_present"].value_counts().sort_index()
    axes[1].pie(
        bin_counts.values, labels=["Healthy (0)", "Heart Disease (1+)"], autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"], explode=(0.04, 0.04), shadow=True
    )
    axes[1].set_title("Binary Heart Disease Prevalence")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_target_distribution.png"), dpi=300)
    plt.close()

    # 3. Demographics vs Disease
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sns.histplot(data=df, x="age", hue="disease_present", kde=True, bins=20, palette=["#2ecc71", "#e74c3c"], ax=axes[0], alpha=0.6)
    axes[0].set_title("Age Distribution by Heart Disease Status")
    sns.countplot(data=df, x="sex", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black")
    axes[1].set_title("Heart Disease Breakdown by Sex")
    sns.countplot(data=df, x="dataset", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[2], edgecolor="black")
    axes[2].set_title("Prevalence across Clinical Cohorts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_demographics_vs_heart_disease.png"), dpi=300)
    plt.close()

    # 4. Chest Pain Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.countplot(data=df, x="cp", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[0], edgecolor="black")
    axes[0].set_title("Chest Pain Type vs. Heart Disease")
    cp_pct = pd.crosstab(df["cp"], df["disease_present"], normalize="index") * 100
    cp_pct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black", alpha=0.85)
    axes[1].set_title("Percentage Disease Presence within Chest Pain Type")
    axes[1].set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_chest_pain_analysis.png"), dpi=300)
    plt.close()

    # 5. Clinical Vitals Distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    vitals = [("trestbps", "Resting Blood Pressure (mm Hg)", axes[0, 0]),
              ("chol", "Serum Cholesterol (mg/dl)", axes[0, 1]),
              ("thalch", "Maximum Heart Rate Achieved (bpm)", axes[1, 0]),
              ("oldpeak", "ST Depression Induced by Exercise", axes[1, 1])]
    for col, label, ax in vitals:
        sns.kdeplot(data=df, x=col, hue="disease_present", palette=["#2ecc71", "#e74c3c"], fill=True, common_norm=False, ax=ax, alpha=0.4)
        ax.set_title(label)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_clinical_vitals_distributions.png"), dpi=300)
    plt.close()

    # 6. Outliers and Boxplots
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for i, col in enumerate(["trestbps", "chol", "thalch", "oldpeak"]):
        sns.boxplot(data=df, x="disease_present", y=col, hue="disease_present", legend=False, palette=["#a9dfbf", "#f5b7b1"], ax=axes[i], boxprops=dict(alpha=0.8))
        axes[i].set_title(f"{col} by Disease Status")
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Healthy", "Disease"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_outliers_and_boxplots.png"), dpi=300)
    plt.close()

    # 7. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "num", "disease_present"]
    corr_mat = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.5, vmax=0.5, square=True)
    plt.title("Correlation Heatmap of Clinical Predictors")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_correlation_matrix.png"), dpi=300)
    plt.close()

    # 8. Cardiac Tests
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sns.countplot(data=df, x="exang", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[0], edgecolor="black")
    axes[0].set_title("Exercise-Induced Angina (exang)")
    sns.countplot(data=df, x="restecg", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black")
    axes[1].set_title("Resting Electrocardiogram (restecg)")
    sns.countplot(data=df, x="fbs", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[2], edgecolor="black")
    axes[2].set_title("Fasting Blood Sugar > 120 mg/dl (fbs)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_cardiac_tests_angina_ecg_fbs.png"), dpi=300)
    plt.close()

    # 9. Before vs After Preprocessing
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    raw_chol = df["chol"].fillna(-50)
    axes[0].hist(raw_chol, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7)
    axes[0].set_title("Raw Cholesterol (0s & Missing Values)")
    axes[0].set_xlabel("Cholesterol (mg/dl)")
    cleaned_chol = df["chol"].fillna(df["chol"].median())
    axes[1].hist(cleaned_chol, bins=30, color="#2ecc71", edgecolor="black", alpha=0.7)
    axes[1].set_title("Cleaned & Imputed Cholesterol")
    axes[1].set_xlabel("Cholesterol (mg/dl)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_before_vs_after_preprocessing.png"), dpi=300)
    plt.close()

    print(f"EDA Complete! All 9 visualization figures exported to: {output_dir}/")

if __name__ == "__main__":
    run_eda()
