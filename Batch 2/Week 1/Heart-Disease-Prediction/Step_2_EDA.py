"""
Class-based visual profiling and statistical reporting.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class ExploratoryDataAnalysis:
    """
    Performs comprehensive statistical profiling and exports 9 publication-grade
    visualization charts into the EDA/ directory.
    """

    def __init__(self, input_path: str = "Datapreparation/cleaned_heart_disease.csv", output_dir: str = "EDA"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self._configure_styles()

    def _configure_styles(self):
        """Configures global Seaborn and Matplotlib visualization styles."""
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

    def load_data(self) -> pd.DataFrame:
        """Loads data and adds diagnostic binary helper target."""
        self.df = pd.read_csv(self.input_path)
        self.df["disease_present"] = (self.df["num"] > 0).astype(int)
        return self.df

    def export_summary_statistics(self) -> str:
        """Generates and exports descriptive statistics table."""
        os.makedirs(self.output_dir, exist_ok=True)
        stats_df = self.df.describe(include="all").transpose()
        stats_path = os.path.join(self.output_dir, "eda_summary_statistics.csv")
        stats_df.to_csv(stats_path)
        print(f"Saved statistical summary table to: {stats_path}")
        return stats_path

    def plot_missing_values(self):
        """Chart 1: Missing values and co-occurrence patterns."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
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

        missing_matrix = self.df.drop(columns=["id", "disease_present"], errors="ignore").isnull().astype(int)
        sns.heatmap(missing_matrix.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1], linewidths=0.5)
        axes[1].set_title("Missingness Pattern Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_missing_values_analysis.png"), dpi=300)
        plt.close()

    def plot_target_distribution(self):
        """Chart 2: Multiclass severity and binary diagnostic prevalence."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        class_counts = self.df["num"].value_counts().sort_index()
        class_labels = [f"Class {i}\n(Stage {i})" if i > 0 else "Class 0\n(Healthy)" for i in class_counts.index]
        palette_multi = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
        bars1 = axes[0].bar(class_labels, class_counts.values, color=palette_multi, edgecolor="black", alpha=0.85)
        axes[0].set_title("Multiclass Disease Severity Distribution (0 to 4)")
        axes[0].set_ylabel("Patient Count")
        for bar in bars1:
            h = bar.get_height()
            pct = (h / len(self.df)) * 100
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 6, f"{h} ({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[0].set_ylim(0, 480)

        bin_counts = self.df["disease_present"].value_counts().sort_index()
        axes[1].pie(
            bin_counts.values, labels=["Healthy (0)", "Heart Disease (1+)"], autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"], explode=(0.04, 0.04), shadow=True
        )
        axes[1].set_title("Binary Heart Disease Prevalence")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "02_target_distribution.png"), dpi=300)
        plt.close()

    def plot_demographics(self):
        """Chart 3: Age and sex breakdown vs heart disease."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        sns.histplot(data=self.df, x="age", hue="disease_present", kde=True, bins=20, palette=["#2ecc71", "#e74c3c"], ax=axes[0], alpha=0.6)
        axes[0].set_title("Age Distribution by Heart Disease Status")
        sns.countplot(data=self.df, x="sex", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black")
        axes[1].set_title("Heart Disease Breakdown by Sex")
        sns.countplot(data=self.df, x="dataset", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[2], edgecolor="black")
        axes[2].set_title("Prevalence across Clinical Cohorts")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_demographics_vs_heart_disease.png"), dpi=300)
        plt.close()

    def plot_chest_pain(self):
        """Chart 4: Chest pain type distribution and disease proportion."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
        sns.countplot(data=self.df, x="cp", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[0], edgecolor="black")
        axes[0].set_title("Chest Pain Type vs. Heart Disease")
        cp_pct = pd.crosstab(self.df["cp"], self.df["disease_present"], normalize="index") * 100
        cp_pct.plot(kind="bar", stacked=True, color=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black", alpha=0.85)
        axes[1].set_title("Percentage Disease Presence within Chest Pain Type")
        axes[1].set_ylabel("Percentage (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_chest_pain_analysis.png"), dpi=300)
        plt.close()

    def plot_clinical_vitals(self):
        """Chart 5: Distributions of resting vitals and exercise indicators."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        vitals = [("trestbps", "Resting Blood Pressure (mm Hg)", axes[0, 0]),
                  ("chol", "Serum Cholesterol (mg/dl)", axes[0, 1]),
                  ("thalch", "Maximum Heart Rate Achieved (bpm)", axes[1, 0]),
                  ("oldpeak", "ST Depression Induced by Exercise", axes[1, 1])]
        for col, label, ax in vitals:
            sns.kdeplot(data=self.df, x=col, hue="disease_present", palette=["#2ecc71", "#e74c3c"], fill=True, common_norm=False, ax=ax, alpha=0.4)
            ax.set_title(label)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "05_clinical_vitals_distributions.png"), dpi=300)
        plt.close()

    def plot_outliers(self):
        """Chart 6: Boxplots across vital parameters."""
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        for i, col in enumerate(["trestbps", "chol", "thalch", "oldpeak"]):
            sns.boxplot(data=self.df, x="disease_present", y=col, hue="disease_present", legend=False, palette=["#a9dfbf", "#f5b7b1"], ax=axes[i], boxprops=dict(alpha=0.8))
            axes[i].set_title(f"{col} by Disease Status")
            axes[i].set_xticks([0, 1])
            axes[i].set_xticklabels(["Healthy", "Disease"])
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "06_outliers_and_boxplots.png"), dpi=300)
        plt.close()

    def plot_correlation_matrix(self):
        """Chart 7: Pearson correlation matrix across numeric features."""
        plt.figure(figsize=(10, 8))
        numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "num", "disease_present"]
        corr_mat = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.5, vmax=0.5, square=True)
        plt.title("Correlation Heatmap of Clinical Predictors")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "07_correlation_matrix.png"), dpi=300)
        plt.close()

    def plot_cardiac_tests(self):
        """Chart 8: Angina, ECG, and blood sugar tests."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        sns.countplot(data=self.df, x="exang", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[0], edgecolor="black")
        axes[0].set_title("Exercise-Induced Angina (exang)")
        sns.countplot(data=self.df, x="restecg", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[1], edgecolor="black")
        axes[1].set_title("Resting Electrocardiogram (restecg)")
        sns.countplot(data=self.df, x="fbs", hue="disease_present", palette=["#2ecc71", "#e74c3c"], ax=axes[2], edgecolor="black")
        axes[2].set_title("Fasting Blood Sugar > 120 mg/dl (fbs)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "08_cardiac_tests_angina_ecg_fbs.png"), dpi=300)
        plt.close()

    def plot_before_after(self):
        """Chart 9: Before vs after anomaly handling for cholesterol."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        raw_chol = self.df["chol"].fillna(-50)
        axes[0].hist(raw_chol, bins=30, color="#e74c3c", edgecolor="black", alpha=0.7)
        axes[0].set_title("Raw Cholesterol (0s & Missing Values)")
        axes[0].set_xlabel("Cholesterol (mg/dl)")
        cleaned_chol = self.df["chol"].fillna(self.df["chol"].median())
        axes[1].hist(cleaned_chol, bins=30, color="#2ecc71", edgecolor="black", alpha=0.7)
        axes[1].set_title("Cleaned & Imputed Cholesterol")
        axes[1].set_xlabel("Cholesterol (mg/dl)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "09_before_vs_after_preprocessing.png"), dpi=300)
        plt.close()

    def run(self):
        """Executes full exploratory data analysis workflow."""
        print("\n" + "="*50)
        print(" [STEP 2] EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*50)
        self.load_data()
        self.export_summary_statistics()
        self.plot_missing_values()
        self.plot_target_distribution()
        self.plot_demographics()
        self.plot_chest_pain()
        self.plot_clinical_vitals()
        self.plot_outliers()
        self.plot_correlation_matrix()
        self.plot_cardiac_tests()
        self.plot_before_after()
        print(f"EDA Complete! All 9 visualization figures exported to: {self.output_dir}/")


if __name__ == "__main__":
    eda = ExploratoryDataAnalysis()
    eda.run()
