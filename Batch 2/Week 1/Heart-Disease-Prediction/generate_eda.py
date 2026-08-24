import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_charts():
    # Setup directory and visual style
    output_dir = "eda_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Modern Seaborn & Matplotlib styling
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.titleweight": "bold"
    })

    # Load dataset
    df = pd.read_csv("heart_disease_uci.csv")
    print(f"Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Add binary target for clinical context
    df["disease_present"] = (df["num"] > 0).astype(int)

    # -------------------------------------------------------------
    # 1. Missing Values Analysis
    # -------------------------------------------------------------
    print("\n--- Generating Chart 1: Missing Values Analysis ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"Feature": missing_pct.index, "Missing_Percent": missing_pct.values})
    missing_df = missing_df.sort_values(by="Missing_Percent", ascending=False)
    
    # Filter features that have missing values or show all
    colors = ["#d9534f" if p > 30 else "#f0ad4e" if p > 5 else "#5cb85c" for p in missing_df["Missing_Percent"]]
    bars = axes[0].barh(missing_df["Feature"], missing_df["Missing_Percent"], color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Missing Percentage (%)")
    axes[0].set_title("Missing Value Percentage by Feature")
    axes[0].axvline(30, color="red", linestyle="--", alpha=0.7, label="30% High-Missing Threshold")
    axes[0].legend(loc="lower right")
    
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            axes[0].text(width + 0.8, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                         va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, 75)

    # Missingness co-occurrence in key clinical features
    missing_matrix = df.drop(columns=["id", "disease_present"]).isnull().astype(int)
    corr_missing = missing_matrix.corr()
    sns.heatmap(corr_missing, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Missingness Correlation Matrix (MCAR/MAR Pattern)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_missing_values_analysis.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 2. Target Variable Distribution (5-Class & Binary)
    # -------------------------------------------------------------
    print("--- Generating Chart 2: Target Variable Distribution ---")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Multiclass
    class_counts = df["num"].value_counts().sort_index()
    class_labels = [f"Class {i}\n(Stage {i})" if i > 0 else "Class 0\n(Healthy)" for i in class_counts.index]
    palette_multi = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
    
    bars1 = axes[0].bar(class_labels, class_counts.values, color=palette_multi, edgecolor="black", alpha=0.85)
    axes[0].set_title("Multiclass Target Distribution (num: 0 to 4)")
    axes[0].set_ylabel("Patient Count")
    for bar in bars1:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        axes[0].text(bar.get_x() + bar.get_width()/2, h + 8, f"{h}\n({pct:.1f}%)", 
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 480)

    # Binary
    bin_counts = df["disease_present"].value_counts().sort_index()
    bin_labels = ["0: Healthy / No Disease", "1: Heart Disease (Stages 1-4)"]
    palette_bin = ["#2ecc71", "#e74c3c"]
    
    wedges, texts, autotexts = axes[1].pie(
        bin_counts.values, labels=bin_labels, autopct="%1.1f%%",
        startangle=140, colors=palette_bin, explode=(0.04, 0.04),
        shadow=True, textprops={"fontsize": 11, "fontweight": "semibold"}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(13)
        at.set_fontweight("bold")
    axes[1].set_title("Binary Target Distribution (Healthy vs Heart Disease)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_target_distribution.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 3. Demographics Analysis: Age & Sex vs Heart Disease
    # -------------------------------------------------------------
    print("--- Generating Chart 3: Demographics vs Heart Disease ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Age distribution
    sns.histplot(data=df, x="age", hue="disease_present", kde=True, bins=25, 
                 palette={0: "#2ecc71", 1: "#e74c3c"}, ax=axes[0], alpha=0.6)
    axes[0].set_title("Age Distribution by Heart Disease Diagnosis")
    axes[0].set_xlabel("Age (Years)")
    axes[0].set_ylabel("Patient Count")
    axes[0].legend(["Disease Present (num > 0)", "Healthy (num = 0)"], loc="upper left")

    # Sex breakdown
    sex_df = pd.crosstab(df["sex"], df["disease_present"], normalize="index") * 100
    sex_counts = pd.crosstab(df["sex"], df["disease_present"])
    
    x = np.arange(len(sex_df.index))
    width = 0.35
    b1 = axes[1].bar(x - width/2, sex_df[0], width, label="Healthy (0)", color="#2ecc71", edgecolor="black", alpha=0.85)
    b2 = axes[1].bar(x + width/2, sex_df[1], width, label="Heart Disease (1+)", color="#e74c3c", edgecolor="black", alpha=0.85)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sex_df.index, fontweight="bold")
    axes[1].set_ylabel("Percentage within Gender (%)")
    axes[1].set_title("Heart Disease Prevalence by Gender")
    axes[1].legend(loc="upper right")
    axes[1].set_ylim(0, 100)

    for bar in b1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_demographics_vs_heart_disease.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 4. Chest Pain Type Analysis
    # -------------------------------------------------------------
    print("--- Generating Chart 4: Chest Pain Type Analysis ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cp_counts = df["cp"].value_counts()
    sns.countplot(data=df, x="cp", order=cp_counts.index, hue="cp", palette="Blues_r", ax=axes[0], edgecolor="black", legend=False)
    axes[0].set_title("Overall Chest Pain Type Frequency")
    axes[0].set_xlabel("Chest Pain Type (cp)")
    axes[0].set_ylabel("Patient Count")
    for p in axes[0].patches:
        h = p.get_height()
        pct = (h / len(df)) * 100
        axes[0].text(p.get_x() + p.get_width()/2, h + 5, f"{int(h)} ({pct:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylim(0, 560)

    # Chest Pain vs Multiclass Target
    cp_target = pd.crosstab(df["cp"], df["num"], normalize="index") * 100
    cp_target.plot(kind="bar", stacked=True, ax=axes[1], colormap="Spectral_r", edgecolor="black", alpha=0.9)
    axes[1].set_title("Heart Disease Severity by Chest Pain Type")
    axes[1].set_xlabel("Chest Pain Type (cp)")
    axes[1].set_ylabel("Proportion (%)")
    axes[1].legend(title="Disease Class (num)", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_chest_pain_analysis.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 5. Clinical Vitals Distributions & Invalid Zero-Value Analysis
    # -------------------------------------------------------------
    print("--- Generating Chart 5: Clinical Vitals Distributions ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Resting BP (trestbps)
    sns.histplot(df["trestbps"].dropna(), kde=True, color="#3498db", ax=axes[0, 0], bins=30)
    axes[0, 0].set_title("Resting Blood Pressure (trestbps) Distribution")
    axes[0, 0].set_xlabel("Resting Blood Pressure (mm Hg)")
    axes[0, 0].axvline(120, color="green", linestyle="--", label="Normal Threshold (120 mm Hg)")
    axes[0, 0].axvline(140, color="red", linestyle="--", label="Hypertension Stage 2 (140 mm Hg)")
    axes[0, 0].legend()

    # Cholesterol (chol) with 0-value highlight
    zero_chol_count = (df["chol"] == 0).sum()
    sns.histplot(df["chol"].dropna(), kde=True, color="#e67e22", ax=axes[0, 1], bins=35)
    axes[0, 1].set_title(f"Serum Cholesterol (chol) - Highlighting {zero_chol_count} Biologically Invalid 0s")
    axes[0, 1].set_xlabel("Serum Cholesterol (mg/dl)")
    axes[0, 1].annotate(f"{zero_chol_count} Zero Values\n(Must be Imputed as NaN)", xy=(0, 150), xytext=(80, 180),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=10, fontweight="bold", color="darkred")

    # Max Heart Rate (thalch)
    sns.histplot(df["thalch"].dropna(), kde=True, color="#2ecc71", ax=axes[1, 0], bins=30)
    axes[1, 0].set_title("Maximum Heart Rate Achieved (thalch) Distribution")
    axes[1, 0].set_xlabel("Max Heart Rate (bpm)")

    # ST Depression (oldpeak)
    sns.histplot(df["oldpeak"].dropna(), kde=True, color="#9b59b6", ax=axes[1, 1], bins=30)
    axes[1, 1].set_title("ST Depression Induced by Exercise (oldpeak)")
    axes[1, 1].set_xlabel("ST Depression (oldpeak)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_clinical_vitals_distributions.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 6. Outliers & Boxplots Analysis across Heart Disease Presence
    # -------------------------------------------------------------
    print("--- Generating Chart 6: Outlier Boxplots ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Replace 0 with NaN for clean outlier exploration
    df_clean_vitals = df.copy()
    df_clean_vitals.loc[df_clean_vitals["trestbps"] == 0, "trestbps"] = np.nan
    df_clean_vitals.loc[df_clean_vitals["chol"] == 0, "chol"] = np.nan
    df_clean_vitals["Diagnosis"] = df_clean_vitals["disease_present"].map({0: "Healthy (0)", 1: "Heart Disease (1+)"})

    palette_disease = {"Healthy (0)": "#2ecc71", "Heart Disease (1+)": "#e74c3c"}
    
    sns.boxplot(data=df_clean_vitals, x="Diagnosis", y="age", hue="Diagnosis", palette=palette_disease, ax=axes[0, 0], legend=False)
    axes[0, 0].set_title("Age Distribution by Heart Disease Status")
    axes[0, 0].set_ylabel("Age (years)")

    sns.boxplot(data=df_clean_vitals, x="Diagnosis", y="trestbps", hue="Diagnosis", palette=palette_disease, ax=axes[0, 1], legend=False)
    axes[0, 1].set_title("Resting Blood Pressure by Heart Disease Status")
    axes[0, 1].set_ylabel("Resting BP (mm Hg)")

    sns.boxplot(data=df_clean_vitals, x="Diagnosis", y="thalch", hue="Diagnosis", palette=palette_disease, ax=axes[1, 0], legend=False)
    axes[1, 0].set_title("Max Heart Rate (thalch) by Heart Disease Status")
    axes[1, 0].set_ylabel("Max Heart Rate (bpm)")

    sns.boxplot(data=df_clean_vitals, x="Diagnosis", y="oldpeak", hue="Diagnosis", palette=palette_disease, ax=axes[1, 1], legend=False)
    axes[1, 1].set_title("ST Depression (oldpeak) by Heart Disease Status")
    axes[1, 1].set_ylabel("ST Depression")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_outliers_and_boxplots.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 7. Correlation Matrix of Clinical Features
    # -------------------------------------------------------------
    print("--- Generating Chart 7: Correlation Heatmap ---")
    fig, ax = plt.subplots(figsize=(12, 10))

    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "num", "disease_present"]
    corr_data = df_clean_vitals[numeric_cols].corr()

    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    sns.heatmap(corr_data, mask=mask, annot=True, fmt=".2f", cmap="vlag", vmin=-0.6, vmax=0.6,
                square=True, linewidths=1.0, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Pearson Correlation Heatmap of Clinical & Target Variables", pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_correlation_matrix.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 8. Cardiac Tests & Clinical Markers (exang, restecg, fbs)
    # -------------------------------------------------------------
    print("--- Generating Chart 8: Cardiac Markers & Diagnostic Tests ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Exercise Induced Angina
    exang_df = pd.crosstab(df["exang"].fillna("Missing"), df["disease_present"], normalize="index") * 100
    exang_df.plot(kind="bar", stacked=True, ax=axes[0], color=["#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[0].set_title("Exercise-Induced Angina (exang)")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xlabel("Exercise Angina")
    axes[0].legend(["Healthy", "Disease"], loc="lower right")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

    # Resting ECG
    restecg_df = pd.crosstab(df["restecg"].fillna("Missing"), df["disease_present"], normalize="index") * 100
    restecg_df.plot(kind="bar", stacked=True, ax=axes[1], color=["#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[1].set_title("Resting ECG Results (restecg)")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_xlabel("Resting ECG Type")
    axes[1].legend(["Healthy", "Disease"], loc="lower right")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15, ha="right")

    # Fasting Blood Sugar
    fbs_df = pd.crosstab(df["fbs"].fillna("Missing"), df["disease_present"], normalize="index") * 100
    fbs_df.plot(kind="bar", stacked=True, ax=axes[2], color=["#2ecc71", "#e74c3c"], edgecolor="black", alpha=0.85)
    axes[2].set_title("Fasting Blood Sugar > 120 mg/dl (fbs)")
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_xlabel("FBS Status")
    axes[2].legend(["Healthy", "Disease"], loc="lower right")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "08_cardiac_tests_angina_ecg_fbs.png"), dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # 9. Before vs. After Preprocessing & Imputation Comparison
    # -------------------------------------------------------------
    print("--- Generating Chart 9: Before vs After Preprocessing ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Cholesterol: Before (Raw with 0s) vs After (Clean & Imputed)
    raw_chol = df["chol"].dropna()
    clean_chol_series = df["chol"].replace(0, np.nan)
    median_chol = clean_chol_series.median()
    imputed_chol = clean_chol_series.fillna(median_chol)

    sns.kdeplot(raw_chol, color="#e74c3c", linewidth=2.5, label=f"Raw Data (172 invalid 0s, mean={raw_chol.mean():.1f})", ax=axes[0, 0], fill=True, alpha=0.2)
    sns.kdeplot(imputed_chol, color="#2ecc71", linewidth=2.5, label=f"After 0->NaN & Median Imputation (mean={imputed_chol.mean():.1f})", ax=axes[0, 0], fill=True, alpha=0.2)
    axes[0, 0].set_title("Serum Cholesterol: Before vs. After Imputation")
    axes[0, 0].set_xlabel("Cholesterol (mg/dl)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(loc="upper right", fontsize=9)

    # 2. Resting Blood Pressure: Before vs After
    raw_trestbps = df["trestbps"].dropna()
    clean_bp_series = df["trestbps"].replace(0, np.nan)
    median_bp = clean_bp_series.median()
    imputed_bp = clean_bp_series.fillna(median_bp)

    sns.kdeplot(raw_trestbps, color="#e74c3c", linewidth=2.5, label=f"Raw Data (contains 0 bpm, min={raw_trestbps.min():.0f})", ax=axes[0, 1], fill=True, alpha=0.2)
    sns.kdeplot(imputed_bp, color="#3498db", linewidth=2.5, label=f"After 0->NaN & Median Imputation (min={imputed_bp.min():.0f})", ax=axes[0, 1], fill=True, alpha=0.2)
    axes[0, 1].set_title("Resting Blood Pressure: Before vs. After Imputation")
    axes[0, 1].set_xlabel("Blood Pressure (mm Hg)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend(loc="upper right", fontsize=9)

    # 3. ST Depression (oldpeak): Before vs After
    raw_oldpeak = df["oldpeak"].dropna()
    clean_oldpeak = df["oldpeak"].copy()
    clean_oldpeak.loc[clean_oldpeak < 0] = np.nan
    imputed_oldpeak = clean_oldpeak.fillna(clean_oldpeak.median())

    sns.kdeplot(raw_oldpeak, color="#e74c3c", linewidth=2.5, label=f"Raw Data (contains negative values, min={raw_oldpeak.min():.1f})", ax=axes[1, 0], fill=True, alpha=0.2)
    sns.kdeplot(imputed_oldpeak, color="#9b59b6", linewidth=2.5, label=f"After Cleaning & Imputation (min={imputed_oldpeak.min():.1f})", ax=axes[1, 0], fill=True, alpha=0.2)
    axes[1, 0].set_title("ST Depression (oldpeak): Before vs. After Cleaning")
    axes[1, 0].set_xlabel("ST Depression")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend(loc="upper right", fontsize=9)

    # 4. Feature Scaling (StandardScaler Z-Scores)
    from sklearn.preprocessing import StandardScaler
    numeric_clean_df = pd.DataFrame({
        "age": df["age"],
        "trestbps": imputed_bp,
        "chol": imputed_chol,
        "thalch": df["thalch"].fillna(df["thalch"].mean()),
        "oldpeak": imputed_oldpeak
    })
    scaler = StandardScaler()
    scaled_vitals = pd.DataFrame(scaler.fit_transform(numeric_clean_df), columns=numeric_clean_df.columns)

    for col in scaled_vitals.columns:
        sns.kdeplot(scaled_vitals[col], label=f"{col} (scaled)", ax=axes[1, 1], linewidth=2)
    axes[1, 1].set_title("Continuous Features: After StandardScaler (Mean=0, Std=1)")
    axes[1, 1].set_xlabel("Standardized Value (Z-Score)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "09_before_vs_after_preprocessing.png"), dpi=300)
    plt.close()

    print("\nAll 9 EDA charts generated successfully in 'eda_charts/' directory!")

if __name__ == "__main__":
    generate_eda_charts()
