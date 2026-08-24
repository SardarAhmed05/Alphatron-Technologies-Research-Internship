"""
Loads the raw UCI Heart Disease dataset, validates clinical structures, handles
biological anomalies (invalid 0s in blood pressure/cholesterol), and prepares the base dataset.
"""

import os
import numpy as np
import pandas as pd

def run_data_preparation(input_path="heart_disease_uci.csv", output_dir="Datapreparation"):
    print("\n" + "="*50)
    print(" [STEP 1] DATA PREPARATION & CLEANING")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    df = pd.read_csv(input_path)
    print(f"Loaded Raw Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Biological anomaly correction (0s to NaN)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df["chol"] = df["chol"].replace(0, np.nan)
    
    # 2. Negative ST depression correction
    df.loc[df["oldpeak"] < 0, "oldpeak"] = np.nan
    
    # 3. Clean string representations for categorical variables
    for col in ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "none": np.nan})

    output_path = os.path.join(output_dir, "cleaned_heart_disease.csv")
    df.to_csv(output_path, index=False)
    print(f"Data Preparation Complete! Saved cleaned dataset to: {output_path}")
    return df

if __name__ == "__main__":
    run_data_preparation()
