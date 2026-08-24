"""
Performs statistical imputation, categorical encoding, and feature scaling.
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def run_data_transformation(input_path="Datapreparation/cleaned_heart_disease.csv", output_dir="TransformedData"):
    print("\n" + "="*50)
    print(" [STEP 3] DATA TRANSFORMATION & ENCODING")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    # 1. Numerical Imputation (Median)
    numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # 2. Categorical Imputation (Most Frequent)
    cat_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang"]
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # 3. Binary Categorical Encoding
    df["sex"] = df["sex"].map({"male": 1, "female": 0, "1": 1, "0": 0}).fillna(1).astype(int)
    df["fbs"] = df["fbs"].map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    df["exang"] = df["exang"].map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)

    # 4. One-Hot Encoding for multi-category columns
    df = pd.get_dummies(df, columns=["dataset", "cp", "restecg"], drop_first=True)

    # 5. Continuous Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    output_path = os.path.join(output_dir, "transformed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data Transformation Complete! Shape: {df.shape}. Saved to: {output_path}")
    return df

if __name__ == "__main__":
    run_data_transformation()
