"""
Partitions the dataset into Stratified 80% Training and 20% Testing sets.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def run_data_split(input_path="TransformedData/selected_features_data.csv", output_dir="TransformedData"):
    print("\n" + "="*50)
    print(" [STEP 5] STRATIFIED TRAIN / TEST DATA SPLIT")
    print("="*50)
    
    df = pd.read_csv(input_path)
    
    X = df.drop(columns=["num"])
    y = df["num"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train Set Shape: {X_train.shape[0]} samples (80%)")
    print(f"Test Set Shape:  {X_test.shape[0]} samples (20%)")
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"Data Split Complete! Saved X_train, X_test, y_train, y_test to: {output_dir}/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_data_split()
