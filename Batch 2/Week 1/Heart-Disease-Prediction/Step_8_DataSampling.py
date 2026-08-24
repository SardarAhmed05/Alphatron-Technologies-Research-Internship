"""
Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance severe minority classes in training data.
"""

import os
import pandas as pd
from imblearn.over_sampling import SMOTE

def run_data_sampling(data_dir="TransformedData"):
    print("\n" + "="*50)
    print(" [STEP 8] DATA SAMPLING & CLASS BALANCING (SMOTE)")
    print("="*50)
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    
    print(f"Original Training Class Distribution: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Apply SMOTE with k_neighbors=3 (due to small minority class sizes)
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sampled, y_train_sampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled Training Class Distribution: {pd.Series(y_train_sampled).value_counts().to_dict()}")
    print(f"Resampled Dataset Shape: {X_train_sampled.shape[0]} samples")
    
    pd.DataFrame(X_train_sampled, columns=X_train.columns).to_csv(os.path.join(data_dir, "X_train_sampled.csv"), index=False)
    pd.DataFrame(y_train_sampled, columns=["num"]).to_csv(os.path.join(data_dir, "y_train_sampled.csv"), index=False)
    
    print("Data Sampling Complete! Saved X_train_sampled.csv and y_train_sampled.csv.")
    return X_train_sampled, y_train_sampled

if __name__ == "__main__":
    run_data_sampling()
