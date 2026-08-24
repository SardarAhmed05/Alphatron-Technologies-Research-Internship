"""
Removes non-predictive identifiers and high-missingness columns based on EDA.
"""

import os
import pandas as pd

def run_feature_selection(input_path="TransformedData/transformed_data.csv", output_dir="TransformedData"):
    print("\n" + "="*50)
    print(" [STEP 4] FEATURE SELECTION")
    print("="*50)
    
    df = pd.read_csv(input_path)
    
    # Drop arbitrary ID and high-missing columns (>30% missing in raw data: ca, thal, slope)
    drop_cols = ["id", "slope", "ca", "thal"]
    existing_drops = [c for c in drop_cols if c in df.columns]
    
    df_selected = df.drop(columns=existing_drops)
    print(f"Dropped non-predictive/high-missing columns: {existing_drops}")
    print(f"Selected Feature Matrix Shape: {df_selected.shape}")

    output_path = os.path.join(output_dir, "selected_features_data.csv")
    df_selected.to_csv(output_path, index=False)
    print(f"Feature Selection Complete! Saved to: {output_path}")
    return df_selected

if __name__ == "__main__":
    run_feature_selection()
