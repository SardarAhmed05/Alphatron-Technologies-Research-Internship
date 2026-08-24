"""
Class-based data ingestion, validation, and biological anomaly cleaning.
"""

import os
import numpy as np
import pandas as pd


class DataPreparation:
    """
    Handles loading the raw heart disease dataset, identifying clinical
    anomalies (e.g. impossible 0 values in cholesterol and blood pressure),
    and saving the sanitized dataset to disk.
    """

    def __init__(self, input_path: str = "heart_disease_uci.csv", output_dir: str = "Datapreparation"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Loads raw CSV data from disk."""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found at: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        print(f"Loaded Raw Dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def clean_anomalies(self) -> pd.DataFrame:
        """Corrects physiological impossibilities and normalizes string values."""
        # 1. Biological anomaly correction (0s to NaN)
        self.df["trestbps"] = self.df["trestbps"].replace(0, np.nan)
        self.df["chol"] = self.df["chol"].replace(0, np.nan)

        # 2. Negative ST depression correction
        self.df.loc[self.df["oldpeak"] < 0, "oldpeak"] = np.nan

        # 3. Clean string representations for categorical variables
        for col in ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()
                self.df[col] = self.df[col].replace({"nan": np.nan, "none": np.nan})

        return self.df

    def save_cleaned_data(self) -> str:
        """Exports the cleaned dataset to the Datapreparation folder."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "cleaned_heart_disease.csv")
        self.df.to_csv(output_path, index=False)
        print(f"Data Preparation Complete! Saved cleaned dataset to: {output_path}")
        return output_path

    def run(self) -> pd.DataFrame:
        """Executes the full data preparation lifecycle."""
        print("\n" + "="*50)
        print(" [STEP 1] DATA PREPARATION & CLEANING")
        print("="*50)
        self.load_data()
        self.clean_anomalies()
        self.save_cleaned_data()
        return self.df


if __name__ == "__main__":
    prep = DataPreparation()
    prep.run()
