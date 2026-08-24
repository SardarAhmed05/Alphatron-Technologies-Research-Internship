"""
Class-based feature pruning of noisy, arbitrary, and high-missingness columns.
"""

import os
import pandas as pd


class FeatureSelector:
    """
    Identifies and removes arbitrary identifiers and columns with excessive missingness (>30%).
    """

    def __init__(self, input_path: str = "TransformedData/transformed_data.csv", output_dir: str = "TransformedData"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.drop_cols = ["id", "slope", "ca", "thal"]

    def load_data(self) -> pd.DataFrame:
        """Loads transformed dataset."""
        self.df = pd.read_csv(self.input_path)
        return self.df

    def select_features(self) -> pd.DataFrame:
        """Drops designated non-predictive columns."""
        existing_drops = [c for c in self.drop_cols if c in self.df.columns]
        self.df = self.df.drop(columns=existing_drops)
        print(f"Dropped non-predictive/high-missing columns: {existing_drops}")
        print(f"Selected Feature Matrix Shape: {self.df.shape}")
        return self.df

    def save_selected_features(self) -> str:
        """Exports selected features to disk."""
        output_path = os.path.join(self.output_dir, "selected_features_data.csv")
        self.df.to_csv(output_path, index=False)
        print(f"Feature Selection Complete! Saved to: {output_path}")
        return output_path

    def run(self) -> pd.DataFrame:
        """Executes feature selection."""
        print("\n" + "="*50)
        print(" [STEP 4] FEATURE SELECTION")
        print("="*50)
        self.load_data()
        self.select_features()
        self.save_selected_features()
        return self.df


if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()
