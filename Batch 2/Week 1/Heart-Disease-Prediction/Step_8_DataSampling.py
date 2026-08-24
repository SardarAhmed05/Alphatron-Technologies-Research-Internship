"""
Class-based synthetic resampling with SMOTE to resolve class imbalance.
"""

import os
import pandas as pd
from imblearn.over_sampling import SMOTE


class DataSampler:
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) strictly to the
    training partition to balance minority severity stages without leaking into test data.
    """

    def __init__(self, data_dir: str = "TransformedData", random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.X_sampled = None
        self.y_sampled = None

    def load_data(self):
        """Loads training data."""
        self.X_train = pd.read_csv(os.path.join(self.data_dir, "X_train.csv"))
        self.y_train = pd.read_csv(os.path.join(self.data_dir, "y_train.csv")).values.ravel()
        print(f"Original Training Class Distribution: {pd.Series(self.y_train).value_counts().to_dict()}")

    def resample(self):
        """Applies SMOTE oversampling."""
        smote = SMOTE(random_state=self.random_state, k_neighbors=3)
        self.X_sampled, self.y_sampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"Resampled Training Class Distribution: {pd.Series(self.y_sampled).value_counts().to_dict()}")
        print(f"Resampled Dataset Shape: {self.X_sampled.shape[0]} samples")

    def save_resampled_data(self):
        """Exports resampled training datasets."""
        pd.DataFrame(self.X_sampled, columns=self.X_train.columns).to_csv(os.path.join(self.data_dir, "X_train_sampled.csv"), index=False)
        pd.DataFrame(self.y_sampled, columns=["num"]).to_csv(os.path.join(self.data_dir, "y_train_sampled.csv"), index=False)
        print("Data Sampling Complete! Saved X_train_sampled.csv and y_train_sampled.csv.")

    def run(self):
        """Executes data sampling."""
        print("\n" + "="*50)
        print(" [STEP 8] DATA SAMPLING & CLASS BALANCING (SMOTE)")
        print("="*50)
        self.load_data()
        self.resample()
        self.save_resampled_data()
        return self.X_sampled, self.y_sampled


if __name__ == "__main__":
    sampler = DataSampler()
    sampler.run()
