"""
Class-based stratified dataset partitioning.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataSplitter:
    """
    Splits the selected feature matrix into stratified training (80%) and testing (20%) sets.
    """

    def __init__(self, input_path: str = "TransformedData/selected_features_data.csv", output_dir: str = "TransformedData", test_size: float = 0.2, random_state: int = 42):
        self.input_path = input_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_split(self):
        """Performs stratified train/test split."""
        df = pd.read_csv(self.input_path)
        X = df.drop(columns=["num"])
        y = df["num"].astype(int)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"Train Set Shape: {self.X_train.shape[0]} samples (80%)")
        print(f"Test Set Shape:  {self.X_test.shape[0]} samples (20%)")

    def save_splits(self):
        """Saves split partitions into CSVs."""
        self.X_train.to_csv(os.path.join(self.output_dir, "X_train.csv"), index=False)
        self.X_test.to_csv(os.path.join(self.output_dir, "X_test.csv"), index=False)
        self.y_train.to_csv(os.path.join(self.output_dir, "y_train.csv"), index=False)
        self.y_test.to_csv(os.path.join(self.output_dir, "y_test.csv"), index=False)
        print(f"Data Split Complete! Saved X_train, X_test, y_train, y_test to: {self.output_dir}/")

    def run(self):
        """Executes full splitting process."""
        print("\n" + "="*50)
        print(" [STEP 5] STRATIFIED TRAIN / TEST DATA SPLIT")
        print("="*50)
        self.load_and_split()
        self.save_splits()
        return self.X_train, self.X_test, self.y_train, self.y_test


if __name__ == "__main__":
    splitter = DataSplitter()
    splitter.run()
