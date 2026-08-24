"""
Class-based statistical imputation, categorical encoding, and feature scaling.
"""

import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    """
    Encapsulates statistical imputation (median/mode), one-hot and binary encoding,
    and continuous feature scaling with StandardScaler.
    """

    def __init__(self, input_path: str = "Datapreparation/cleaned_heart_disease.csv", output_dir: str = "TransformedData"):
        self.input_path = input_path
        self.output_dir = output_dir
        self.df = None
        self.numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]
        self.cat_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang"]

    def load_data(self) -> pd.DataFrame:
        """Loads cleaned dataset."""
        self.df = pd.read_csv(self.input_path)
        return self.df

    def impute_missing(self) -> pd.DataFrame:
        """Applies median imputation to numeric features and mode imputation to categoricals."""
        num_imputer = SimpleImputer(strategy="median")
        self.df[self.numeric_cols] = num_imputer.fit_transform(self.df[self.numeric_cols])

        cat_imputer = SimpleImputer(strategy="most_frequent")
        self.df[self.cat_cols] = cat_imputer.fit_transform(self.df[self.cat_cols])
        return self.df

    def encode_features(self) -> pd.DataFrame:
        """Applies binary mapping and One-Hot Encoding."""
        self.df["sex"] = self.df["sex"].map({"male": 1, "female": 0, "1": 1, "0": 0}).fillna(1).astype(int)
        self.df["fbs"] = self.df["fbs"].map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
        self.df["exang"] = self.df["exang"].map({"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)

        self.df = pd.get_dummies(self.df, columns=["dataset", "cp", "restecg"], drop_first=True)
        return self.df

    def scale_features(self) -> pd.DataFrame:
        """Normalizes continuous variables using StandardScaler."""
        scaler = StandardScaler()
        self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])
        return self.df

    def save_transformed_data(self) -> str:
        """Exports transformed dataframe to disk."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "transformed_data.csv")
        self.df.to_csv(output_path, index=False)
        print(f"Data Transformation Complete! Shape: {self.df.shape}. Saved to: {output_path}")
        return output_path

    def run(self) -> pd.DataFrame:
        """Executes full transformation pipeline."""
        print("\n" + "="*50)
        print(" [STEP 3] DATA TRANSFORMATION & ENCODING")
        print("="*50)
        self.load_data()
        self.impute_missing()
        self.encode_features()
        self.scale_features()
        self.save_transformed_data()
        return self.df


if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.run()
