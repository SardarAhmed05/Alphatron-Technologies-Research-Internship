"""
Step 5: Deep Learning Model Benchmarking & Comparative Analysis
Pure Object-Oriented Architecture for Deep Learning Pipeline
"""

import os
import warnings
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.datasets import mnist, imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")


class DLModelComparator:
    """
    Evaluates and benchmarks all trained Deep Learning architectures (ANN, CNN, RNN, LSTM),
    generates domain-specific and cross-architecture comparison plots, and exports structured CSV reports.
    """

    def __init__(
        self,
        models_dir: str = "Models",
        output_dir: str = "ModelComparison",
        visualizations_dir: str = "Visualizations"
    ):
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.visualizations_dir = visualizations_dir
        self.results_df: Optional[pd.DataFrame] = None
        self.benchmark_records: List[Dict[str, Any]] = []

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def add_benchmark_result(self, result: Dict[str, Any]) -> None:
        """Appends an in-memory benchmark result dictionary from pipeline execution."""
        self.benchmark_records.append(result)

    def evaluate_saved_models_if_needed(self) -> None:
        """If benchmark records are empty, loads serialized .keras models and evaluates them."""
        if len(self.benchmark_records) >= 4:
            return

        print("[Comparator] Evaluating serialized models from disk...")
        self.benchmark_records = []

        # 1. Load MNIST for Vision models
        (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
        X_test_mnist_norm = X_test_mnist.astype("float32") / 255.0

        # Evaluate ANN
        ann_path = os.path.join(self.models_dir, "ann_mnist_model.keras")
        if os.path.exists(ann_path):
            ann_model = load_model(ann_path)
            loss, acc = ann_model.evaluate(X_test_mnist_norm, y_test_mnist, verbose=0)
            params = sum([tf.keras.backend.count_params(w) for w in ann_model.trainable_weights])
            self.benchmark_records.append({
                "model_name": "Artificial Neural Network (ANN)",
                "domain": "Computer Vision (MNIST)",
                "architecture": "MLP (Flatten -> Dense 128 -> Dense 64 -> Softmax 10)",
                "parameters": params,
                "test_accuracy": acc,
                "test_loss": loss,
                "training_time_sec": None,
                "epochs": 10,
                "batch_size": 32
            })

        # Evaluate CNN
        cnn_path = os.path.join(self.models_dir, "cnn_mnist_model.keras")
        if os.path.exists(cnn_path):
            cnn_model = load_model(cnn_path)
            X_test_cnn = X_test_mnist_norm.reshape(-1, 28, 28, 1)
            loss, acc = cnn_model.evaluate(X_test_cnn, y_test_mnist, verbose=0)
            params = sum([tf.keras.backend.count_params(w) for w in cnn_model.trainable_weights])
            self.benchmark_records.append({
                "model_name": "Convolutional Neural Network (CNN)",
                "domain": "Computer Vision (MNIST)",
                "architecture": "Conv2D (32) -> MaxPool -> Conv2D (64) -> MaxPool -> Dense (128) -> Dropout -> Softmax",
                "parameters": params,
                "test_accuracy": acc,
                "test_loss": loss,
                "training_time_sec": None,
                "epochs": 10,
                "batch_size": 64
            })

        # 2. Load IMDB for Sequence models
        (X_train_imdb, y_train_imdb), (X_test_imdb, y_test_imdb) = imdb.load_data(num_words=10000)
        X_test_imdb_pad = pad_sequences(X_test_imdb, maxlen=200)

        # Evaluate RNN
        rnn_path = os.path.join(self.models_dir, "rnn_imdb_model.keras")
        if os.path.exists(rnn_path):
            rnn_model = load_model(rnn_path)
            loss, acc = rnn_model.evaluate(X_test_imdb_pad, y_test_imdb, verbose=0)
            params = sum([tf.keras.backend.count_params(w) for w in rnn_model.trainable_weights])
            self.benchmark_records.append({
                "model_name": "Recurrent Neural Network (SimpleRNN)",
                "domain": "Natural Language Processing (IMDB)",
                "architecture": "Embedding (32d) -> SimpleRNN (64) -> Dense Sigmoid (1)",
                "parameters": params,
                "test_accuracy": acc,
                "test_loss": loss,
                "training_time_sec": None,
                "epochs": 5,
                "batch_size": 64
            })

        # Evaluate LSTM
        lstm_path = os.path.join(self.models_dir, "lstm_imdb_model.keras")
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path)
            loss, acc = lstm_model.evaluate(X_test_imdb_pad, y_test_imdb, verbose=0)
            params = sum([tf.keras.backend.count_params(w) for w in lstm_model.trainable_weights])
            self.benchmark_records.append({
                "model_name": "Long Short-Term Memory (LSTM)",
                "domain": "Natural Language Processing (IMDB)",
                "architecture": "Embedding (64d) -> LSTM (64) -> Dense Sigmoid (1)",
                "parameters": params,
                "test_accuracy": acc,
                "test_loss": loss,
                "training_time_sec": None,
                "epochs": 5,
                "batch_size": 64
            })

    def generate_summary_table(self) -> pd.DataFrame:
        """Builds and prints a clean formatted benchmark DataFrame."""
        self.evaluate_saved_models_if_needed()
        self.results_df = pd.DataFrame(self.benchmark_records)

        # Format display dataframe
        csv_path = os.path.join(self.output_dir, "deep_learning_benchmark_results.csv")
        self.results_df.to_csv(csv_path, index=False)

        print("\n" + "=" * 85)
        print("                 DEEP LEARNING ARCHITECTURAL BENCHMARK REPORT")
        print("=" * 85)
        display_df = self.results_df.copy()
        display_df["test_accuracy"] = display_df["test_accuracy"].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else "N/A")
        display_df["test_loss"] = display_df["test_loss"].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        display_df["parameters"] = display_df["parameters"].apply(lambda x: f"{x:,}" if pd.notnull(x) else "N/A")
        print(display_df[["model_name", "domain", "test_accuracy", "test_loss", "parameters"]].to_string(index=False))
        print("=" * 85)
        print(f"[Comparator] Saved benchmark metrics CSV to: {csv_path}")

        return self.results_df

    def plot_comparisons(self) -> None:
        """Generates domain-wise and unified benchmark visualizations."""
        if self.results_df is None or len(self.results_df) == 0:
            self.generate_summary_table()

        sns.set_theme(style="whitegrid")

        # 1. Vision Models Comparison (ANN vs CNN)
        vision_df = self.results_df[self.results_df["domain"].str.contains("Computer Vision")].copy()
        if len(vision_df) >= 2:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

            # Accuracy
            sns.barplot(data=vision_df, x="model_name", y="test_accuracy", ax=ax[0], palette=["#3498db", "#2ecc71"], edgecolor="black")
            ax[0].set_title("Vision Models: Test Accuracy", fontsize=12, fontweight="bold")
            ax[0].set_ylabel("Accuracy")
            ax[0].set_ylim(0.9, 1.0)
            for p in ax[0].patches:
                h = p.get_height()
                ax[0].text(p.get_x() + p.get_width() / 2, h + 0.003, f"{h * 100:.2f}%", ha="center", fontweight="bold")

            # Loss
            sns.barplot(data=vision_df, x="model_name", y="test_loss", ax=ax[1], palette=["#e74c3c", "#e67e22"], edgecolor="black")
            ax[1].set_title("Vision Models: Test Loss (Lower is Better)", fontsize=12, fontweight="bold")
            ax[1].set_ylabel("Crossentropy Loss")
            for p in ax[1].patches:
                h = p.get_height()
                ax[1].text(p.get_x() + p.get_width() / 2, h + 0.003, f"{h:.4f}", ha="center", fontweight="bold")

            plt.tight_layout()
            vis_path = os.path.join(self.output_dir, "vision_models_comparison.png")
            plt.savefig(vis_path, dpi=300)
            plt.close()
            print(f"[Comparator] Saved Vision Models Comparison to: {vis_path}")

        # 2. Sequence Models Comparison (RNN vs LSTM)
        seq_df = self.results_df[self.results_df["domain"].str.contains("Natural Language")].copy()
        if len(seq_df) >= 2:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

            # Accuracy
            sns.barplot(data=seq_df, x="model_name", y="test_accuracy", ax=ax[0], palette=["#9b59b6", "#1abc9c"], edgecolor="black")
            ax[0].set_title("Sequence Models: Test Accuracy", fontsize=12, fontweight="bold")
            ax[0].set_ylabel("Accuracy")
            ax[0].set_ylim(0.7, 1.0)
            for p in ax[0].patches:
                h = p.get_height()
                ax[0].text(p.get_x() + p.get_width() / 2, h + 0.005, f"{h * 100:.2f}%", ha="center", fontweight="bold")

            # Loss
            sns.barplot(data=seq_df, x="model_name", y="test_loss", ax=ax[1], palette=["#f39c12", "#d35400"], edgecolor="black")
            ax[1].set_title("Sequence Models: Test Loss (Lower is Better)", fontsize=12, fontweight="bold")
            ax[1].set_ylabel("Binary Crossentropy Loss")
            for p in ax[1].patches:
                h = p.get_height()
                ax[1].text(p.get_x() + p.get_width() / 2, h + 0.005, f"{h:.4f}", ha="center", fontweight="bold")

            plt.tight_layout()
            seq_path = os.path.join(self.output_dir, "sequence_models_comparison.png")
            plt.savefig(seq_path, dpi=300)
            plt.close()
            print(f"[Comparator] Saved Sequence Models Comparison to: {seq_path}")

        # 3. Overall Cross-Architecture Comparison
        if len(self.results_df) >= 4:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))

            # All accuracies
            palette = ["#3498db", "#2ecc71", "#9b59b6", "#1abc9c"]
            sns.barplot(data=self.results_df, x="model_name", y="test_accuracy", hue="domain", ax=ax[0], dodge=False, edgecolor="black")
            ax[0].set_title("Cross-Architecture Test Accuracy Benchmark", fontsize=13, fontweight="bold")
            ax[0].set_ylabel("Accuracy Score")
            ax[0].set_ylim(0.7, 1.05)
            ax[0].tick_params(axis="x", rotation=15)
            for p in ax[0].patches:
                h = p.get_height()
                if h > 0:
                    ax[0].text(p.get_x() + p.get_width() / 2, h + 0.008, f"{h * 100:.2f}%", ha="center", fontweight="bold", fontsize=9.5)

            # Parameter counts (Log Scale)
            sns.barplot(data=self.results_df, x="model_name", y="parameters", hue="domain", ax=ax[1], dodge=False, edgecolor="black")
            ax[1].set_title("Model Parameter Complexity (Total Trainable Weights)", fontsize=13, fontweight="bold")
            ax[1].set_ylabel("Number of Parameters (Log Scale)")
            ax[1].set_yscale("log")
            ax[1].tick_params(axis="x", rotation=15)
            for p in ax[1].patches:
                h = p.get_height()
                if h > 0:
                    ax[1].text(p.get_x() + p.get_width() / 2, h * 1.08, f"{int(h):,}", ha="center", fontweight="bold", fontsize=9)

            plt.tight_layout()
            overall_path = os.path.join(self.output_dir, "overall_dl_benchmark_comparison.png")
            plt.savefig(overall_path, dpi=300)
            plt.close()
            print(f"[Comparator] Saved Overall Benchmark Comparison to: {overall_path}")

    def run(self) -> pd.DataFrame:
        """Executes full benchmarking lifecycle."""
        print("\n" + "=" * 60)
        print(" [STEP 5] DEEP LEARNING MODEL COMPARISON & BENCHMARKING")
        print("=" * 60)
        self.generate_summary_table()
        self.plot_comparisons()
        print("\n[Comparator] Step 5 Execution Successfully Completed!")
        return self.results_df


if __name__ == "__main__":
    comparator = DLModelComparator()
    comparator.run()
