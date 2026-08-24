"""
Step 4: Long Short-Term Memory (LSTM) for IMDB Sentiment Classification
Pure Object-Oriented Architecture for Deep Learning Pipeline
"""

import os
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


class LSTMNeuralNetwork:
    """
    Modular Object-Oriented Long Short-Term Memory (LSTM) network
    for capturing long-term sequential dependencies in sentiment classification.
    """

    def __init__(
        self,
        num_words: int = 10000,
        max_len: int = 200,
        embedding_dim: int = 64,
        lstm_units: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 5,
        batch_size: int = 64,
        models_dir: str = "Models",
        visualizations_dir: str = "Visualizations",
        random_state: int = 42
    ):
        self.num_words = num_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.models_dir = models_dir
        self.visualizations_dir = visualizations_dir
        self.random_state = random_state

        # Internal state
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.word_index: Optional[Dict[str, int]] = None
        self.reverse_word_index: Optional[Dict[int, str]] = None
        self.model: Optional[Sequential] = None
        self.history: Optional[keras.callbacks.History] = None
        self.test_loss: Optional[float] = None
        self.test_accuracy: Optional[float] = None
        self.training_time: Optional[float] = None

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Loads the IMDB movie reviews sentiment dataset."""
        print(f"[LSTM] Loading IMDB dataset (top {self.num_words} words)...")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = imdb.load_data(num_words=self.num_words)
        self.word_index = imdb.get_word_index()
        self.reverse_word_index = {value + 3: key for key, value in self.word_index.items()}
        self.reverse_word_index[0] = "<PAD>"
        self.reverse_word_index[1] = "<START>"
        self.reverse_word_index[2] = "<UNK>"
        self.reverse_word_index[3] = "<UNUSED>"

        print(f"[LSTM] Raw Data Loaded - Train Sequences: {len(self.X_train)}, Test Sequences: {len(self.X_test)}")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Pads sequence vectors to uniform length of max_len."""
        if self.X_train is None or self.X_test is None:
            self.load_data()

        print(f"[LSTM] Padding sequence representations to uniform length of {self.max_len} tokens...")
        self.X_train = pad_sequences(self.X_train, maxlen=self.max_len)
        self.X_test = pad_sequences(self.X_test, maxlen=self.max_len)
        print(f"[LSTM] Padded Data Shape - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test

    def build_model(self) -> Sequential:
        """Constructs the sequential LSTM neural network architecture."""
        print(f"[LSTM] Building LSTM Model (Embedding {self.embedding_dim}d -> LSTM {self.lstm_units} -> Sigmoid)...")
        self.model = Sequential([
            Embedding(
                input_dim=self.num_words,
                output_dim=self.embedding_dim,
                name="embedding_layer"
            ),
            LSTM(units=self.lstm_units, name="lstm_memory_layer"),
            Dense(1, activation="sigmoid", name="dense_sentiment_output")
        ], name="LSTM_IMDB_Sentiment_Classifier")

        return self.model

    def compile_model(self) -> None:
        """Compiles the LSTM with Adam optimizer and binary crossentropy loss."""
        if self.model is None:
            self.build_model()

        print(f"[LSTM] Compiling LSTM model with Adam(lr={self.learning_rate}) and Binary Crossentropy...")
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()

    def train(self, validation_split: float = 0.2) -> keras.callbacks.History:
        """Trains the LSTM model on padded sequence tensors."""
        if self.model is None:
            self.compile_model()

        print(f"[LSTM] Training LSTM for {self.epochs} epochs with batch size {self.batch_size}...")
        start_time = time.time()
        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            verbose=1
        )
        self.training_time = time.time() - start_time
        print(f"[LSTM] Model training completed in {self.training_time:.2f} seconds.")
        return self.history

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates model generalization performance on unseen test reviews."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model must be built and trained before evaluation.")

        print("[LSTM] Evaluating LSTM model performance on held-out test reviews...")
        self.test_loss, self.test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[LSTM] Test Loss: {self.test_loss:.4f} | Test Accuracy: {self.test_accuracy * 100:.2f}%")
        return self.test_loss, self.test_accuracy

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Generates continuous probability and discrete class predictions."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")
        probabilities = self.model.predict(samples, verbose=0)
        return (probabilities >= 0.5).astype(int)

    def decode_review(self, token_indices: np.ndarray) -> str:
        """Decodes token IDs back into human-readable English text."""
        if self.reverse_word_index is None:
            return ""
        return " ".join([self.reverse_word_index.get(i, "?") for i in token_indices if i > 3])

    def plot_training_history(self, save_filename: str = "lstm_training_history.png") -> str:
        """Generates and saves high-resolution Loss and Accuracy curves for LSTM."""
        if self.history is None:
            raise ValueError("No training history found. Model must be trained first.")

        save_path = os.path.join(self.visualizations_dir, save_filename)
        epochs_range = range(1, len(self.history.history["loss"]) + 1)

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history.history["loss"], "o-", label="Train Loss", color="#1f77b4")
        plt.plot(epochs_range, self.history.history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
        plt.title("LSTM - Loss vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Crossentropy")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["accuracy"]], "o-", label="Train Accuracy", color="#2ca02c")
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["val_accuracy"]], "s--", label="Val Accuracy", color="#d62728")
        plt.title("LSTM - Accuracy vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[LSTM] Saved training history curves to: {save_path}")
        return save_path

    def plot_sample_predictions(self, num_samples: int = 4, save_filename: str = "lstm_sample_predictions.png") -> str:
        """Visualizes sample review excerpts alongside LSTM sentiment predictions."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model and test dataset must be loaded.")

        save_path = os.path.join(self.visualizations_dir, save_filename)
        sample_indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        sample_seqs = self.X_test[sample_indices]
        sample_true = self.y_test[sample_indices]
        sample_probs = self.model.predict(sample_seqs, verbose=0).ravel()

        labels_map = {0: "Negative (0)", 1: "Positive (1)"}

        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2.5 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i, idx in enumerate(range(num_samples)):
            ax = axes[i]
            review_text = self.decode_review(sample_seqs[idx])
            truncated_text = (review_text[:250] + "...") if len(review_text) > 250 else review_text
            pred_class = 1 if sample_probs[idx] >= 0.5 else 0
            is_correct = pred_class == sample_true[idx]
            status_color = "#2ca02c" if is_correct else "#d62728"

            ax.text(0.01, 0.75, f"Review excerpt: \"{truncated_text}\"", wrap=True, fontsize=9.5, transform=ax.transAxes, verticalalignment="top")
            ax.text(0.01, 0.20, f"True: {labels_map[sample_true[idx]]}  |  Predicted: {labels_map[pred_class]}  (Confidence: {sample_probs[idx]*100:.1f}%)",
                    fontsize=10.5, fontweight="bold", color=status_color, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(status_color)
                spine.set_linewidth(1.5)

        plt.suptitle("LSTM (IMDB Sentiment) - Sample Predictions", fontsize=13, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[LSTM] Saved sample predictions visualization to: {save_path}")
        return save_path

    def save_model(self, model_filename: str = "lstm_imdb_model.keras") -> str:
        """Serializes trained LSTM model artifact to disk."""
        if self.model is None:
            raise ValueError("Model is not initialized.")

        save_path = os.path.join(self.models_dir, model_filename)
        self.model.save(save_path)
        print(f"[LSTM] Serialized model saved to: {save_path}")
        return save_path

    def run(self) -> Dict[str, Any]:
        """Executes full end-to-end LSTM training, evaluation, and artifact generation lifecycle."""
        print("\n" + "=" * 60)
        print(" [STEP 4] LONG SHORT-TERM MEMORY (LSTM) PIPELINE")
        print("=" * 60)

        self.load_data()
        self.preprocess_data()
        self.build_model()
        self.compile_model()
        self.train()
        self.evaluate()
        self.plot_training_history()
        self.plot_sample_predictions()
        self.save_model()

        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])

        results = {
            "model_name": "Long Short-Term Memory (LSTM)",
            "domain": "Natural Language Processing (IMDB)",
            "architecture": f"Embedding ({self.embedding_dim}d) -> LSTM ({self.lstm_units}) -> Sigmoid",
            "parameters": trainable_params,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "training_time_sec": self.training_time,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        print("\n[LSTM] Step 4 Execution Successfully Completed!")
        return results


if __name__ == "__main__":
    lstm_pipeline = LSTMNeuralNetwork()
    lstm_pipeline.run()
