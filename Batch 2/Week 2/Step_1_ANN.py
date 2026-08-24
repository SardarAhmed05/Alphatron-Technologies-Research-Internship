"""
Step 1: Artificial Neural Network (ANN) for MNIST Digit Classification
Pure Object-Oriented Architecture for Deep Learning Pipeline
"""

import os
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix


class ArtificialNeuralNetwork:
    """
    Modular Object-Oriented Artificial Neural Network (ANN / Multi-Layer Perceptron)
    for handwritten digit classification using the MNIST dataset.
    """

    def __init__(
        self,
        hidden_units: Tuple[int, int] = (128, 64),
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        models_dir: str = "Models",
        visualizations_dir: str = "Visualizations",
        random_state: int = 42
    ):
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.models_dir = models_dir
        self.visualizations_dir = visualizations_dir
        self.random_state = random_state

        # Initialize internal state
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.model: Optional[Sequential] = None
        self.history: Optional[keras.callbacks.History] = None
        self.test_loss: Optional[float] = None
        self.test_accuracy: Optional[float] = None
        self.training_time: Optional[float] = None

        # Ensure output directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        # Set random seed for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Loads the MNIST dataset from Keras."""
        print("[ANN] Loading MNIST dataset...")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        print(f"[ANN] Raw Data Loaded - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes pixel values from [0, 255] to [0.0, 1.0]."""
        if self.X_train is None or self.X_test is None:
            self.load_data()

        print("[ANN] Normalizing pixel values to [0.0, 1.0] range...")
        self.X_train = self.X_train.astype("float32") / 255.0
        self.X_test = self.X_test.astype("float32") / 255.0
        return self.X_train, self.X_test

    def build_model(self) -> Sequential:
        """Constructs the Multi-Layer Perceptron architecture."""
        print(f"[ANN] Building Sequential ANN architecture with hidden units: {self.hidden_units}...")
        self.model = Sequential([
            Flatten(input_shape=(28, 28), name="input_flatten"),
            Dense(self.hidden_units[0], activation="relu", name="dense_hidden_1"),
            Dense(self.hidden_units[1], activation="relu", name="dense_hidden_2"),
            Dense(10, activation="softmax", name="dense_output")
        ], name="ANN_MNIST_Classifier")

        return self.model

    def compile_model(self) -> None:
        """Compiles the model with Adam optimizer and sparse categorical crossentropy loss."""
        if self.model is None:
            self.build_model()

        print(f"[ANN] Compiling model with Adam(lr={self.learning_rate}) and Sparse Categorical Crossentropy...")
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()

    def train(self, validation_split: float = 0.2) -> keras.callbacks.History:
        """Trains the ANN model on normalized training data."""
        if self.model is None:
            self.compile_model()

        print(f"[ANN] Training model for {self.epochs} epochs with batch size {self.batch_size}...")
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
        print(f"[ANN] Model training completed in {self.training_time:.2f} seconds.")
        return self.history

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates the trained ANN model on unseen test set."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model must be built and trained before evaluation.")

        print("[ANN] Evaluating model performance on held-out test data...")
        self.test_loss, self.test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[ANN] Test Loss: {self.test_loss:.4f} | Test Accuracy: {self.test_accuracy * 100:.2f}%")
        return self.test_loss, self.test_accuracy

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Generates class predictions for given image samples."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")
        probabilities = self.model.predict(samples, verbose=0)
        return np.argmax(probabilities, axis=1)

    def plot_training_history(self, save_filename: str = "ann_training_history.png") -> str:
        """Generates and saves high-resolution Loss and Accuracy curves."""
        if self.history is None:
            raise ValueError("No training history found. Model must be trained first.")

        save_path = os.path.join(self.visualizations_dir, save_filename)
        epochs_range = range(1, len(self.history.history["loss"]) + 1)

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history.history["loss"], "o-", label="Train Loss", color="#1f77b4")
        plt.plot(epochs_range, self.history.history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
        plt.title("ANN - Loss vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Sparse Categorical Crossentropy")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["accuracy"]], "o-", label="Train Accuracy", color="#2ca02c")
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["val_accuracy"]], "s--", label="Val Accuracy", color="#d62728")
        plt.title("ANN - Accuracy vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[ANN] Saved training history curves to: {save_path}")
        return save_path

    def plot_sample_predictions(self, num_samples: int = 10, save_filename: str = "ann_sample_predictions.png") -> str:
        """Plots sample test digits alongside True vs Predicted labels."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model and test dataset must be loaded.")

        save_path = os.path.join(self.visualizations_dir, save_filename)
        sample_indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        sample_images = self.X_test[sample_indices]
        sample_true = self.y_test[sample_indices]
        sample_preds = self.predict(sample_images)

        cols = 5
        rows = int(np.ceil(num_samples / cols))
        plt.figure(figsize=(15, 3.2 * rows))

        for idx in range(num_samples):
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(sample_images[idx], cmap="gray")
            is_correct = sample_true[idx] == sample_preds[idx]
            color = "green" if is_correct else "red"
            plt.title(f"True: {sample_true[idx]} | Pred: {sample_preds[idx]}", color=color, fontweight="bold", fontsize=11)
            plt.axis("off")

        plt.suptitle("ANN (MNIST) - Sample Test Predictions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[ANN] Saved sample predictions visualization to: {save_path}")
        return save_path

    def save_model(self, model_filename: str = "ann_mnist_model.keras") -> str:
        """Serializes the trained ANN model artifact to disk."""
        if self.model is None:
            raise ValueError("Model is not initialized.")

        save_path = os.path.join(self.models_dir, model_filename)
        self.model.save(save_path)
        print(f"[ANN] Serialized model saved to: {save_path}")
        return save_path

    def run(self) -> Dict[str, Any]:
        """Executes the full end-to-end ANN training, evaluation, and artifact generation lifecycle."""
        print("\n" + "=" * 60)
        print(" [STEP 1] ARTIFICIAL NEURAL NETWORK (ANN) PIPELINE")
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
            "model_name": "Artificial Neural Network (ANN)",
            "domain": "Computer Vision (MNIST)",
            "architecture": "MLP (Flatten -> Dense 128 -> Dense 64 -> Dense 10)",
            "parameters": trainable_params,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "training_time_sec": self.training_time,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        print("\n[ANN] Step 1 Execution Successfully Completed!")
        return results


if __name__ == "__main__":
    ann_pipeline = ArtificialNeuralNetwork()
    ann_pipeline.run()
