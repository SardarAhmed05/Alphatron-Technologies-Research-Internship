"""
Step 2: Convolutional Neural Network (CNN) for MNIST Digit Classification
Pure Object-Oriented Architecture for Deep Learning Pipeline
"""

import os
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


class ConvolutionalNeuralNetwork:
    """
    Modular Object-Oriented Convolutional Neural Network (CNN) for image
    feature extraction and digit classification using the MNIST dataset.
    """

    def __init__(
        self,
        conv_filters: Tuple[int, int] = (32, 64),
        kernel_size: Tuple[int, int] = (3, 3),
        pool_size: Tuple[int, int] = (2, 2),
        dense_units: int = 128,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 64,
        models_dir: str = "Models",
        visualizations_dir: str = "Visualizations",
        random_state: int = 42
    ):
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
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
        """Loads raw MNIST dataset from Keras."""
        print("[CNN] Loading MNIST dataset...")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        print(f"[CNN] Raw Data Loaded - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalizes and reshapes image matrices to (N, 28, 28, 1) for Conv2D layers."""
        if self.X_train is None or self.X_test is None:
            self.load_data()

        print("[CNN] Normalizing pixel values to [0.0, 1.0] and reshaping for Conv2D...")
        self.X_train = (self.X_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
        self.X_test = (self.X_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
        print(f"[CNN] Reshaped Data - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return self.X_train, self.X_test

    def build_model(self) -> Sequential:
        """Constructs 2D Convolutional neural network architecture."""
        print("[CNN] Building CNN Architecture (Conv2D -> MaxPool -> Conv2D -> MaxPool -> Dense -> Dropout -> Softmax)...")
        self.model = Sequential([
            Conv2D(
                filters=self.conv_filters[0],
                kernel_size=self.kernel_size,
                activation="relu",
                input_shape=(28, 28, 1),
                name="conv2d_1"
            ),
            MaxPooling2D(pool_size=self.pool_size, name="maxpool_1"),
            Conv2D(
                filters=self.conv_filters[1],
                kernel_size=self.kernel_size,
                activation="relu",
                name="conv2d_2"
            ),
            MaxPooling2D(pool_size=self.pool_size, name="maxpool_2"),
            Flatten(name="flatten"),
            Dense(self.dense_units, activation="relu", name="dense_features"),
            Dropout(self.dropout_rate, name="dropout_regularization"),
            Dense(10, activation="softmax", name="dense_output")
        ], name="CNN_MNIST_Classifier")

        return self.model

    def compile_model(self) -> None:
        """Compiles the CNN model with Adam optimizer and sparse categorical crossentropy loss."""
        if self.model is None:
            self.build_model()

        print(f"[CNN] Compiling CNN with Adam(lr={self.learning_rate}) and Sparse Categorical Crossentropy...")
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        self.model.summary()

    def train(self, validation_split: float = 0.2) -> keras.callbacks.History:
        """Trains the CNN on normalized image tensors."""
        if self.model is None:
            self.compile_model()

        print(f"[CNN] Training CNN for {self.epochs} epochs with batch size {self.batch_size}...")
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
        print(f"[CNN] Model training completed in {self.training_time:.2f} seconds.")
        return self.history

    def evaluate(self) -> Tuple[float, float]:
        """Evaluates CNN generalization performance on held-out test data."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model must be built and trained before evaluation.")

        print("[CNN] Evaluating model performance on held-out test data...")
        self.test_loss, self.test_accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"[CNN] Test Loss: {self.test_loss:.4f} | Test Accuracy: {self.test_accuracy * 100:.2f}%")
        return self.test_loss, self.test_accuracy

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """Generates class predictions for image samples."""
        if self.model is None:
            raise ValueError("Model is not initialized or trained.")
        probabilities = self.model.predict(samples, verbose=0)
        return np.argmax(probabilities, axis=1)

    def plot_training_history(self, save_filename: str = "cnn_training_history.png") -> str:
        """Generates and saves high-resolution Loss and Accuracy curves for CNN."""
        if self.history is None:
            raise ValueError("No training history found. Model must be trained first.")

        save_path = os.path.join(self.visualizations_dir, save_filename)
        epochs_range = range(1, len(self.history.history["loss"]) + 1)

        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history.history["loss"], "o-", label="Train Loss", color="#1f77b4")
        plt.plot(epochs_range, self.history.history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
        plt.title("CNN - Loss vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Sparse Categorical Crossentropy")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["accuracy"]], "o-", label="Train Accuracy", color="#2ca02c")
        plt.plot(epochs_range, [acc * 100 for acc in self.history.history["val_accuracy"]], "s--", label="Val Accuracy", color="#d62728")
        plt.title("CNN - Accuracy vs. Epochs", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend(frameon=True)
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[CNN] Saved training history curves to: {save_path}")
        return save_path

    def plot_sample_predictions(self, num_samples: int = 10, save_filename: str = "cnn_sample_predictions.png") -> str:
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
            plt.imshow(sample_images[idx].reshape(28, 28), cmap="gray")
            is_correct = sample_true[idx] == sample_preds[idx]
            color = "green" if is_correct else "red"
            plt.title(f"True: {sample_true[idx]} | Pred: {sample_preds[idx]}", color=color, fontweight="bold", fontsize=11)
            plt.axis("off")

        plt.suptitle("CNN (MNIST) - Sample Test Predictions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[CNN] Saved sample predictions visualization to: {save_path}")
        return save_path

    def save_model(self, model_filename: str = "cnn_mnist_model.keras") -> str:
        """Serializes trained CNN model artifact to disk."""
        if self.model is None:
            raise ValueError("Model is not initialized.")

        save_path = os.path.join(self.models_dir, model_filename)
        self.model.save(save_path)
        print(f"[CNN] Serialized model saved to: {save_path}")
        return save_path

    def run(self) -> Dict[str, Any]:
        """Executes the full end-to-end CNN training, evaluation, and artifact generation lifecycle."""
        print("\n" + "=" * 60)
        print(" [STEP 2] CONVOLUTIONAL NEURAL NETWORK (CNN) PIPELINE")
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
            "model_name": "Convolutional Neural Network (CNN)",
            "domain": "Computer Vision (MNIST)",
            "architecture": "Conv2D (32) -> Pool -> Conv2D (64) -> Pool -> Dense (128) -> Dropout (0.5) -> Softmax",
            "parameters": trainable_params,
            "test_accuracy": self.test_accuracy,
            "test_loss": self.test_loss,
            "training_time_sec": self.training_time,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }

        print("\n[CNN] Step 2 Execution Successfully Completed!")
        return results


if __name__ == "__main__":
    cnn_pipeline = ConvolutionalNeuralNetwork()
    cnn_pipeline.run()
