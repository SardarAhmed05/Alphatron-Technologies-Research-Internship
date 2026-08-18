# Import
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Load dataset ( MNIST )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Dataset visualization
plt.imshow(X_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()


# ==========================================
# 3. Normalize Data
# ==========================================

# Pixel values are between 0-255
# Convert them to 0-1 range

X_train = X_train / 255.0
X_test = X_test / 255.0

# Initializing the Sequential model ( ANN )
model = Sequential()

# Input Layer and First Hidden Layer
model.add(
    Flatten(input_shape=(28,28))
)

model.add(
    Dense(
        units=128,
        activation="relu"
    )
)

# Second Hidden Layer

model.add(
    Dense(
        units=64,
        activation="relu"
    )
)


# Output Layer

model.add(
    Dense(
        units=10,
        activation="softmax"
    )
)


# Model summary

model.summary()

# Compiling
model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Training step

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Testing and evaluation
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", test_accuracy)

# Making predictions
prediction = model.predict(X_test)

predicted_label = np.argmax(prediction[0])


print("Actual Label:", y_test[0])
print("Predicted Label:", predicted_label)

# Predicted Image
plt.imshow(X_test[0], cmap="gray")
plt.title(
    f"Actual: {y_test[0]}, Predicted: {predicted_label}"
)

plt.show()