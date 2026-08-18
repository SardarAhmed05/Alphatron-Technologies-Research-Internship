# Import
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

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



# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)



# Normalization
X_train = X_train / 255.0
X_test = X_test / 255.0



# CNN expects different shape of data so reshaping
X_train = X_train.reshape(
    X_train.shape[0],
    28,
    28,
    1
)


X_test = X_test.reshape(
    X_test.shape[0],
    28,
    28,
    1
)



print(X_train.shape)

# Output:
# (60000, 28, 28, 1)



# Build CNN model ( Sequential )
model = Sequential()



# 1st Convulation Layer
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(28,28,1)
    )
)



# Pooling Layer
model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)



# 2nd Convulation Layer
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    )
)



# 2nd Pooling Layer
model.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)


# Flatten
model.add(
    Flatten()
)



# Connected Layer
model.add(
    Dense(
        units=128,
        activation="relu"
    )
)



# Dropout
model.add(
    Dropout(
        0.5
    )
)



# Output Layer
model.add(
    Dense(
        units=10,
        activation="softmax"
    )
)



# summary
model.summary()



# compilation
model.compile(
    optimizer=Adam(),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)



# Training
history = model.fit(

    X_train,

    y_train,

    epochs=10,

    batch_size=64,

    validation_split=0.2

)



# Eval
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)


print(
    "Test Accuracy:",
    test_accuracy
)



# Making predictions
prediction = model.predict(
    X_test
)



predicted_label = np.argmax(
    prediction[0]
)



print(
    "Actual:",
    y_test[0]
)


print(
    "Predicted:",
    predicted_label
)



# Show image
plt.imshow(
    X_test[0].reshape(28,28),
    cmap="gray"
)

plt.title(
    f"Actual: {y_test[0]} | Predicted: {predicted_label}"
)

plt.show()