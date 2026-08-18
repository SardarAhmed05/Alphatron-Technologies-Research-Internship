import tensorflow as tf

from tensorflow.keras.datasets import imdb

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense
)

from tensorflow.keras.preprocessing.sequence import pad_sequences



# Load dataset

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)



# Make all reviews same length

X_train = pad_sequences(
    X_train,
    maxlen=200
)

X_test = pad_sequences(
    X_test,
    maxlen=200
)



# Build Model

model = Sequential()



# Convert words into vectors

model.add(
    Embedding(
        input_dim=10000,
        output_dim=64,
        input_length=200
    )
)



# LSTM Layer

model.add(
    LSTM(
        units=64
    )
)



# Output

model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



# Compile

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# Train

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)



# Test

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("Accuracy:", accuracy)