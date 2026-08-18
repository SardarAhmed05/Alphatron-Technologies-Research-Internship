# RNN Implementation with IMDB dataset
import tensorflow as tf

from tensorflow.keras.datasets import imdb

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    SimpleRNN,
    Dense
)

from tensorflow.keras.preprocessing.sequence import pad_sequences



# load dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)


# padding
X_train = pad_sequences(
    X_train,
    maxlen=200
)

X_test = pad_sequences(
    X_test,
    maxlen=200
)



# Building an RNN model
model = Sequential()



# Convert words into vectors
model.add(
    Embedding(
        input_dim=10000,
        output_dim=32,
        input_length=200
    )
)



# RNN Layer
model.add(
    SimpleRNN(
        units=64
    )
)



# Output Layer
model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)



# compilation
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)



# training
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)



# eval
loss, accuracy = model.evaluate(
    X_test,
    y_test
)


print(
    "Test Accuracy:",
    accuracy
)