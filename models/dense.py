from keras import Sequential
from keras.layers import Dense

import pytest


def dense_model(dataset):
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(optimizer='Adam', loss='mse', metrics=['acc'])

    train_X, train_Y, test_X, test_Y = dataset
    model.fit(train_X, train_Y,
              epochs=100,
              batch_size=20,
              verbose=1,
              validation_data=(test_X, test_Y))
    return model
