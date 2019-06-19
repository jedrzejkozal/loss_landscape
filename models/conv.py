from keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import pytest
import numpy as np


def conv_model(dataset):
    train_X, train_Y, test_X, test_Y = dataset
    num_classes = train_Y.shape[1]

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])

    model.fit(train_X, train_Y,
              epochs=1,  # 100,
              batch_size=10,
              verbose=1,
              validation_data=(test_X, test_Y))
    return model
