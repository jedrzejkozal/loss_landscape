import numpy as np
from keras import Sequential
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical


def resnet_model(dataset):
    x_train, y_train, x_test, y_test = dataset
    num_classes = y_train.shape[1]

    model = Sequential()
    model.add(ResNet50(include_top=False, input_shape=(
        32, 32, 3), weights='imagenet'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['categorical_accuracy'])

    model.fit(x_train, y_train,
              epochs=1,
              batch_size=32,
              verbose=1,
              validation_data=(x_test, y_test))
    return model
