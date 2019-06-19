from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np


def normalize(vector):
    return (vector - np.mean(vector, axis=0)) / (np.std(vector, axis=0))


def toy_dataset():
    x_train = np.random.random((100, 10))
    y_train = np.random.random((100, 1))
    x_test = np.random.random((100, 10))
    y_test = np.random.random((100, 1))

    x_train = normalize(x_train)
    y_train = normalize(y_train)
    x_test = normalize(x_test)
    y_test = normalize(y_test)

    return x_train, y_train, x_test, y_test


def mnist_dataset():
    img_rows, img_cols = 28, 28
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test
