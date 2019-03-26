import numpy as np


def normalize(x):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0))


def toy_dataset():
    train_X = np.random.random((100,10))
    train_Y = np.random.random((100,1))
    test_X = np.random.random((100,10))
    test_Y = np.random.random((100,1))

    train_X = normalize(train_X)
    train_Y = normalize(train_Y)
    test_X = normalize(test_X)
    test_Y = normalize(test_Y)

    return train_X, train_Y, test_X, test_Y


from keras.datasets import mnist
from keras.utils import to_categorical


def mnist_dataset():
    img_rows, img_cols = 28, 28
    num_classes = 10
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()

    train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
    test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)


    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X /= 255
    test_X /= 255

    train_Y = to_categorical(train_Y, num_classes)
    test_Y = to_categorical(test_Y, num_classes)

    return train_X, train_Y, test_X, test_Y
