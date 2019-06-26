from tests.models.dense import dense_model
from tests.models.conv import conv_model
from datasets import mnist_single_items, toy_dataset
from losslandscape import *


def test_dense_toy_dataset_model():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss(model, x_test, y_test)


def test_unlearned_conv_model():
    model = conv_model(mnist_single_items())
    _, _, x_test, y_test = mnist_single_items()
    plot_loss(model, x_test, y_test)
