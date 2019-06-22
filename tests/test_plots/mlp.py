from tests.models.dense import dense_model
from datasets import toy_dataset
from losslandscape import *


def test_conv_mnist_model():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss(model, "levels", x_test, y_test)
