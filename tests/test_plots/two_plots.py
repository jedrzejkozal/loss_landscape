from tests.models.conv import conv_model
from datasets import mnist_single_items
from losslandscape import *


def test_two_plots():
    model = conv_model(mnist_single_items())
    _, _, x_test, y_test = mnist_single_items()
    plot_loss(model, ("levels", "3d"), x_test, y_test)
