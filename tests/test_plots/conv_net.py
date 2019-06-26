from tests.models.conv import conv_model
from datasets import mnist_dataset
from losslandscape import *


def test_conv_mnist_model():
    model = conv_model(mnist_dataset())
    _, _, x_test, y_test = mnist_dataset()
    plot_loss_3D(model, "levels", x_test, y_test)
