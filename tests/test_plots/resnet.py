from tests.models.resnet import resnet_model
from datasets import cifar10_single_items
from losslandscape import *


def test_resnet_cifar10_model():
    model = resnet_model(cifar10_single_items())
    _, _, x_test, y_test = cifar10_single_items()
    plot_loss(model, "levels", x_test, y_test)
