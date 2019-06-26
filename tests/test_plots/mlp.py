from tests.models.dense import dense_model
from datasets import toy_dataset
from losslandscape import *


def test_dense_toy_dataset_model():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss_3D(model, "levels", x_test, y_test)
