from tests.models.dense import dense_model
from datasets import toy_dataset
from losslandscape import *


def test_dense_toy_dataset_num_points_3():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss_3D(model, "levels", x_test, y_test, number_of_points=3)


def test_dense_toy_dataset_two_types_num_points_3():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss_3D(model, ("levels", "3d"), x_test, y_test, number_of_points=3)
