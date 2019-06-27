import pytest

from datasets import mnist_single_items, toy_dataset
from losslandscape import *
from tests.models.conv import conv_model
from tests.models.dense import dense_model


def test_dense_toy_dataset_model():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss(model, (x_test, y_test))


def test_unlearned_conv_model():
    model = conv_model(mnist_single_items())
    _, _, x_test, y_test = mnist_single_items()
    plot_loss(model, (x_test, y_test))


def test_dense_toy_dataset_model_multiple_datasets():
    model = dense_model(toy_dataset())
    x_train, y_train, x_test, y_test = toy_dataset()
    plot_loss(model, ((x_train, y_train), (x_test, y_test)))


def test_dense_toy_dataset_model_with_labels():
    model = dense_model(toy_dataset())
    x_train, y_train, x_test, y_test = toy_dataset()
    plot_loss(model, ((x_train, y_train), (x_test, y_test)),
              dataset_labels=('tranining set', 'test set'))


def test_dense_toy_dataset_model_with_missing_labels():
    model = dense_model(toy_dataset())
    x_train, y_train, x_test, y_test = toy_dataset()
    with pytest.raises(AssertionError) as err:
        plot_loss(model, ((x_train, y_train), (x_test, y_test)),
                  dataset_labels=('tranining set',))

    assert err.type is AssertionError
    assert "Datasets and labels length must be the same" in str(err.value)


def test_dense_toy_dataset_number_of_points_41():
    model = dense_model(toy_dataset())
    _, _, x_test, y_test = toy_dataset()
    plot_loss(model, (x_test, y_test), number_of_points=41)
