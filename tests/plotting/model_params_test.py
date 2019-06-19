import numpy as np
import pytest

from datasets import toy_dataset, mnist_dataset
from models.conv import conv_model
from models.dense import dense_model
from plotting.model_params import *


def test_get_model_weights_dense_all_weights_are_close_to_original():
    model = dense_model(toy_dataset())
    read_weights, is_bias, bias_values = get_model_weights(model)
    print(len(is_bias))

    i = 0
    for l in model.layers:
        for w in l.get_weights():
            is_close(w, is_bias[i], bias_values[i], read_weights[i])
            i += 1


def test_get_model_weights_conv_all_weights_are_close_to_original():
    model = conv_model(mnist_dataset())
    read_weights, is_bias, bias_values = get_model_weights(model)
    print(len(is_bias))

    i = 0
    for l in model.layers:
        for w in l.get_weights():
            is_close(w, is_bias[i], bias_values[i], read_weights[i])
            i += 1


def is_close(w, is_bias, bias_value, read_weights):
    if is_bias:
        assert np.isclose(bias_value, w).all()
    else:
        assert np.isclose(read_weights, w.flatten()).all()
