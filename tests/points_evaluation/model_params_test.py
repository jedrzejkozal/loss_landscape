import numpy as np
import pytest

from tests.datasets import toy_dataset, mnist_dataset
from tests.models.conv import conv_model
from tests.models.dense import dense_model
from points_evaluation.model_params import *


class model_params_test(object):

    @classmethod
    def setup(cls):
        cls.dense = dense_model(toy_dataset())
        cls.conv = conv_model(mnist_dataset())

    def test_get_model_weights_dense_all_weights_are_close_to_original(self):
        model = self.dense
        read_weights, is_bias, bias_values = get_model_weights(model)

        i = 0
        for l in model.layers:
            for w in l.get_weights():
                is_close(self, w, is_bias[i], bias_values[i], read_weights[i])
                i += 1

    def test_get_model_weights_conv_all_weights_are_close_to_original(self):
        model = self.conv
        read_weights, is_bias, bias_values = get_model_weights(model)

        i = 0
        for l in model.layers:
            for w in l.get_weights():
                is_close(self, w, is_bias[i], bias_values[i], read_weights[i])
                i += 1

    def is_close(self, w, is_bias, bias_value, read_weights):
        if is_bias:
            assert np.isclose(bias_value, w).all()
        else:
            assert np.isclose(read_weights, w.flatten()).all()

    def test_get_model_weights_dense_all_results_have_same_size(self):
        model = self.dense
        read_weights, is_bias, bias_values = get_model_weights(model)
        assert len(read_weights) == len(is_bias)
        assert len(is_bias) == len(bias_values)

    def test_get_model_weights_conv_all_results_have_same_size(self):
        model = self.conv
        read_weights, is_bias, bias_values = get_model_weights(model)
        assert len(read_weights) == len(is_bias)
        assert len(is_bias) == len(bias_values)
