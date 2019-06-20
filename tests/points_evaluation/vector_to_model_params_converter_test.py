import pytest

from tests.datasets import toy_dataset, mnist_dataset
from tests.models.conv import conv_model
from tests.models.dense import dense_model
from points_evaluation.model_params import *
from points_evaluation.ploting_points import *


class TestVectorToModelParams(object):

    def get_sut_and_input(self, model):
        model_weights, is_bias, biases = get_model_weights(model)
        params_shapes, params_sizes = params_shape_and_size(model)
        converter = VectorToModelParams(
            params_shapes, params_sizes, is_bias, biases)
        # print(params_shapes, params_sizes)
        return converter, model_weights, is_bias, biases

    def test_to_model_params_dense_net_values_are_close_to_original(self):
        model = dense_model(toy_dataset())
        sut, model_weights, is_bias, biases = self.get_sut_and_input(model)
        model_weights = list(filter(lambda x: x is not None, model_weights))

        recreated_params = sut.to_model_params(
            np.hstack(model_weights).flatten())
        recreated_params = [
            x for inner_list in recreated_params for x in inner_list]

        for weights_matrix, is_b, bias, recreated_weights in zip(model_weights, is_bias, biases, recreated_params):
            self.is_close(recreated_weights, is_b,
                          bias, weights_matrix)

    def test_to_model_params_conv_net_values_are_close_to_original(self):
        model = conv_model(mnist_dataset())
        sut, model_weights, is_bias, biases = self.get_sut_and_input(model)
        model_weights = list(filter(lambda x: x is not None, model_weights))

        recreated_params = sut.to_model_params(
            np.hstack(model_weights).flatten())
        recreated_params = [
            x for inner_list in recreated_params for x in inner_list]

        for weights_matrix, is_b, bias, recreated_weights in zip(model_weights, is_bias, biases, recreated_params):
            self.is_close(recreated_weights, is_b,
                          bias, weights_matrix)

    def is_close(self, recreated, is_bias, bias_value, weights_matrix):
        if is_bias:
            assert np.isclose(bias_value, recreated.flatten()).all()
        else:
            assert np.isclose(weights_matrix, recreated.flatten()).all()
