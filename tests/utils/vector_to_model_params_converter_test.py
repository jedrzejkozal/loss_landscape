import pytest

from tests.datasets import *
from tests.models.conv import conv_model
from tests.models.dense import dense_model
from utils.model_params import *
#from points_evaluation.ploting_points import *
from utils.vector_to_model_params_converter import *


class TestVectorToModelParams(object):

    def get_sut_and_test_input(self, model):
        model_weights, is_bias, biases = get_model_weights(model)
        params_shapes, params_sizes = params_shape_and_size(model)
        converter = VectorToModelParams(
            params_shapes, params_sizes, is_bias, biases)
        return converter, model_weights, is_bias, biases

    def test_to_model_params_dense_net_values_are_close_to_original(self):
        model = dense_model(toy_dataset())
        sut, model_weights, is_bias, biases = self.get_sut_and_test_input(
            model)
        input = self.weights_as_single_vector(model_weights)

        recreated_params = sut.to_model_params(input)

        recreated_params = self.unwrap_list(recreated_params)
        for weights_matrix, is_b, bias, recreated_weights in zip(model_weights, is_bias, biases, recreated_params):
            self.is_close(recreated_weights, is_b,
                          bias, weights_matrix)

    def test_to_model_params_conv_net_values_are_close_to_original(self):
        model = conv_model(mnist_single_items())
        sut, model_weights, is_bias, biases = self.get_sut_and_test_input(
            model)
        input = self.weights_as_single_vector(model_weights)

        recreated_params = sut.to_model_params(input)

        recreated_params = self.unwrap_list(recreated_params)
        for weights_matrix, is_b, bias, recreated_weights in zip(model_weights, is_bias, biases, recreated_params):
            self.is_close(recreated_weights, is_b,
                          bias, weights_matrix)

    def weights_as_single_vector(self, weights):
        weights = list(filter(lambda x: x is not None, weights))
        weights = list(map(lambda x: x.flatten(), weights))
        return np.hstack(weights)

    def unwrap_list(self, l):
        return [x for inner_list in l for x in inner_list]

    def is_close(self, recreated, is_bias, bias_value, weights_matrix):
        if is_bias:
            assert np.isclose(recreated.flatten(), bias_value).all()
        else:
            assert np.isclose(recreated.flatten(), weights_matrix).all()
