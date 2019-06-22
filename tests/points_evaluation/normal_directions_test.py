from functools import reduce

import numpy as np
import pytest

from points_evaluation.normal_directions import *


class BaseTest(object):

    def get_config(self):
        raise NotImplementedError

    def test_vectors_after_normalization_have_same_size_as_before(self):
        direction, theta, params_shapes, is_bias = self.get_config()
        direction_shape_before = direction.shape

        normalize_direction(direction, theta, params_shapes, is_bias)

        assert direction.shape == direction_shape_before

    def test_vectors_after_normalization_have_same_type_as_before(self):
        direction, theta, params_shapes, is_bias = self.get_config()
        direction_type_before = direction.dtype

        normalize_direction(direction, theta, params_shapes, is_bias)

        assert direction.dtype == direction_type_before

    def test_normalization_against_same_vector_does_nothing(self):
        direction, theta, params_shapes, is_bias = self.get_config()
        theta = np.copy(direction)
        direction_before = np.copy(direction)

        normalize_direction(direction, theta, params_shapes, is_bias)

        assert np.isclose(direction, direction_before).all()

    def test_diffrent_theta_and_direction_size_exception_raised(self):
        direction, theta, params_shapes, is_bias = self.get_config()
        theta = theta[:-2]

        with pytest.raises(AssertionError) as err:
            normalize_direction(direction, theta, params_shapes, is_bias)

        assert err.type is AssertionError
        assert "normalized vector and theta shuld have same size" in str(
            err.value)


class TestOneLayerMLP(BaseTest):

    def get_config(self):
        direction = np.random.randn(4)
        theta = np.random.rand(4)
        params_shapes = [[(2, 2), (2,)]]
        is_bias = [False, True]
        return direction, theta, params_shapes, is_bias


class TestTwoLayerMLP(BaseTest):

    def get_config(self):
        direction = np.random.randn(8)
        theta = np.random.rand(8)
        params_shapes = [[(2, 2), (2,)], [(2, 2), (2,)]]
        is_bias = [False, True, False, True]
        return direction, theta, params_shapes, is_bias


class TestOneLayerConv(BaseTest):

    def get_config(self):
        direction = np.random.randn(45)
        theta = np.random.rand(45)
        params_shapes = [[(3, 3, 1, 5), (2,)]]
        is_bias = [False, True]
        return direction, theta, params_shapes, is_bias


class TestTwoLayerConv(BaseTest):

    def get_config(self):
        direction = np.random.randn(90)
        theta = np.random.rand(90)
        params_shapes = [[(3, 3, 1, 5), (2,)], [(3, 3, 1, 5), (2,)]]
        is_bias = [False, True, False, True]
        return direction, theta, params_shapes, is_bias
