import pytest

from plotting.ploting_points import *
from models.dense import dense_model
from datasets import toy_dataset


def test_recreate_model_params():
    model = dense_model(toy_dataset())
    read_params = get_model_params(model)

    params_shapes, params_sizes = params_shape_and_size(model)

    recreated_params = array_to_model_params(np.hstack(read_params), params_shapes, params_sizes)
    recreated_params = [x for inner_list in recreated_params for x in inner_list]

    for read_weights, recreated_weights in zip(read_params, recreated_params):
        assert np.isclose(read_weights.all(), recreated_weights.flatten().all())
