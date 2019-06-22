from plots import *
from points_evaluation.model_params import *
from points_evaluation.model_wrapper import *
from points_evaluation.ploting_points import *


def plot_loss(model, plot_type, x_test, y_test):
    model_wrapper = ModelWrapper(model, x_test, y_test)
    model_weights, is_bias, biases = get_model_weights(model)
    params_shapes, params_sizes = params_shape_and_size(model)
    weights_vec = weights_as_single_vector(model_weights)

    x, y, z = get_ploting_points(
        model_wrapper, weights_vec, is_bias, biases, params_shapes, params_sizes)

    plot_points(plot_type, x, y, z)

def weights_as_single_vector(weights):
    weights = list(filter(lambda x: x is not None, weights))
    weights = list(map(lambda x: x.flatten(), weights))
    return np.hstack(weights)

def plot_points(plot_type, x, y, z):
    if plot_type == "levels":
        plot_levels(x, y, z)
    elif plot_type == "3d":
        plot_3d(x, y, z)
    else:
        raise RuntimeError("Unknown plot type")
