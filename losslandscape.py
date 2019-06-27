from plots import *
from points_evaluation.model_wrapper import *
from points_evaluation.ploting_points import *
from utils.model_params import *


def plot_loss(model, datasets, dataset_labes=None):
    datasets = make_tuple(datasets)
    dataset_labes = make_tuple(dataset_labes)
    if len(datasets) == 2 and type(datasets[0]) is not tuple:
        return [datasets]

    if dataset_labes[0] is not None and len(datasets) != len(dataset_labes):
        raise AssertionError(
            "Datasets and labels length must be the same, got {} and {}".format(len(datasets), len(dataset_labes)))

    plot_all_series(model, datasets, dataset_labes)


def plot_all_series(model, datasets, dataset_labels):
    weights_vec, is_bias, biases, params_shapes, params_sizes = get_model_params(
        model)

    for (x_test, y_test), label in zip(datasets, dataset_labels):
        model_wrapper = ModelWrapper(model, x_test, y_test)
        x, y = get_ploting_points(model_wrapper, weights_vec, is_bias,
                                  biases, params_shapes, params_sizes)
        plot_2d(x, y, figure_index=0, label=label)
    show_all()


def plot_loss_3D(model, plot_types, x_test, y_test):
    plot_types = make_tuple(plot_types)

    model_wrapper = ModelWrapper(model, x_test, y_test)
    weights_vec, is_bias, biases, params_shapes, params_sizes = get_model_params(
        model)

    x, y, z = get_ploting_points_3D(
        model_wrapper, weights_vec, is_bias, biases, params_shapes, params_sizes)

    plot_all(plot_types, x, y, z)


def make_tuple(obj):
    if type(obj) is not tuple and type(obj) is not list:
        return (obj, )
    return obj


def get_model_params(model):
    model_weights, is_bias, biases = get_model_weights(model)
    params_shapes, params_sizes = params_shape_and_size(model)
    weights_vec = weights_as_single_vector(model_weights)
    return weights_vec, is_bias, biases, params_shapes, params_sizes


def weights_as_single_vector(weights):
    weights = list(filter(lambda x: x is not None, weights))
    weights = list(map(lambda x: x.flatten(), weights))
    return np.hstack(weights)


def plot_all(plot_types, x, y, z=None):
    for i, plot_type in enumerate(plot_types):
        plot_points(plot_type, x, y, z, i)

    show_all()


def plot_points(plot_type, x, y, z, index):
    if plot_type == "levels":
        plot_levels(x, y, z, figure_index=index)
    elif plot_type == "3d":
        plot_3d(x, y, z, figure_index=index)
    elif plot_type == "2d":
        plot_2d(x, y, figure_index=index)
    else:
        raise RuntimeError("Unknown plot type")
