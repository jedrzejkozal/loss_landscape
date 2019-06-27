from plots import *
from points_evaluation.model_wrapper import *
from points_evaluation.ploting_points import *
from utils.model_params import *


def plot_loss(model, datasets, dataset_labels=None, number_of_points=21):
    datasets = make_tuple(datasets)
    dataset_labels = make_tuple(dataset_labels)
    if len(datasets) == 2 and type(datasets[0]) is not tuple:
        datasets = [datasets]

    if dataset_labels[0] is not None and len(datasets) != len(dataset_labels):
        raise AssertionError(
            "Datasets and labels length must be the same, got {} and {}".format(len(datasets), len(dataset_labels)))

    functions = get_all_functions(
        model, datasets, number_of_points=number_of_points)
    plot_all_functions(model, functions, dataset_labels)


def get_all_functions(model, datasets, number_of_points=21):
    model_params = get_model_params(model)

    functions = []
    for x_test, y_test in datasets:
        x, y = loss_value_around_single_point(
            model, x_test, y_test, model_params, number_of_points=number_of_points)
        functions.append((x, y))
    return functions


def loss_value_around_single_point(model, x_test, y_test, model_params, number_of_points=21):
    model_wrapper = ModelWrapper(model, x_test, y_test)
    weights_vec, is_bias, biases, params_shapes, params_sizes = model_params
    return get_ploting_points(model_wrapper, weights_vec, is_bias,
                              biases, params_shapes, params_sizes, number_of_points=21)


def plot_all_functions(model, functions, dataset_labels):
    for (x, y), label in zip(functions, dataset_labels):
        plot_2d(x, y, figure_index=0, label=label)
    show_all()


def plot_loss_3D(model, plot_types, x_test, y_test, number_of_points=21):
    plot_types = make_tuple(plot_types)

    model_wrapper = ModelWrapper(model, x_test, y_test)
    weights_vec, is_bias, biases, params_shapes, params_sizes = get_model_params(
        model)

    x, y, z = get_ploting_points_3D(
        model_wrapper, weights_vec, is_bias, biases, params_shapes, params_sizes, number_of_points=21)

    plot_all_types(plot_types, x, y, z)


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


def plot_all_types(plot_types, x, y, z=None):
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
