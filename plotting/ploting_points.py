from plotting.calc_loss_single_process import *
from plotting.timeit_decorator import *


import numpy as np

#unormalized plotting poc:
def get_ploting_points(model, test_X, test_Y):
    theta_begin = np.hstack(get_model_params(model))
    delta_direction = np.random.normal(size=theta_begin.size)
    eta_direction = np.random.normal(size=theta_begin.size)

    alpha_coor = np.linspace(-1.0, 1.0, num=21)
    beta_coor = np.linspace(-1.0, 1.0, num=21)

    params_shapes, params_sizes = params_shape_and_size(model)
    params_to_evaluate = calc_params_to_evaluate(delta_direction, eta_direction,
                alpha_coor, beta_coor,
                params_shapes, params_sizes,
                theta_begin)
    print("caluclataring params_to_evaluate finished")
    plot_points = calc_loss_for_test_set(model, params_to_evaluate, test_X, test_Y)

    x, y = np.meshgrid(alpha_coor, beta_coor)
    return x, y, np.array(plot_points).reshape(21, 21)



def get_model_params(model):
    params = []
    for layer in model.layers:
        for w in layer.get_weights():
            params.append(w.flatten())
    return params


def params_shape_and_size(model):
    shape_selector = lambda w: w.shape
    size_selector = lambda w: w.size
    params_shapes = get_model_quantity(model, shape_selector)
    params_sizes = get_model_quantity(model, size_selector)

    return params_shapes, params_sizes


def get_model_quantity(model, selector):
    result = []
    for layer in model.layers:
        result.append(get_layer_quantity(layer, selector))
    return result


def get_layer_quantity(layer, selector):
    layer_result = []
    for w in layer.get_weights():
        layer_result.append(selector(w))
    return layer_result

@timeit
def calc_params_to_evaluate(x_base_vec, y_base_vec,
            x_scaling_grid, y_scaling_grid,
            params_shapes, params_sizes,
            starting_point):
    params_to_evaluate = []

    for x_scaling in x_scaling_grid:
        for y_scaling in y_scaling_grid:
            point_to_evaluate = starting_point + x_scaling*x_base_vec + y_scaling*y_base_vec
            model_params = array_to_model_params(point_to_evaluate,
                    params_shapes, params_sizes)
            params_to_evaluate.append(model_params)

    return params_to_evaluate


def array_to_model_params(array, params_shapes, params_sizes):
    params = []
    last_index = 0

    for layer_shapes, layer_sizes in zip(params_shapes, params_sizes):
        layer_params = []
        for weights_shape, weights_size in zip(layer_shapes, layer_sizes):
            a = np.array(array[last_index:last_index+weights_size]).reshape(weights_shape)
            layer_params.append(a)
            last_index += weights_size
        params.append(layer_params)

    return params
