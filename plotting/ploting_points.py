import numpy as np

from plotting.calc_loss import *


def get_loss_landscape(model,
                        test_X, test_Y,
                        model_weights,
                        is_bias, biases,
                        params_shapes, params_sizes):
    theta = np.hstack(model_weights)
    del model_weights
    delta_direction = np.random.normal(size=theta.size)
    eta_direction = np.random.normal(size=theta.size)

    normalize_directions(delta_direction, eta_direction, theta, params_shapes, is_bias)

    number_of_points=21 #num=21
    alpha_coor = np.linspace(-1.0, 1.0, num=number_of_points)
    beta_coor = np.linspace(-1.0, 1.0, num=number_of_points)

    lossCalc = LossCalc(model, test_X, test_Y)
    plot_points = []

    for point_to_evaluate in yield_point_to_evaluate(delta_direction, eta_direction,
                                                    alpha_coor, beta_coor,
                                                    params_shapes, params_sizes,
                                                    theta,
                                                    is_bias, biases):
        plot_points.append(lossCalc.calc_loss(point_to_evaluate))

    x, y = np.meshgrid(alpha_coor, beta_coor)
    plot_points = np.array(plot_points).reshape(number_of_points, number_of_points)
    return x, y, plot_points


def normalize_directions(delta_direction, eta_direction, theta, params_shapes, is_bias):
    i = -1
    last_index = 0

    for layer_shapes in params_shapes:
        for weights_shape in layer_shapes:
            i += 1
            if is_bias[i]:
                continue
            else:
                if len(weights_shape) == 4: #is filter with shape: (3, 3, 1, 32)
                    for filter_ in range(weights_shape[3]):
                        num_of_filter_params = weights_shape[0]*weights_shape[1]*weights_shape[2]
                        filter_weights = theta[last_index:last_index+num_of_filter_params]
                        delta_direction[last_index:last_index+num_of_filter_params] = delta_direction[last_index:last_index+num_of_filter_params] / np.linalg.norm(delta_direction[last_index:last_index+num_of_filter_params]) * np.linalg.norm(filter_weights)
                        eta_direction[last_index:last_index+num_of_filter_params] = eta_direction[last_index:last_index+num_of_filter_params] / np.linalg.norm(eta_direction[last_index:last_index+num_of_filter_params]) * np.linalg.norm(filter_weights)
                        last_index += num_of_filter_params
                else:
                    for neuron in range(weights_shape[0]):
                        neuron_weights = theta[last_index:last_index+weights_shape[1]]
                        delta_direction[last_index:last_index+weights_shape[1]] = delta_direction[last_index:last_index+weights_shape[1]] / np.linalg.norm(delta_direction[last_index:last_index+weights_shape[1]]) * np.linalg.norm(neuron_weights)
                        eta_direction[last_index:last_index+weights_shape[1]] = eta_direction[last_index:last_index+weights_shape[1]] / np.linalg.norm(eta_direction[last_index:last_index+weights_shape[1]]) * np.linalg.norm(neuron_weights)
                        last_index += weights_shape[1]


def yield_point_to_evaluate(x_base_vec, y_base_vec,
                            x_scaling_grid, y_scaling_grid,
                            params_shapes, params_sizes,
                            starting_point,
                            is_bias, biases):
    point_index = 0
    all_points = len(x_scaling_grid)*len(y_scaling_grid)
    for x_scaling in x_scaling_grid:
        for y_scaling in y_scaling_grid:
            point_index += 1
            print("evaluating point {}/{}".format(point_index, all_points))
            point_to_evaluate = starting_point + x_scaling*x_base_vec + y_scaling*y_base_vec
            model_params = array_to_model_params(point_to_evaluate,
                                                params_shapes, params_sizes,
                                                is_bias, biases)
            yield model_params


def array_to_model_params(array, params_shapes, params_sizes, is_bias, biases):
    params = []
    last_index = 0
    i = 0

    for layer_shapes, layer_sizes in zip(params_shapes, params_sizes):
        layer_params = []
        for weights_shape, weights_size in zip(layer_shapes, layer_sizes):
            if is_bias[i] == True:
                layer_params.append(biases[i]) #biases are not changed
            else:
                a = np.array(array[last_index:last_index+weights_size]).reshape(weights_shape)
                layer_params.append(a)
                last_index += weights_size
            i += 1
        params.append(layer_params)

    return params
