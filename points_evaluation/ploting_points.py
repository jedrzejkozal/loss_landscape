import numpy as np

from points_evaluation.normal_directions import *
from points_evaluation.vector_to_model_params_converter import *


def get_ploting_points(model_wrapper,
                       theta,
                       is_bias, biases,
                       params_shapes, params_sizes):
    delta_direction, eta_direction = get_normalized_directions(
        theta.size, theta, params_shapes, is_bias)

    number_of_points = 21
    alpha_range = np.linspace(-1.0, 1.0, num=number_of_points)
    beta_ranage = np.linspace(-1.0, 1.0, num=number_of_points)

    x, y = np.meshgrid(alpha_range, beta_ranage)
    points_generator = yield_grid_point(delta_direction, eta_direction,
                                        alpha_range, beta_ranage,
                                        theta)
    converter = VectorToModelParams(
        params_shapes, params_sizes, is_bias, biases)

    z = get_loss_values(model_wrapper, number_of_points,
                        points_generator, converter)

    return x, y, z


def get_loss_values(model_wrapper, number_of_points, points_generator, converter):
    loss = []
    point_index = 0
    num_all_points = number_of_points**2
    for weights_to_evaluate in points_generator:
        point_index += 1
        print("evaluating point {}/{}".format(point_index, num_all_points))
        model_params = converter.to_model_params(weights_to_evaluate)
        loss.append(model_wrapper.calc_loss(model_params))
    return np.array(loss).reshape(number_of_points, number_of_points)


def yield_grid_point(x_base_vec, y_base_vec,
                     x_scaling_grid, y_scaling_grid,
                     starting_point):
    for x_scaling in x_scaling_grid:
        for y_scaling in y_scaling_grid:
            yield starting_point + x_scaling*x_base_vec + y_scaling*y_base_vec
