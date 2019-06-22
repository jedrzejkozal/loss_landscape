import numpy as np


def get_normalized_directions(vector_size, theta, params_shapes, is_bias):
    delta_direction = np.random.normal(size=vector_size)
    eta_direction = np.random.normal(size=vector_size)

    normalize_direction(delta_direction, theta, params_shapes, is_bias)
    normalize_direction(eta_direction, theta, params_shapes, is_bias)
    return delta_direction, eta_direction


def normalize_direction(direction, theta, params_shapes, is_bias):
    assert direction.size == theta.size, "normalized vector and theta shuld have same size, got {} and {}".format(
        direction.size, theta.size)
    i = -1
    last_index = 0

    for layer_shapes in params_shapes:
        for weights_shape in layer_shapes:
            i += 1
            if is_bias[i]:
                continue
            else:
                direction, last_index = normalize_single_unit(
                    direction, weights_shape, last_index, theta)
    assert last_index == len(direction)


def normalize_single_unit(direction, weights_shape, last_index, theta):
    if is_filter(weights_shape):  # is filter with shape like: (3, 3, 1, 32)
        params_per_filter = weights_shape[0] * \
            weights_shape[1]*weights_shape[2]
        direction, last_index = normalize_all_filters(
            weights_shape[3], direction, last_index, params_per_filter, theta)
    else:
        params_per_neuron = weights_shape[1]
        direction, last_index = normalize_all_neurons(
            weights_shape[0], direction, last_index, params_per_neuron, theta)
    return direction, last_index


def is_filter(weights_shape):
    return len(weights_shape) == 4


def normalize_all_filters(num_of_filters, direction, last_index, params_per_filter, theta):
    for filter_ in range(num_of_filters):
        filter_weights, slice_index = get_vector_slice(
            theta, last_index, params_per_filter)
        direction[slice_index] = change_vector_norm(
            direction[slice_index], np.linalg.norm(filter_weights))
        last_index += params_per_filter
    return direction, last_index


def normalize_all_neurons(num_of_neurons, direction, last_index, params_per_neuron, theta):
    for neuron in range(num_of_neurons):
        neuron_weights, slice_index = get_vector_slice(
            theta, last_index, params_per_neuron)
        direction[slice_index] = change_vector_norm(
            direction[slice_index], np.linalg.norm(neuron_weights))
        last_index += params_per_neuron
    return direction, last_index


def get_vector_slice(vec, start, offset):
    return vec[start:start+offset+1], slice(start, start+offset+1, 1)


def change_vector_norm(vec, norm):
    vec = normalize_vector(vec)
    return vec * norm


def normalize_vector(vec):
    return vec / np.linalg.norm(vec)
