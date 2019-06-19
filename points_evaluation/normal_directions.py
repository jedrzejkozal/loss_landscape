import numpy as np


def get_normalized_directions(vector_size, theta, params_shapes, is_bias):
    delta_direction = np.random.normal(size=vector_size)
    eta_direction = np.random.normal(size=vector_size)

    normalize_directions(delta_direction, eta_direction,
                         theta, params_shapes, is_bias)
    return delta_direction, eta_direction


def normalize_directions(delta_direction, eta_direction, theta, params_shapes, is_bias):
    i = -1
    last_index = 0

    for layer_shapes in params_shapes:
        for weights_shape in layer_shapes:
            i += 1
            if is_bias[i]:
                continue
            else:
                if len(weights_shape) == 4:  # is filter with shape: (3, 3, 1, 32)
                    for filter_ in range(weights_shape[3]):
                        num_of_filter_params = weights_shape[0] * \
                            weights_shape[1]*weights_shape[2]
                        filter_weights = theta[last_index:last_index +
                                               num_of_filter_params]
                        delta_direction[last_index:last_index+num_of_filter_params] = delta_direction[last_index:last_index+num_of_filter_params] / \
                            np.linalg.norm(
                                delta_direction[last_index:last_index+num_of_filter_params]) * np.linalg.norm(filter_weights)
                        eta_direction[last_index:last_index+num_of_filter_params] = eta_direction[last_index:last_index+num_of_filter_params] / \
                            np.linalg.norm(
                                eta_direction[last_index:last_index+num_of_filter_params]) * np.linalg.norm(filter_weights)
                        last_index += num_of_filter_params
                else:
                    for neuron in range(weights_shape[0]):
                        neuron_weights = theta[last_index:last_index +
                                               weights_shape[1]]
                        delta_direction[last_index:last_index+weights_shape[1]] = delta_direction[last_index:last_index+weights_shape[1]
                                                                                                  ] / np.linalg.norm(delta_direction[last_index:last_index+weights_shape[1]]) * np.linalg.norm(neuron_weights)
                        eta_direction[last_index:last_index+weights_shape[1]] = eta_direction[last_index:last_index+weights_shape[1]
                                                                                              ] / np.linalg.norm(eta_direction[last_index:last_index+weights_shape[1]]) * np.linalg.norm(neuron_weights)
                        last_index += weights_shape[1]
