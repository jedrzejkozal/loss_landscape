import numpy as np


class VectorToModelParams(object):

    def __init__(self, params_shapes, params_sizes, is_bias, biases):
        self.params_shapes = params_shapes
        self.params_sizes = params_sizes
        self.is_bias = is_bias
        self.biases = biases

    def to_model_params(self, array):
        params = []
        last_index = 0
        i = 0

        for layer_shapes, layer_sizes in zip(self.params_shapes, self.params_sizes):
            layer_params = []
            for weights_shape, weights_size in zip(layer_shapes, layer_sizes):
                if self.is_bias[i]:
                    # biases are not changed
                    layer_params.append(self.biases[i])
                else:
                    a = np.array(array[last_index:last_index +
                                       weights_size]).reshape(weights_shape)
                    layer_params.append(a)
                    last_index += weights_size
                i += 1
            params.append(layer_params)

        return params
