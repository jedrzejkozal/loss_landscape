import numpy as np


class VectorToModelParams(object):

    def __init__(self, params_shapes, params_sizes, is_bias, biases):
        self.params_shapes = params_shapes
        self.params_sizes = params_sizes
        self.is_bias = is_bias
        self.biases = biases

    def to_model_params(self, vector):
        params = []
        self.last_index = 0
        self.i = 0

        for layer_shapes, layer_sizes in zip(self.params_shapes, self.params_sizes):
            layer_params = self.get_layer_params(
                layer_shapes, layer_sizes, vector)
            params.append(layer_params)

        return params

    def get_layer_params(self, layer_shapes, layer_sizes, vector):
        layer_params = []
        for weights_shape, weights_size in zip(layer_shapes, layer_sizes):
            param = self.get_vector_or_bias(
                vector, weights_shape, weights_size)
            layer_params.append(param)
            self.i += 1
        return layer_params

    def get_vector_or_bias(self, vector, weights_shape, weights_size):
        if self.is_bias[self.i]:
            return self.biases[self.i]  # biases are not changed
        else:
            return self.get_reshaped_slice(vector, weights_shape, weights_size)

    def get_reshaped_slice(self, vector, weights_shape, weights_size):
        vector_slice = np.array(
            vector[self.last_index:self.last_index + weights_size])
        reshaped_slice = vector_slice.reshape(weights_shape)
        self.last_index += weights_size
        return reshaped_slice
