
def get_model_weights(model):
    params = []
    is_bias = []
    biases = []
    for layer in model.layers:
        print("layer: ", layer.name)
        for w in layer.get_weights():
            print("weights shape: ", w.shape)
            if it_is_bias(w):
                print("bias matix, skipping")
                is_bias.append(True)
                biases.append(w)
            else:
                params.append(w.flatten())
                is_bias.append(False)
                biases.append(None)
    return params, is_bias, biases


def it_is_bias(weights):
    return len(weights.shape) == 1

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
