
def get_model_weights(model):
    weights, is_bias_list, bias_values = [], [], []
    for w in model.get_weights():
        add_weight(w, weights, is_bias_list, bias_values)
    return weights, is_bias_list, bias_values


def add_weight(w, weights, is_bias_list, bias_values):
    if is_bias(w):
        weights.append(None)
        is_bias_list.append(True)
        bias_values.append(w)
    else:
        weights.append(w.flatten())
        is_bias_list.append(False)
        bias_values.append(None)


def is_bias(weights):
    return len(weights.shape) == 1


def params_shape_and_size(model):
    return params_shape(model), params_size(model)


def params_shape(model):
    def shape_selector(w): return w.shape
    params_shapes = get_model_quantity(model, shape_selector)
    return params_shapes


def params_size(model):
    def size_selector(w): return w.size
    params_sizes = get_model_quantity(model, size_selector)
    return params_sizes


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
