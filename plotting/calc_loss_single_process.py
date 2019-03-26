from plotting.timeit_decorator import *


@timeit
def calc_loss_for_test_set(model, params_to_evaluate, test_X, test_Y):
    plot_points = []

    for params_set in params_to_evaluate:
        set_params(model, params_set)
        loss = evaluate_loss(model, test_X, test_Y)
        plot_points.append(loss)

    return plot_points


def set_params(model, params):
    for layer, params_to_set in zip(model.layers, params):
        layer.set_weights(params_to_set)


def evaluate_loss(model, x, y):
    results = model.evaluate(x=x, y=y, verbose=0)

    loss_index = model.metrics_names.index('loss')
    return results[loss_index]
