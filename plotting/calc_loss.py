import numpy as np

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


class LossCalc(object):

    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def calc_loss_for_grid(self, params_to_evaluate):
        plot_points = list(map(self.calc_loss, params_to_evaluate))
        return np.array(plot_points)

    def calc_loss(self, params_set):
        self.set_params(params_set)
        return self.evaluate_loss()

    def set_params(self, params):
        for layer, params_to_set in zip(self.model.layers, params):
            layer.set_weights(params_to_set)

    def evaluate_loss(self):
        results = self.model.evaluate(x=self.x_test, y=self.y_test, verbose=0)

        loss_index = self.model.metrics_names.index('loss')
        return results[loss_index]
