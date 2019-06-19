import numpy as np

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


class LossCalc(object):

    def __init__(self, model, test_X, test_Y):
        self.model = model
        self.test_X = test_X
        self.test_Y = test_Y

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
        results = self.model.evaluate(x=self.test_X, y=self.test_Y, verbose=0)

        loss_index = self.model.metrics_names.index('loss')
        return results[loss_index]
