
def get(optimizer):
    if optimizer == 'classic':
        return ClassicOptimizer()
    else:
        raise NotImplementedError


class Optimizer(object):

    def optimize(self, layers, predicted_y, real_y):
        raise NotImplementedError


class ClassicOptimizer(Optimizer):

    def optimize(self, layers, predicted_y, real_y):
        pass
