import random


def get_initializer(name):
    if name == 'standard':
        return standard_weight_initializer
    else:
        raise NotImplementedError


def standard_weight_initializer(num):
    weights = []

    for i in range(0, num):
        weights.append(random.uniform(0.01, 0.3))

    return weights
