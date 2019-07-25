import random

max_weight_value = 0.3
min_weight_value = 0.05


def get_initial_weights(unit_before, current_units):
    """
    :param unit_before:
    :param current_units:
    :return: 2D array of weight. array[previous_l_node][current_l_node]
    """
    return [[random.uniform(min_weight_value, max_weight_value) for j in range(0, current_units)] for i in range(0, unit_before)]