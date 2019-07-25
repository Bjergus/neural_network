"""
RETURNS OUTPUT OF ACTIVATION FUNCTION AND DERIVATION
"""
e = 2.71828


def get(activation):
    if activation == 'sigmoid':
        return sigmoid
    elif activation == 'relu':
        return relu
    else:
        raise NotImplementedError


def sigmoid(x):
    out = 1 / (1 + e ** (-x))
    derivation = out * (1 - out)

    return out, derivation


def relu(x):
    if x <= 0:
        return 0, 0

    return x, 1