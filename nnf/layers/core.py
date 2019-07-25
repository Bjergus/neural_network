from nnf import activations
from nnf.backend import forwardpass


class Layer(object):

    def __init__(self, **kwargs):
        """
        weights: list of weight connected this layer with l-1
        :param kwargs:
            activation = activation function
            units = number of units
        """
        self.weights = []
        self.activation = kwargs.get('activation')
        self.units = kwargs.get('units')

    def set_weight(self, index, weight):
        self.weights[index] = weight

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_output(self, input):
        """
        :param input: list of values of l-1
        :return: output for each node of layer
        """
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, units, activation):
        super(Dense, self).__init__(units=units, activation=activations.get(activation))

    def get_output(self, input):
        return forwardpass.get_layer_output(inputs=input, weights=self.weights, activation=self.activation)


class Input(Layer):

    def __init__(self, units, activation):
        super(Input, self).__init__(units=units, activation=activations.get(activation))

    def get_output(self, input):
        out = []
        for i in range(0, self.units):
            output, _ = self.activation(input[i])
            out.append(output)
        return out
