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
        self.last_full_out = []
        self.last_input = []

    def set_weight(self, previous_node, current_node, weight):
        self.weights[previous_node][current_node] = weight

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

    def get_last_full_out(self):
        raise NotImplementedError

    def get_last_input(self):
        raise NotImplementedError


class Dense(Layer):

    def __init__(self, units, activation):
        super(Dense, self).__init__(units=units, activation=activations.get(activation))

    def get_output(self, input):
        output = forwardpass.get_layer_output(inputs=input, weights=self.weights, activation=self.activation)
        self.last_full_out = output
        self.last_input = input
        return output[:, 0]     # returns net outputs of layer

    def get_last_full_out(self):
        return self.last_full_out

    def get_last_input(self):
        return self.last_input


class Input(Layer):

    def __init__(self, units, activation):
        super(Input, self).__init__(units=units, activation=activations.get(activation))

    def get_output(self, input):
        out = []
        for i in range(0, self.units):
            output, _ = self.activation(input[i])
            out.append(output)
        return out
