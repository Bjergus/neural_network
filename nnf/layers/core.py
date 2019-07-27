from __future__ import absolute_import
from .structure import Neuron
from nnf.exception import BadShapeException
from nnf import activations


class Layer(object):

    def feed_forward(self, inputs):
        raise NotImplementedError

    def inspect(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError


class Input(Layer):

    def __init__(self, units):
        self.units = units
        self.output = []
        self.nodes = [None for i in range(0, units)]

    def feed_forward(self, inputs):
        if len(inputs) != self.units:
            raise BadShapeException()
        self.output = inputs
        return inputs

    def inspect(self):
        print('Input layer: '+ str(self.units) + ' neurons')

    def get_output(self):
        return self.output


class Dense(Layer):

    def __init__(self, units, activation):
        self.activation = activations.get(activation)
        self.nodes = []
        for i in range(0, units):
            self.nodes.append(Neuron(activation=self.activation))

    def feed_forward(self, inputs):
        output = []

        for i in range(0, len(self.nodes)):
            output.append(self.nodes[i].calculate_output(inputs))

        return output

    def get_output(self):
        """
        :return: last output of neural layer
        """
        output = []

        for i in range(0, len(self.nodes)):
            output.append(self.nodes[i].output)

        return output

    def inspect(self, print__weights=True):
        print('Neurons: ' + str(len(self.nodes)))
        print('Activation function: ' + self.activation.__name__)
        if print__weights:
            print('----------------')
            for i in range(0, len(self.nodes)):
                print('Neuron (' + str(i) + ')')
                for w in range(0, len(self.nodes[i].weights)):
                    print('Weight (' + str(w) + '): ' + str(self.nodes[i].weights[w]))
        print('+++++++++++++++++++++++++')

