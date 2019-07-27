from nnf.backend.weights import get_initializer
from nnf.backend.optimizers import get_optimizer
from nnf.exception import BadShapeException


class NeuralNetwork(object):

    def __init__(self, layers, initializer='standard', optimizer='standard'):
        self.layers = layers
        self.weight_initializer = get_initializer(name=initializer)
        self.optimizer = get_optimizer(name=optimizer)
        self.__init_weights()

    def train(self, x, y, num_epochs=100):
        if len(x) != len(y):
            raise BadShapeException()

        for epoch in range(0, num_epochs):
            for i in range(0, len(x)):
                output = self.__feed_forward(x[i])
                self.optimizer.optimize(self.layers, output, y[i])

    def __feed_forward(self, inputs):
        output = inputs

        for i in range(0, len(self.layers)):
            output = self.layers[i].feed_forward()

        return output

    def __init_weights(self):
        for i in range(1, len(self.layers)):
            previous_nodes = len(self.layers[i - 1].nodes)
            for node_index in range(0, len(self.layers[i].nodes)):
                node = self.layers[i].nodes[node_index]
                node.weights = self.weight_initializer(previous_nodes)

    def inspect(self):
        print('Layers: ' + str(len(self.layers)))
        for i in range(0, len(self.layers)):
            print(self.layers[i].inspect)

