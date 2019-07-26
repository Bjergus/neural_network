from nnf.backend.weights import get_initial_weights
from nnf.backend import optimizers
from nnf.exception import BadShapeException


class NeuralNetwork(object):

    def __init__(self, layers):
        """
        :param layers: array of layers for neural network
        """
        self.layers = layers
        self.num_epochs = 0
        self.optimizer = None
        self.learning_rate = 0.001
        self.__init_default_weights()

    def __init_default_weights(self):
        input_size = self.layers[0].units
        self.layers[1].set_weights(get_initial_weights(input_size, self.layers[1].units))
        for i in range(2, len(self.layers)):
            self.layers[i].set_weights(get_initial_weights(self.layers[i - 1].units, self.layers[i].units))

    def compile(self, **kwargs):
        """
        :param kwargs: num_epochs
                       learning_rate
                       optimizer - 'classic'
        :return:
        """
        self.learning_rate = kwargs.get('learning_rate')
        self.optimizer = optimizers.get(kwargs.get('optimizer'))
        self.num_epochs = kwargs.get('num_epochs')

    def train(self, x, y):
        if self.__is_shape_valid(x) is not True:
            raise BadShapeException('Shape of training data is not OK')

        for epoch in range(0, self.num_epochs):
            print('Training epochs: ' + str(epoch + 1) + ' of ' + str(self.num_epochs))
            for input_index in range(0, len(x)):
                layer_output = self.layers[0].get_output(x[input_index])    # calculate first layer
                for layer_index in range(1, len(self.layers) - 1):  # calculate hidden layers
                    layer_output = self.layers[layer_index].get_output(layer_output)
                pred_y = self.layers[len(self.layers) - 1].get_output(layer_output)     # output of the neural network
                real_y = y[input_index]

                self.optimizer.optimize(self.layers, pred_y, real_y, self.learning_rate)    # optimize weights in neural network

    def __is_shape_valid(self, x):
        input_shape = self.layers[0].units
        return len(x[0]) == input_shape
