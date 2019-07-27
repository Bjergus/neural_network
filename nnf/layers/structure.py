class Neuron(object):

    def __init__(self, activation):
        self.weights = []
        self.inputs = []
        self.output = 0
        self.activation = activation

    def calculate_output(self, inputs):
        self.inputs = inputs

        net_input = self.__calculate_net_input()
        output, _ = self.activation(net_input)
        self.output = output

        return output

    def calculate_error(self, target_y):
        return 0.5 * (target_y - self.output) ** 2

    def calculate_output_wrt_net(self):
        out, derivate = self.activation(self.__calculate_net_input())
        return derivate

    def calculate_net_wrt_weight(self, index):
        """
        :param index: index of  node in previous layer
        :return: partial derivation of net input function with respect to given variable => given input from that node
        """
        return self.inputs[index]

    def __calculate_net_input(self):
        net_input = 0

        for i in range(0, len(self.inputs)):
            net_input += self.inputs[i] * self.weights[i]

        return net_input
