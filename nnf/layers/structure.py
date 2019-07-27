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

    # Determine how much the neuron's total input has to change to move closer to the expected output
    #
    # Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
    # the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
    # the partial derivative of the error with respect to the total net input.
    # This value is also known as the delta (δ) [1]
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def __calculate_pd_error_wrt_net_input(self, target_uotput):
        return self.__calculate_pd_error_wrt_output(target_uotput) * self.calculate_output_wrt_net()

    # ∂E /∂yⱼ = -(tⱼ - yⱼ)
    def __calculate_pd_error_wrt_output(self, target_output):
        return - (target_output - self.output)

    def __calculate_net_input(self):
        net_input = 0

        for i in range(0, len(self.inputs)):
            net_input += self.inputs[i] * self.weights[i]

        return net_input
