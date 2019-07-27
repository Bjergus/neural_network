"""
 pd => partial derivation
 wrt => with respect to
"""


def get_optimizer(name, learning_rate=0.01):
    if name == 'standard':
        return StandardOptimizer(learning_rate=learning_rate)
    else:
        raise NotImplementedError


class Optimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, layers, pred_y, real_y):
        raise NotImplementedError


class StandardOptimizer(Optimizer):

    def optimize(self, layers, pred_y, real_y):
        total_error = self.__calculate_total_error(pred_y, real_y)
        output_deltas = self.__optimizer_output_layer(layers[-1], real_y)
        self.__optimize_hidden_layers(self.__get_hidden_layers(layers), output_deltas, layers[-1])

        return total_error

    def __optimizer_output_layer(self, layer, real_y):
        # calculate delta of output layer
        pd_errors_wrt_output_neuron_net_input = [0] * len(layer.nodes)

        for i in range(0, len(layer.nodes)):
            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_net_input[i] = layer.nodes[i]\
                .__calculate_pd_error_wrt_total_net_input(real_y)

        # Update output neuron weights
        for o in range(len(layer.nodes)):
            for w in range(len(layer.nodes[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_net_input[o] * layer.nodes[
                    o].calculate_pd_total_net_input_wrt_weight(w)

                # Δw = α * ∂Eⱼ/∂wᵢ
                layer.nodes[o].weights[w] -= self.learning_rate * pd_error_wrt_weight

        return pd_errors_wrt_output_neuron_net_input

    def __optimize_hidden_layers(self, layers, output_deltas, output_layer):
        pd_errors_wrt_hidden_neuron_total_net_input = []
        for i in range(len(layers)):
            pd_errors_wrt_hidden_neuron_total_net_input = [[0] * len(
                layers[-1 - i].nodes)] + pd_errors_wrt_hidden_neuron_total_net_input
            for j in range(len(layers[-1 - i].nodes)):
                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
                d_error_wrt_hidden_neuron_output = 0
                if i == 0:
                    for o in range(len(output_layer)): # for each node in output layer
                        d_error_wrt_hidden_neuron_output += output_deltas[o] * output_layer.nodes[o].weights[j]
                else:
                    for o in range(len(layers[-i].nodes)):
                        d_error_wrt_hidden_neuron_output += pd_errors_wrt_hidden_neuron_total_net_input[1][o] * layers[-i].nodes[o].weights[j]

                # ∂E /∂zⱼ = dE / dyⱼ * ∂zⱼ /∂
                pd_errors_wrt_hidden_neuron_total_net_input[0][j] = d_error_wrt_hidden_neuron_output * \
                                                                    layers[-i - 1].neurons[
                                                                        j].calculate_pd_total_net_input_wrt_input()
                # TODO: UPDATE WEIGHTS
            return pd_errors_wrt_hidden_neuron_total_net_input

    def __calculate_pd_etotal_wrt_out(self, pred_y, real_y):
        return - (real_y - pred_y)

    def __calculate_total_error(self, pred_y, real_y):
        total_error = 0

        for i in range(0, len(pred_y)):
            total_error += 0.5 * (real_y[i] - pred_y[i]) ** 2

        return total_error

    def __get_hidden_layers(self, layers):
        hidden = []
        for i in range(1, len(layers) - 1):
            hidden.append(layers[i])
        return hidden
