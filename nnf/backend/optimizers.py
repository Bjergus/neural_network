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



        return total_error

    def __optimizer_output_layer(self, layer, pred_y, real_y):
        pd_etotal_wrt_out = [self.__calculate_pd_etotal_wrt_out(pred_y[i], real_y[i]) for i in range(0, len(layer.nodes))]

        self.__optimize_layer(layer, pd_etotal_wrt_out)

    def __optimize_layer(self, layer,  pd_etotal_wrt_out):
        pd_out_wrt_net = []
        for i in range(0, len(layer.nodes)):

            pd_out_wrt_net.append(layer.nodes[i].calculate_output_wrt_net())
            for weight_index in range(0, len(layer.nodes[i].weights)):
                node = layer.nodes[i]
                weight = node.weights[weight_index]
                pd_net_wrt_w = node.calculate_net_wrt_weight(weight_index)
                pd_etotal_wrt_w = pd_net_wrt_w * pd_etotal_wrt_out[i] * pd_out_wrt_net[i]

                node.weights[weight_index] = weight - self.learning_rate * pd_etotal_wrt_w

    def __calculate_pd_etotal_wrt_out(self, pred_y, real_y):
        return - (real_y - pred_y)

    def __calculate_total_error(self, pred_y, real_y):
        total_error = 0

        for i in range(0, len(pred_y)):
            total_error += 0.5 * (real_y - pred_y) ** 2

        return total_error
