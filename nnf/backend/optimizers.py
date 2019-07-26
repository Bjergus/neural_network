from nnf.ui import informator


def get(optimizer):
    if optimizer == 'classic':
        return ClassicOptimizer()
    else:
        raise NotImplementedError


class Optimizer(object):

    def optimize(self, layers, predicted_y, real_y, learning_rate):
        raise NotImplementedError


class ClassicOptimizer(Optimizer):

    def optimize(self, layers, predicted_y, real_y, learning_rate):
        self.__optimize_output_layer(layer=layers[len(layers) - 1], predicted_y=predicted_y,
                                     real_y=real_y, learning_rate=learning_rate)

    def __optimize_output_layer(self, layer, predicted_y, real_y, learning_rate):
        """
        EQUATION: ∂Etotal/∂wx = ∂Etotal/∂outx * ∂outx/∂netx * ∂netx/∂wx
        :param layer:
        :param predicted_y:
        :param real_y:
        :return: new weights for output layer
        """
        Etotal = self.__get_total_error(predicted_y=predicted_y, real_y=real_y)
        # ∂Etotal/∂outx (Errors for all output nodes)
        errors = self.__get_output_errors(predicted_y=predicted_y, real_y=real_y)
        # ∂outx/∂netx
        activation_derivate = layer.get_last_full_out()[:, 1]
        # CALCULATE NEW WEIGHT VALUE
        # ∂netx/∂wx
        last_input = layer.get_last_input()
        for current_node_index in range(0, layer.units):
            for previous_node_index in range(0, len(layer.weights)):
                out_previous_node = last_input[previous_node_index]
                # ∂Etotal/∂wx
                ETotal_wx = errors[current_node_index] * activation_derivate[current_node_index] * out_previous_node

                old_weight = layer.weights[previous_node_index][current_node_index]
                new_weight = old_weight - learning_rate * ETotal_wx

                layer.set_weight(previous_node_index, current_node_index, new_weight)

        informator.add_total_error(error=Etotal)

    def __get_total_error(self, predicted_y, real_y):
        total = 0

        for i in range(0, len(predicted_y)):
            total += 0.5 * (real_y[i] - predicted_y[i])

        return total

    def __get_output_errors(self, predicted_y, real_y):
        # By applying chain rule on loss function we get this equation
        # -(target - out)
        errors = []

        for i in range(0, len(predicted_y)):
            error = - (real_y[i] - predicted_y[i])
            errors.append(error)

        return errors

    def __optimize_layer(self, weights, activation, error):
        pass
