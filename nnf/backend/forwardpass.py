import numpy as np

def get_layer_output(inputs, weights, activation):
    """
    :param inputs: 1D array. Ouput of previous layer
    :param weights: 2D array. All weights connecting this layer to previous layer
    :param activation: activation function of layer
    :return: 2D array. Output of all nodes (net out and derivation out)
    """
    output = []

    for node_index in range(0, len(weights[0])):
        node_weights = [weights[previous_node_index][node_index] for previous_node_index in range(0, len(inputs))]
        output.append(get_node_output(inputs=inputs, weights=node_weights, activation=activation))

    return np.array(output)


def get_node_output(inputs, weights, activation):
    """
    :param inputs: inputs passed from previous layer
    :param weights: 1D array. Weights connected to specific node
    :param activation: activation function
    :return: output for specific node
    """
    summary = 0
    for i in range(0, len(inputs)):
        summary += inputs[i] * weights[i]

    output, derivation_out = activation(summary)

    return [output, derivation_out]