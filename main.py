
from nnf import NeuralNetwork
from nnf.layers import Input, Dense

nn = NeuralNetwork([
    Input(1),
    Dense(2, activation='sigmoid')
])

nn.train([[1]], [[1, 2]])