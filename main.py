from nnf import Dense, NeuralNetwork, Input

nn = NeuralNetwork([
    Input(units=1, activation='relu'),
    Dense(units=4, activation='sigmoid'),
    Dense(units=1, activation='sigmoid')
])

nn.compile(optimizer='classic', num_epochs=5)

train_x = [[255]]
train_y = [[1]]

nn.train(train_x, train_y)

print('test')
