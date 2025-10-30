import numpy as np
from nn.dense import Dense
from nn.optimizers_functions import Adam
from nn.model import NeuralNetwork
from nn.loss import mse_diff,mse
from nn.activation_function import Tanh,Sigmoid

# XOR dataset
x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
y_train = np.array([[0],[1],[1],[0]])

# Build model
model = NeuralNetwork([
    Dense(2, 4),
    Tanh(),
    Dense(4, 1),
    Sigmoid()
], loss=mse, loss_prime=mse_diff)

# Initialize optimizer
optimizer = Adam(lr=0.01)

# Train
model.fit(x_train, y_train, epochs=500, optimizer=optimizer, verbose=True)

# Evaluate
print("Final Loss:", model.evaluate(x_train, y_train))

# Predictions
for x in x_train:
    print(f"{x} → {model.predict([x])[0].flatten()}")
