import numpy as np
import matplotlib.pyplot as plt
from nn.dense import Dense
from nn.activation_function import Tanh
from nn.optimizers_functions import Adam
from nn.model import NeuralNetwork
from nn.loss import mse, mse_diff

# --- Create sine data (each row = one sample) ---
x_train = np.linspace(-np.pi, np.pi, 100).reshape(100, 1)   # shape (samples, features)
y_train = np.sin(x_train)                                   # shape (samples, 1)

# --- Build model ---
model = NeuralNetwork([
    Dense(1, 10),
    Tanh(),
    Dense(10, 10),
    Tanh(),
    Dense(10, 1)
], loss=mse, loss_prime=mse_diff)

# --- Optimizer ---
optimizer = Adam(lr=0.001)

# --- Train model ---
model.fit(x_train, y_train, epochs=2000, optimizer=optimizer, verbose=True)

# --- Evaluate ---
print("Final Loss:", model.evaluate(x_train, y_train))

# --- Predict ---
# model.predict() returns a list of numpy arrays → stack them into one array
y_pred_list = model.predict(x_train)
y_pred = np.hstack(y_pred_list)  # shape (1, N)
y_pred = y_pred.flatten()        # to 1D for plotting

# --- Plot ---
plt.plot(x_train.flatten(), y_train.flatten(), label="True Sine")
plt.plot(x_train.flatten(), y_pred, '--', label="Predicted")
plt.legend()
plt.title("Sine Curve Regression")
plt.show()
