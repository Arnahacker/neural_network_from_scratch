import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from nn.dense import Dense
from nn.activation_function import Tanh, Sigmoid
from nn.optimizers_functions import Adam
from nn.model import NeuralNetwork
from nn.loss import binary_cross_entropy, binary_cross_entropy_diff

x, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.reshape(-1, 1)   

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = NeuralNetwork([
    Dense(2, 16),    
    Tanh(),
    Dense(16, 8),
    Tanh(),
    Dense(8, 1),
    Sigmoid()        
], loss=binary_cross_entropy, loss_prime=binary_cross_entropy_diff)

optimizer = Adam(lr=0.01)

model.fit(x_train, y_train, epochs=1000, optimizer=optimizer, verbose=True)

print("Final Test Loss:", model.evaluate(x_test, y_test))

def predict_meshgrid(model, xx, yy):
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = [model.forward(p.reshape(-1, 1))[0, 0] for p in grid_points]
    return np.array(preds).reshape(xx.shape)

# Create a grid
x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = predict_meshgrid(model, xx, yy)
Z = (Z > 0.5).astype(int)  # Threshold at 0.5

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train.flatten(), cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("Two Moons Classification")
plt.show()
