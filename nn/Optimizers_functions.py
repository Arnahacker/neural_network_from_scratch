import numpy as np
from layers import Layer
from optimizers import optimizers

class SGD(optimizers):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "weights_gradient"):
                layer.weights -= self.lr * layer.weights_gradient
            if hasattr(layer, "bias_gradient"):
                layer.bias -= self.lr * layer.bias_gradient

class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity_w = {}  # store velocity for weights per layer
        self.velocity_b = {}  # store velocity for biases per layer

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "weights_gradient"):
                # Initialize if not present
                if layer not in self.velocity_w:
                    self.velocity_w[layer] = np.zeros_like(layer.weights)
                if layer not in self.velocity_b:
                    self.velocity_b[layer] = np.zeros_like(layer.bias)

                # Update velocity
                self.velocity_w[layer] = (
                    self.momentum * self.velocity_w[layer]
                    - self.lr * layer.weights_gradient
                )
                self.velocity_b[layer] = (
                    self.momentum * self.velocity_b[layer]
                    - self.lr * layer.bias_gradient
                )

                # Update parameters
                layer.weights += self.velocity_w[layer]
                layer.bias += self.velocity_b[layer]
