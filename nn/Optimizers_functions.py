import numpy as np
from layers import Layer
from optimizers import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def step(self, layers):
        for layer in layers:
            layer.weights -= self.lr * layer.weights_gradient
            layer.bias -= self.lr * layer.bias_gradient

class SGDMomentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity_w = {}  
        self.velocity_b = {}  

    def step(self, layers):
        for layer in layers:
            if layer not in self.velocity_w:
                self.velocity_w[layer] = np.zeros_like(layer.weights)
            if layer not in self.velocity_b:
                self.velocity_b[layer] = np.zeros_like(layer.bias)
            
            self.velocity_w[layer] = (
                self.momentum * self.velocity_w[layer]
                - self.lr * layer.weights_gradient
            )
            self.velocity_b[layer] = (
                self.momentum * self.velocity_b[layer]
                - self.lr * layer.bias_gradient
            )

            layer.weights += self.velocity_w[layer]
            layer.bias += self.velocity_b[layer]


class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        self.lr = learning_rate
        self.beta = beta
        self.eps = epsilon
        self.sq_grad = None

    def update(self, params, grads):
        if self.sq_grad is None:
            self.sq_grad = [np.zeros_like(g) for g in grads]

        for i in range(len(params)):
            self.sq_grad[i] = self.beta * self.sq_grad[i] + (1 - self.beta) * grads[i] ** 2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.sq_grad[i]) + self.eps)


class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(g) for g in grads]
            self.v = [np.zeros_like(g) for g in grads]

        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


