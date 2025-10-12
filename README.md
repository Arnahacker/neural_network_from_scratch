# 🧠 Neural Network From Scratch (NumPy)

A fully-from-scratch implementation of a **neural network in Python using NumPy** — including forward propagation, backpropagation, and gradient descent — built to understand deep learning fundamentals without any frameworks.  
Tested on **XOR**, **Make Moons**, **Iris**, **MNIST**, and **Sine Curve regression** examples.

---

## 🚀 Overview

This project demonstrates the inner workings of a neural network — from the math of gradient descent to the code behind activations and losses.  
Every layer, weight update, and derivative is manually implemented for complete transparency.

### 🎯 Goal
To **learn deep learning by building it**, not by using black-box libraries.

---

## 🧩 Key Features
- Build and train neural networks **from scratch** using NumPy  
- Implement **forward and backward propagation** manually  
- Support for **dense layers**, **activations**, **losses**, and **optimizers**  
- Modular architecture for easy extension  
- Visualize training performance (loss and accuracy)  
- Example scripts for both classification and regression tasks

---

# Folder Structure

```
neural-network-from-scratch/
│
├── nn/
│   ├── layers.py                # Dense layer implementation
│   ├── activations.py           # Activation functions (ReLU, Sigmoid, etc.)
│   ├── loss.py                  # Loss functions (MSE, CrossEntropy)
│   ├── optimizers.py            # Optimizers (SGD, Adam)
│   ├── model.py                 # NeuralNetwork class (fit, predict, evaluate)
│
├── examples/
│   ├── xor_gate.py              # XOR logic gate classification
│   ├── make_moons.py            # Non-linear boundary demo
│   ├── iris_classification.py   # Multi-class structured data
│   ├── mnist_digits.py          # Handwritten digit recognition
│   ├── regression_sine_curve.py # Continuous regression test
│
├── tests/
│   ├── test_forward_backward.py # Unit tests for gradients and layers
│
└── README.md                    # Project documentation
```


## 📊 Example Datasets

| Example | Dataset | Type | Goal |
|----------|----------|------|------|
| `xor_gate.py` | XOR Logic Gate | Binary Classification | Verify nonlinear learning & gradient correctness |
| `make_moons.py` | sklearn `make_moons` | Binary Classification | Test non-linear decision boundaries |
| `iris_classification.py` | Iris Dataset | 3-Class Classification | Structured, small dataset |
| `mnist_digits.py` | MNIST Digits | Multi-Class Classification | 28×28 grayscale images (flattened) |
| `regression_sine_curve.py` | Generated Sine Data | Regression | Test continuous output learning |

---


