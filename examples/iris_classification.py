import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from nn.dense import Dense
from nn.activation_function import Tanh, Softmax
from nn.model import NeuralNetwork
from nn.optimizers_functions import Adam
from nn.loss import mse, mse_diff

iris = load_iris()
x = iris.data  
y = iris.target.reshape(-1, 1)  

encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)  

x = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = NeuralNetwork([
    Dense(4, 10),   
    Dense(10, 8),   
    Tanh(),
    Dense(8, 3),    
    Softmax()       
], loss=mse, loss_prime=mse_diff)


optimizer = Adam(lr=0.01)


model.fit(x_train, y_train, epochs=1000, optimizer=optimizer, verbose=True)

print("Final Test Loss:", model.evaluate(x_test, y_test))

def predict_class(model, x):
    outputs = model.predict(x)
    preds = [np.argmax(o) for o in outputs]
    return np.array(preds)

y_pred = predict_class(model, x_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
