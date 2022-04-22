from abc import ABC, abstractmethod
from re import L
from typing import List

import numpy as np

def weights_initialization(input_size, output_size):
    sigma = np.sqrt(2 / input_size)
    return np.random.normal(0, sigma, size=(input_size, output_size))

class Layer(ABC):

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_rate, learning_rate):
        raise NotImplementedError
        
class LinearLayer(Layer):

    def __init__(self, input_size, output_size):
        self.weights = weights_initialization(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.data = None 

    def forward(self, x):
        self.data = x
        return x.dot(self.weights) + self.biases

    def backward(self, output_rate, learning_rate):
        weights_old = self.weights.copy()
        self.weights = self.weights - self.data.reshape(-1, 1).dot(output_rate.reshape(1, -1)) * learning_rate
        self.biases = self.biases - (learning_rate*output_rate
)
        return output_rate.dot(weights_old.T)
    
class ReLU(Layer):

    def __init__(self) -> None:
        self.data = None

    def forward(self, x):
        self.data = x.copy()
        x[x<=0]=0
        return x

    def backward(self, output_rate, learning_rate):
        mask = (self.data>0).astype(np.float32)
        return mask*output_rate

    
class Softmax(Layer):

    def __init__(self) -> None:
        self.data = None

    def forward(self, x):
        self.data = 1/sum(np.exp(x)) * np.exp(x)
        return self.data

    def backward(self, output_rate, learning_rate):
        self.data[output_rate] -= 1
        return self.data

class Graph:

    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.lr = learning_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        for layer in reversed(self.layers):
            y = layer.backward(y, self.lr)