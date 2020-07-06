import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
# 100 feature sets of 3 classifications
X, y = spiral_data(100,3)

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing parameters like this removes the need of transposing the weight matrix later on
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU activation function
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
print(layer1.output)

activation1.forward(layer1.output)
print(layer1.output)