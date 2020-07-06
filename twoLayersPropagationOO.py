import numpy as np

# Gaussian disribution rounded around 0
np.random.seed(0)

# Batch size of inputs/features
X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing parameters like this removes the need of transposing the weight matrix later on
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# The number of neurons can be anything you want        
layer1 = Layer_Dense(4,5)
# Output the previous layer becomes the input of the second layer
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)