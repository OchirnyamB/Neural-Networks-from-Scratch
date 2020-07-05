# Modeling a network of  4-3-3
#                        4
#                        4
# using dot product
import numpy as np

# Batch size of inputs/features
inputs = [[1,2,3,2.5],
[2.0,5.0,-1.0,2.0],
[-1.5,2.7,3.3,-0.8]]
# shape(inputs) = 3x4

# Our layer of 3 neurons
# The size of weights and biases are how many neurons we have in the layer
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# shape(weights) = 3x4           
biases = [2,3,0.5]

# The 1st element when you do a dot product in numpy is how the return is going to be
# (3x4) x (3x4) is not possible in matrices product
# therefore transpose of a matrix is needed (3x4)x(4x3) = (3x3)
layer1_outputs = np.dot(inputs, np.array(weights).T)+biases

# Add an another layer
weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]        
biases2 = [-1,2,-0.5]

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T)+biases2

print(layer2_outputs)


