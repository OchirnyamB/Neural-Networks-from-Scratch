# Modeling a network 4-3-3
# using dot product
import numpy as np

# Batch size of inputs/features
inputs = [[1,2,3,2.5],
[2.0,5.0,-1.0,2.0],
[-1.5,2.7,3.3,-0.8]]
# shape(inputs) = 3x4

# Our layer of 3 neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# shape(weights) = 3x4           
biases = [2,3,0.5]

# The 1st element when you do a dot product in numpy is how the return is going to be
# (3x4) x (3x4) is not possible in matrices product
# therefore transpose of a matrix is needed (3x4)x(4x3) = (3x3)
output = np.dot(inputs, np.array(weights).T)+biases
print(output)
