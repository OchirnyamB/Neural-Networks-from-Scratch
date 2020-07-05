# Modeling a network 4-3-3
# using dot product
import numpy as np
inputs = [1,2,3,2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
print(np.shape(weights))

# The 1st element when you do a dot product in numpy
# is how the return is going to be
output = np.dot(weights, inputs)+biases
print(output)
