import numpy as np

# Gaussian disribution rounded around 0
np.random.seed(0)

# Basic Perceptron training algorithm that solves the BITWISE operators: OR, AND

# N-inputs - 1 hidden layer with 1 neuron - Output Layer
class Perceptron:
    # The value of alpha controls how large or small of a step we take in gradient descent
    def __init__(self, N, alpha=0.1):
        # Initialize the weight matrix with random values sampled from "normal"Gaussian disribution with 0 mean and unit variance
        # Then scaling the weight matrix
        self.W = np.random.rand(N+1)/np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # Apply the step function
        return 1 if x > 0 else 0 

    # Training procedure function
    def fit(self, X, y, epochs):
        # Insert column of 1s as the last entry in feature matrix for bias
        X=np.c_[X, np.ones((X.shape[0]))]
        # Loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each individual data point
            for(x, target) in zip(X, y):
                pred = self.step(np.dot(x, self.W))
                
                # Only perform a weight update if our prediction does not match the target
                if(pred != target):
                    error = pred-target

                    # Update the weight matrix
                    self.W += -self.alpha*error*x
    
    def predict(self, X):
        # Ensure that our input is a matrix
        X=np.atleast_2d(X)
        X=np.c_[X, np.ones((X.shape[0]))]
        return self.step(np.dot(X, self.W))