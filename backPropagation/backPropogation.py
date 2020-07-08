import numpy as np

# Basic Perceptron training algorithm that solves the BITWISE operators XOR using Backpropogation

# 3-3-1 Network due to the addition of the bias term embedded in the weight matrix
class NeuralNetwork:
    # The value of alpha controls how large or small of a step we take in gradient descent
    def __init__(self, layers, alpha=0.1):
        # Initialize the weight matrix with random values sampled from "normal"Gaussian disribution with 0 mean and unit variance
        # Then scaling the weight matrix
        self.W = []
        self.layers = layers
        self.alpha = alpha
        
        # Start looping from the index of the first layer but
        # stop before we reach the last two layers
        for i in np.arange(0, len(layers)-2):
            # +1 for the bias vector 2-2-1 -> 3-3-1
            w = np.random.rand(layers[i]+1, layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[i]))
        # The last two layers are a special case where the input
        # connections aneed a bias term but the output does not
        w = np.random.rand(layers[-2]+1, layers[-1])
        self.W.append(w/np.sqrt(layers[-2]))
        
    def __repr__(self):
        # Construct and return a string that represents the network architecture
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))
 
    # Sigmoid activation function
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def sigmoid_deriv(self,x):
        return x*(1-x)

    # Training procedure function
    def fit(self, X, y, epochs=1000, displayUpdate=100):
        # Insert column of 1s as the last entry in feature matrix for bias
        X = np.c_[X, np.ones((X.shape[0]))] 
        # Loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            # Loop over each individual data point
            for(x, target) in zip(X, y):
                self.fit_partial(x, target)
                
            # Check to see if we should display a training update
            if epoch == 0 or (epoch+1) % displayUpdate == 0:
                loss = self.calculate_loss(X,y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch+1, loss))
    
    # The actual heart of the backpropagation algorithm
    def fit_partial(self, x, y):
        # The first activation is a special case, it's just the input feature vector itself
        A = [np.atleast_2d(x)]

        # Forward pass - loop over the layers of the network
        for layer in np.arange(0, len(self.W)):
           # Weighted sum
           wsum = A[layer].dot(self.W[layer])
           # Activation function
           net = self.sigmoid(wsum)
           # Once we have the net output, add it to our list of acivations
           A.append(net)

        # Backward pass
        error = A[-1]-y
        D = [error * self.sigmoid_deriv(A[-1])]
        
        # Chain Rule
        for layer in np.arange(len(A)-2, 0, -1):
        # The delta for the current layer is equal to the delta of the previous layer D[-1]
        # dotted with the weight matrix of the current layer, followed by multiplying the delta
        # by the derivative of the nonlinear activation function
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        # Since we looped over our layers in reverse order we need to reverse the deltas
        D = D[::-1]

        # Weight update
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.alpha*A[layer].T.dot(D[layer])

    # Forward propagation to obtain the final output prediction
    def predict(self, X, addBias=True):
        # Ensure that our input is a matrix
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        # Loop over our layers in the network
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))
        return p

    # Calculate the loss across our entire training set
    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)

        loss = 0.5*np.sum((predictions-targets)**2)
        return loss