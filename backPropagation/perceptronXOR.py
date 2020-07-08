from backPropogation import NeuralNetwork
import numpy as np

# Construct the XOR dataset 
X = np.array(([0,0],[0,1],[1,0],[1,1]))
y = np.array([[0], [1], [1], [0]])

# Define our 2-2-1 neural network and train it
nn = NeuralNetwork([2,2,1], alpha=0.5)
print(nn)
print("[INFO] training perceptron..." )
nn.fit(X, y, epochs=30000)

# Now that our perceptron is trained we can evaluate it
print("[INFO] evaluating perceptron..." )

for(x,target) in zip(X,y):
    # Make a prediction on the data point and display the result
    pred = nn.predict(x)[0][0]
    step = 1 if pred > 0.5 else 0
    print("[INFO] data={}, ground-truth={}, pred={:.4f}, step={}".format(x, target[0], pred, step))