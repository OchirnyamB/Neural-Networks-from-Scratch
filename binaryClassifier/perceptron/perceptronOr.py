from perceptron import Perceptron
import numpy as np

# Construct the OR dataset 
X = np.array(([0,0],[0,1],[1,0],[1,1]))
y = np.array([[0], [1], [1], [1]])

# Define our perceptron and train it
print("[INFO] training perceptron...")
per = Perceptron(X.shape[1], bias=1)
per.fit(X, y, epochs=10)

# Now that our perceptron is trained we can evaluate it
print("[INFO] testing perceptron..." )

for(x,target) in zip(X,y):
    # Make a prediction on the data point and display the result
    pred = per.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))