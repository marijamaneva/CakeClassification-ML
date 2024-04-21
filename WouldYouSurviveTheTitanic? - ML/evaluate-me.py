import numpy as np
import matplotlib.pyplot as plt

# loading the parameters of the trained model from train-me.py
data = np.load("model.npz")
w = data["arr_0"]
b = data["arr_1"]
print(w, b)

# function that loads the file and obtain the data
def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

# function that determines the logistic regression model and calculates the probability 
def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p

X, Y = load_file("titanic-test.txt")
print("Loaded", X.shape[0], "feature vectors")

# calculation of the test accuracy 
P = logreg_inference(X, w, b)
predictions = (P > 0.5)
accuracy = (Y == predictions).mean()
print("Test accuracy:", accuracy * 100)

