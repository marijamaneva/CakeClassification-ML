# Script that builds and trains a SVM model

import numpy as np

def svm_train(X,Y, lambda_, lr = 0.035, steps = 10000):
    m,n = X.shape
    w = np.zeros(n)
    b = 0
    for step in range(steps):
        z = X @ w + b
        hinge_diff = -Y * (z < 1) + (1 - Y) * (z > -1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w,b

def svm_inference(X, w, b):
    z = X @ w + b
    labels = (z > 0).astype(int)
    return labels


def svm_accuracy(X,Y, w, b):
    labels = svm_inference(X,w,b)
    accuracy = (labels == Y).mean()
    return accuracy 

train_data = np.loadtxt("train.txt.gz")
test_data = np.loadtxt("test.txt.gz")
train_X = train_data[:, :-1]
train_Y = train_data[:, -1]
test_X = test_data[:,:-1]
test_Y = test_data[:,-1]

w, b = svm_train(train_X, train_Y, lambda_ = 0, lr=0.035, steps=10000)

train_predictions = svm_inference(train_X,w,b)
train_accuracy = (train_predictions==train_Y).mean()
print('Train Accuracy SVM=', train_accuracy *100)

test_predictions = svm_inference(test_X,w,b)
test_accuracy = (test_predictions==test_Y).mean()
print('Test Accuracy SVM=', test_accuracy*100)
