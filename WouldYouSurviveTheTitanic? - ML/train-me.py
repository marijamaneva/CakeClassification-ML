#Machine Learning assignment
#Would you survive the Titanic?


import numpy as np
import matplotlib.pyplot as plt


#The linear regression is a model that combines a linear function with the sigmoid
# function that computes the sigmoid 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# function that determines the logistic regression model and calculates the probability 
def logreg_inference(x, w, b):
    logit  = (x @ w) + b
    p = sigmoid(logit)
    return p

# function that computes the cross-entropy
def cross_entropy(P,Y):
    #to avoid big values for P
    P = np.clip(P, 0.0001, 0.9999)
    ce = (-Y * np.log(P) - (1-Y) * np.log(1-P)).mean()
    return ce 

# function that trains the model obtained and computes the loss and accuracy values at each step
# the descent method approach is used to train the model and get the parameters  w and b
def logreg_train(X,Y, lr, steps):
    m,n = X.shape       #m= size of the training set, n = number of features
    w = np.zeros(n)
    b = 0
    accuracies = []
    losses = []

    
    for step in range(steps):
        P = logreg_inference(X, w, b)
        if step % 1000 == 0:
            loss = cross_entropy(P,Y)
            prediction = (P>0.5)
            accuracy = ( Y== prediction).mean()
            print(step,loss, accuracy*100)
            losses.append(loss)
            accuracies.append(accuracy)
        grad_w = (X.T @ (P - Y)) / m       #partial derivative of L in confront of w
        grad_b = (P - Y).mean()            #partial derivative of L in confront of b 
        # Gradient descent updates
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, losses, accuracies 
    

# function that loads the file and obtain the data
def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:,:-1]
    Y = data[:,-1]
    return X, Y
    
X, Y = load_file("titanic-train.txt")    
print(X.shape)
print(Y.shape)
w, b, losses, accuracies = logreg_train(X,Y, 0.005, 200000)

# print the values obtained 
print("w=", w)
print("b=",b)
print("accuracy",accuracies)

# plots of the loss and accuracy characterizing the model 
plt.plot(losses)
plt.xlabel("iterations")
plt.ylabel("losses")
plt.figure()
plt.plot(accuracies)
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.show()

# numpy command to save the parameters obtained by the training 
np.savez("model.npz", w, b)

# scatter plot of the distribution of the two most influential features (class and sex)
xrand= X + np.random.rand(X.shape[0],X.shape[1])/2
plt.scatter(xrand[:,0],xrand[:,1], c=Y)
plt.xlabel("class")
plt.ylabel("sex")


#class, sex, age, siblings/spouses aboard, parents/children aboard, fare
my_x =[2, 1, 23 ,1, 2, 50]
print('My probability of surviving the Titanic would have been:', 100*logreg_inference(my_x, w,b), '%' )



