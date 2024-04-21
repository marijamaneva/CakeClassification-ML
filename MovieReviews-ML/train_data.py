# Script that trains the data and builds a classifier

import numpy as np

# function that trains a binary NB classifier
def train_nb(X,Y):
    n = len(X)
    #let's obtain the probability of the positive and negative words 
    #because some words may not appear at all (we could have log(0))
    #number of repetition of words in each row where Y = 1
    pos_p = X[Y == 1, :].sum(0) 
    pos_p = pos_p + 1 
    #probability for class 1 
    pos_p = pos_p / pos_p.sum()
    #number of repetition of words in each row where Y = 0
    neg_p = X[Y == 0, :].sum(0)
    neg_p = neg_p +1 
    neg_p = neg_p / neg_p.sum()
    #score w1-w0
    w = np.log(pos_p) - np.log(neg_p)
    # Estimate P(0) and P(1) and compute b
    pos_prior = Y.mean()
    neg_prior = 1 - pos_prior
    #score b1 - b0
    b = np.log(pos_prior) - np.log(neg_prior)
    return w, b

# function that computes the prediction of the NB classifier
def inference_nb(X,w,b):
    score = X @ w + b
    return (score > 0).astype(int)
    
# the following script load a training and test data, trains a classifier
# and evaluates it on the test data

train_data = np.loadtxt("train.txt.gz")
test_data = np.loadtxt("test.txt.gz")
train_X = train_data[:, :-1]
train_Y = train_data[:, -1]
test_X = test_data[:,:-1]
test_Y = test_data[:,-1]

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

w,b = train_nb(train_X,train_Y)

train_predictions = inference_nb(train_X,w,b)
train_accuracy = (train_predictions==train_Y).mean()
print('Train Accuracy NB=', train_accuracy *100)

test_predictions = inference_nb(test_X,w,b)
test_accuracy = (test_predictions==test_Y).mean()
print('Test Accuracy NB=', test_accuracy*100)

#X contains the number of words for every file . Every row represents a different file
#and every column represents a different word with the element of the matrix as number that 
#indicates how much that word appears in that file

#Y represents a label. 1 and 0 if the files are positive or negative 

#we have to do the sum of the words in which the Y is equal to 1 and then we 
#have to do the same thing when the Y is equal to 0 


# detection of the most relevant words of the classifier 
f = open("vocabulary.txt")
words = f.read().split()
f.close()

indices = np.argsort(w)
print('NEGATIVE')
for i in indices[:20]:
    print(words[i], w[i])

print()
print('POSITIVE')
for i in indices[-20:]:
    print(words[i],w[i])   
    
    
# compute test set predictions and confidence scores
test_scores = test_X @ w + b
test_preds = inference_nb(test_X, w, b)
test_confidence = np.abs(test_scores)

# find misclassified instances with highest confidence scores
worst_errors = np.argsort(test_confidence[test_preds != test_Y])[-20:]

# print worst errors
for i in worst_errors:
    if test_preds[i] != test_Y[i]:
        print("Instance:", i)
        print("Prediction:", test_preds[i])
        print("Actual Label:", test_Y[i])
        print("Confidence:", test_confidence[i])
        print("Words:", [words[j] for j in range(len(test_X[i])) if test_X[i,j] != 0])
        print() 

