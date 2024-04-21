# Script that extracts the features and builds the feature vectors

#import collections 
import numpy as np
import os

# function that loads the vocabulary and returns it
# the output is a list of words mapped to numerical indices
def load_vocabulary(filename):
    voc = {}
    f = open(filename)
    text = f.read()
    f.close()
    n = 0
    for word in text.split():
        voc[word] = n
        n += 1
    return voc


# function that removes the punctuation for the process and replaces them with spaces ""
def remove_puntuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_'{|}~"
    for p in punct:
        text = text.replace(p," ")
    return text


# function that reads a document and retursn bow
def read_document(filename, voc):
    f = open(filename, encoding = "utf-8")
    text = f.read()
    f.close()
    bow = np.zeros(len(voc))
    text = text.lower()
    text = remove_puntuation(text)
    for word in text.split():
        if word in voc:
            #increment of the counter
            index = voc[word]
            bow[index] += 1 
    return bow

# the following script computes the bow representation of all the training documents
# this operation needs to be applied to the validation and test set
 
# load the vocabulary
voc = load_vocabulary("vocabulary.txt")

# initialize lists to store documents and labels 
train_documents = []
train_labels = []
test_documents =[]
test_labels = []

# process the training set
for f in os.listdir("smalltrain/pos"):
    filename = "smalltrain/pos/" + f
    bow = read_document(filename,voc)
    train_documents.append(bow)
    train_labels.append(1)
    
for f in os.listdir("smalltrain/neg"):
    filename = "smalltrain/neg/" + f 
    bow = read_document(filename,voc)
    train_documents.append(bow)
    train_labels.append(0)
    
# process the test set
for f in os.listdir("test/pos"):
    filename = "test/pos/" + f
    bow = read_document(filename, voc)
    test_documents.append(bow)
    test_labels.append(1)
    
for f in os.listdir("test/neg"):
    filename = "test/neg/" + f 
    bow = read_document(filename, voc)
    test_documents.append(bow)
    test_labels.append(0)


# np.stack converts the vectors in a 2D array
train_X = np.stack(train_documents)
train_Y = np.array(train_labels)
test_X = np.stack(test_documents)
test_Y = np.array(test_labels)

# np.concatenate append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt
train_data = np.concatenate([train_X, train_Y[:, None]], 1)
test_data = np.concatenate([test_X, test_Y[:, None]], 1)

# save the train and test data as files with np.savetxt command
np.savetxt("train.txt.gz", train_data)
np.savetxt("test.txt.gz", test_data)
    
voc = load_vocabulary("vocabulary.txt")
print(voc)


