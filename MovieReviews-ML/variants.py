# Script that adds the varians requested in the assignment:
    #- removes the stop words
    #- applies the Porter stemming algorithm

import collections
import os
import numpy as np


# function that removes the punctuation for the process and replaces them with spaces ""
def remove_puntuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_'{|}~"
    for p in punct:
        text = text.replace(p," ")
    return text


# function that removes the stopwords
def remove_common(words):
    f = open("stopwords.txt", encoding ="utf-8")
    badword = f.read()
    f.close()
    j=0
    for i in range(len(words)-j):
        if words[i-j] in badword:
            words.remove(words[i-j])
            j += 1
    return words


# function that applies the Porter stemming algorithm to a word
def porter_stem(word):
    suffixes = {
        'sses': 'ss',
        'ies': 'i',
        'ss':'ss',
        's': ''
    }
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)] + suffixes[suffix]
    return word
    

# function that opens the file with encoding utf-8 and returns a list of words
def read_document(filename):
    f = open(filename, encoding = "utf-8")
    text = f.read()
    f.close()
    words = []
# to put all the words in lower case
    text = text.lower()
    text = remove_puntuation(text)
    for word in text.split():
        if len(word)>2:
            stemmed_word = porter_stem(word)
            words.append(stemmed_word)
    words = remove_common(words)
    return words


# function that opens the file in writing mode and prints the most frequent words
def write_vocabulary(voc,filename,n):
    f = open(filename, "w")
    for word, count in voc.most_common(n):
        print(word, file = f)
    f.close()


# commands that read the file from the smalltrain directory, they are used to form
# a vocabulary in the file "vocabulary.txt"

# used to build a list
voc = collections.Counter()

for f in os.listdir("smalltrain/pos"):
    filename = "smalltrain/pos/" + f
    words = read_document(filename)
    voc.update(words)
    
for f in os.listdir("smalltrain/neg"):
    filename = "smalltrain/neg/" + f 
    words = read_document(filename)
    voc.update(words)
    
write_vocabulary(voc, "vocabulary.txt",1000)

filename = "smalltrain/pos/0_9.txt"
words = read_document(filename)
voc.update(words)
print(voc)
