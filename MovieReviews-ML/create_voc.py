# Script that builds a vocabulary 

import collections
import os

# function that removes the punctuation for the process and replaces them with spaces ""
def remove_puntuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_'{|}~"
    for p in punct:
        text = text.replace(p," ")
    return text

# function that opens the file with encoding utf-8 and returns a list of words
def read_document(filename):
    f = open(filename, encoding = "utf-8")
# command that opens the file in reading mode
    text = f.read()
    f.close()
    words = []
    text = text.lower()
    text = remove_puntuation(text)
    for word in text.split():
        #print words bigger than 2 letters in order to obtain only verbs, nouns and adjectives
        if len(word)>2:
            words.append(word)
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







