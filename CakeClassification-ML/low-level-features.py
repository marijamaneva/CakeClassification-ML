#  FEATURE EXTRACTION - Low-level features

import numpy as np
import matplotlib.pyplot as plt
import os 
import image_features

# listing the content of the directories; these will represent the classes
classes = os.listdir("images/train")
classes = [c for c in classes if not c.startswith(".")]

# function that will extract and read the images from the specified folder
def process_directory(path):
    all_features = []       # list to store extracted features
    all_labels = []         # list to store corresponding labels
    klass_label = 0         # label for the current class
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        image_files = [c for c in image_files if not c.startswith(".")]
        # read each image in the class
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename 
            image = plt.imread(image_path) / 255.0                      # read and normalize the image
            print(image_path)
            
            # Extract the first set of features from the image
            features1 = image_features.color_histogram(image)
            features1 = features1.reshape(-1)                              # flatten the features
            
            # Extract the second set of features from the image
            features2 = image_features.edge_direction_histogram(image)
            features2 = features2.reshape(-1)                               # flatten the features
            
            # Extract the third set of features from the image
            features3 = image_features.cooccurrence_matrix(image)
            features3 = features3.reshape(-1)
            
            # Concatenate the feature vectors
            combined_features = np.concatenate([features1, features2,features3])
                         
            all_features.append(combined_features)                         # add features to the list
            all_labels.append(klass_label)                                 # add label to the list
        klass_label += 1 
    X = np.stack(all_features, 0)                                          # stack the features into an array
    Y = np.array(all_labels)                                               # convert labels to numpy array
    return X, Y

# process the images in the "test" directory
X, Y = process_directory("images/test")
print("test", X.shape, Y.shape)
data = np.concatenate([X,Y[:,None]], 1)     # concatenate features and labels
np.savetxt("test.txt.gz",data)              # save the data to a text file

# process the images in the "train" directory
X, Y = process_directory("images/train")
print("train", X.shape, Y.shape)
data = np.concatenate([X,Y[:,None]], 1)     # concatenate features and labels
np.savetxt("train.txt.gz",data)             # save the data to a text file
  

