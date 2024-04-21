# transfer learning 

import numpy as np
import pvml
import matplotlib.pyplot as plt
import os

#new model of the combination od the two

cnn = pvml.CNN.load("pvmlnet.npz")
mlp = pvml.MLP.load("cakes-mlp.npz")

cnn.weights[-1] = mlp.weights[0][None,None,:,:]
cnn.biases[-1] = mlp.biases[0]

cnn.save("cakes-cnn.npz")

# imagepath = "images/test/donuts/2512789.jpg"
# image = plt.imread(imagepath)/ 255
# labels, probs = cnn.inference(image[None,:,:,:])

classes = os.listdir("images/test")
classes = [c for c in classes if not c.startswith(".")]
classes.sort()

# indices = (-probs[0]).argsort()
# for k in range(5):
#     index = indices[k]
#     print(f"{k+1} {classes[index]:10} {probs[0][index] * 100: .1f}")

# plt.imshow(image)
# plt.show()


# Function to create a confusion matrix
def make_confusion_matrix_calc_display(path,cnn,name):
    confusionMatrix = []
    incorrect = 0
    total = 0
    confusionMatrix = np.zeros((len(classes), len(classes)), dtype = float)
    for klass in classes: 
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path)/255.0
            labels, probs = cnn.inference(image[None, :, :, :])
            confusionMatrix[classes.index(klass)][labels[0]] +=1
            if (classes.index(klass) != labels[0]):
                incorrect += 1
            total += 1
        
    confusionMatrix = confusionMatrix / confusionMatrix.sum(1,keepdims=True)
    print  (name + "accuracy is : " ,((total-incorrect)/total)*100 )
    plt.figure(figsize=(30, 30))
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(name +"Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = confusionMatrix.max() / 2.0
    for i in range(confusionMatrix.shape[0]):
        for j in range(confusionMatrix.shape[1]):
            if confusionMatrix[i, j] == 0:
                text = '0'
            else: 
                text = format(confusionMatrix[i, j], '.1')
                print(text)
            plt.text(j, i, text,
                     ha="center", va="center",
                     color="white" if confusionMatrix[i, j] > thresh else "black")
    plt.savefig(name + "confusion")
    plt.show()
    

    

make_confusion_matrix_calc_display("images/train",cnn,"first-try")