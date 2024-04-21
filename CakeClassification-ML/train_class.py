# Train classifier 

import numpy as np
import matplotlib.pyplot as plt
import pvml 


data = np.loadtxt("train.txt.gz")
X = data[:,:-1]
Y = data[:,-1].astype(int)

data = np.loadtxt("test.txt.gz")
Xtest= data[:,:-1]
Ytest = data[:,-1].astype(int)

nclasses = Y.max() +1 
mlp = pvml.MLP([X.shape[1],  nclasses])


plt.ion()
train_accs=[]
test_accs =[]
epochs =[]

for epoch in range(5000):
    steps = X.shape[0] // 50
    mlp.train(X,Y, lr=0.00001, batch = 50, steps= steps )
    
    if epoch % 100 == 0:
        predictions, probs = mlp.inference(X)
        train_acc = (predictions == Y).mean()
        
        predictions, probs = mlp.inference(Xtest)
        test_acc = (predictions == Ytest).mean()
        
        print(f"{epoch} {train_acc*100 : .1f} {test_acc*100 : .1f}")
        train_accs.append(train_acc*100)
        epochs.append(epoch)
        test_accs.append(test_acc*100)
        
        plt.clf()
        plt.plot(epochs, train_accs)
        plt.plot(epochs, test_accs)
        plt.legend(["train", "test"])
        plt.pause(0.01)
        

mlp.save("cakes-mlp.npz")
plt.ioff()
plt.show()


#with idk2 we are overfitting a little bit, which is quite common with this type of models

