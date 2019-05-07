
import numpy as np
import math

class ANN():
    
    def __init__(self, layers):
        super(ANN, self)
        self.layers = layers
    
    def __call__(self, x, classify = False):
        for layer in self.layers:
            x = Layer.sigmoid(layer(x))
        if classify:
            for i in range(len(x)):
                x[i, x[i,:] != np.max(x[i,:])] = 0
                x[i, x[i,:] == np.max(x[i,:])] = 1
        return x
    
    def loss(self, x, y):
        return np.sum((self.__call__(x) - y)**2) / len(x)
    
    def lossP(self, x, y):
        return 2 * (self.__call__(x) - y)
    
    def forward(self, x):
        activations = []
        activations.append(x)
        for layer in self.layers:
            x = Layer.sigmoid(layer(x))
            activations.append(x)
        return activations
    
    def backwards(self, x, y, learningRate = 1, reg = 0):
        activations = self.forward(x)
        delta = self.lossP(x, y) * activations[-1] * (1 - activations[-1])
        for i in range(1, len(self.layers) + 1):
            layer = self.layers[-1 * i]
            weightGrad = np.matmul(np.transpose(activations[-1 * (i + 1)]), delta) / len(x)
            biasGrad = np.reshape(np.sum(delta, 0), layer.biases.shape) / len(x)
            delta = np.matmul(delta, np.transpose(layer.weights)) * activations[-1 * (i + 1)] * (1 - activations[-1 * (i + 1)])
            layer.weights -= learningRate * (weightGrad + reg * layer.weights)
            layer.biases -= learningRate * biasGrad
            

class Layer():
    
    def __init__(self, inSize, outSize):
        super(Layer, self)
        self.weights = np.random.randn(inSize, outSize)
        self.biases = np.random.randn(1, outSize)
        
    def __call__(self, x):
        return np.matmul(x, self.weights) + self.biases
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))
    



#Data input
dataPath = "C:/Users/Justin Wang/Documents/GitHub/Machine-Learning/data/"
imageSize = 28 * 28
train_data = np.loadtxt(dataPath + "mnist_train.csv", delimiter = ",")
test_data = np.loadtxt(dataPath + "mnist_test.csv", delimiter = ",")

#Data processing & normalization
fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

train_labels = (np.arange(10) == train_labels).astype(np.float)
test_labels = (np.arange(10) == test_labels).astype(np.float)

#Define ANN
net = ANN([Layer(784, 64), Layer(64, 32), Layer(32, 10)])

#Network training
epochs = 20
batch_size = 128
intermLossUpkeep = 100

print("Initializing Training")
for epoch in range(epochs):
    train_idx = np.random.permutation(len(train_imgs))
    runningLoss = 0
    for i in range(math.floor(len(train_imgs) / batch_size)):
        batch_idx = train_idx[i * batch_size : (i + 1) * batch_size]
        runningLoss += net.loss(train_imgs[batch_idx], train_labels[batch_idx])
        if ((i + 1)%intermLossUpkeep == 0):
            runningLoss /= intermLossUpkeep
            print("[%d %5d] loss: %.5f" % (epoch + 1, i + 1, runningLoss))
            runningLoss = 0
        net.backwards(train_imgs[batch_idx], train_labels[batch_idx], reg = 0.0001)
    

#Network testing
y_hat = net(test_imgs, classify = True)

numRight = 0
numTotal = 0
for i in range(len(y_hat)):
    if (np.array_equal(y_hat[i,:], test_labels[i,:])):
        numRight += 1
    numTotal += 1

print("%.2f accuracy rate." % (numRight / numTotal * 100))