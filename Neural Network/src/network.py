import numpy as np
import random
class Network:
    def __init__(self, sizes):
        self.nlayers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[: -1], sizes[1 :])]


    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidPrime(self, z):
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))

    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, trainingData, epochs, batchSize, eta, testData = None):
        if testData:
            nTest = len(testData)
        n = len(trainingData)
        for j in range(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k : k + batchSize] for k in range(0, n, batchSize)]
            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)
            if testData:
                print("Epoch {0} : {1} / {2}".format(j, self.evaluate(testData), nTest))
            else:
                print("Epoch {0} complete".format(j))

    def updateMiniBatch(self, miniBatch, eta):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(x, y)
            nablaB = [nb + dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw + dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.biases = [b - (eta/len(miniBatch)) * nb for b, nb in zip(self.biases, nablaB)]
        self.weights = [w - (eta/len(miniBatch)) * nw for w, nw in zip(self.weights, nablaW)]

    def evaluate(self, testData):
        testResults = [(np.argmax(self.feedForward(x)), y) for x, y in testData]
        return sum(int(x == y) for x, y in testResults)

    def costDerivative(self, outputActivations, y):
        return (outputActivations - y)

    def backprop(self, x, y):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.costDerivative(activations[-1], y) * self.sigmoidPrime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.nlayers):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l - 1].transpose())
        return(nablaB, nablaW)
