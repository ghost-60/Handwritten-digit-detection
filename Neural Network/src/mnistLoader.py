import pickle
import gzip

import numpy as np

def loadData():
    f = gzip.open('mnist.pkl.gz', 'rb')
    trainingData, validationData, testData = pickle.load(f, encoding='latin1')
    f.close()
    return (trainingData, validationData, testData)

def loadDataWrapper():
    trD, vaD, teD = loadData()
    trainingInputs = [np.reshape(x, (784, 1)) for x in trD[0]]
    trainingResults = [vectorizedResult(y) for y in trD[1]]
    trainingData = list(zip(trainingInputs, trainingResults))
    validationInputs = [np.reshape(x, (784, 1)) for x in vaD[0]]
    validationData = list(zip(validationInputs, vaD[1]))
    testInputs = [np.reshape(x, (784, 1)) for x in teD[0]]
    testData = list(zip(testInputs, teD[1]))
    return (trainingData, validationData, testData)

def vectorizedResult(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
