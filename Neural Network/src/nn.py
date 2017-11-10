import mnistLoader
import network

trainingData, validationData, testData = mnistLoader.loadDataWrapper()
net = network.Network([784, 30, 10])
net.SGD(trainingData, 30, 10, 3.0, testData=testData)
