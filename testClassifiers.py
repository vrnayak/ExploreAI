# testClassifiers.py
# OVERVIEW: Tests ability of AI algorithms to classify handwritten digits from MNIST database

import numpy as np
import idx2numpy as idx
from Classifiers import SimpleNeuralClassifier

def load_mnist_data(imgPath, lblPath):

	examples = idx.convert_from_file(imgPath)
	labels = idx.convert_from_file(lblPath)

	examples = examples.reshape((examples.shape[0], 784))
	labels = labels.reshape((labels.size, 1))
	data = np.concatenate((examples, labels), axis=1)
	return data

def run_mnist_network():

	mnistTrainingSet = 'Datasets/mnistTrainingImgs-60k-idx1-ubyte'
	mnistTrainingLbls = 'Datasets/mnistTrainingLbls-60k-idx1-ubyte'
	mnistTestingSet = 'Datasets/mnistTestingImgs-10k-idx1-ubyte'
	mnistTestingLbls = 'Datasets/mnistTestingLbls-10k-idx1-ubyte'

	trainingData = load_mnist_data(mnistTrainingSet, mnistTrainingLbls)
	testingData = load_mnist_data(mnistTestingSet, mnistTestingLbls)

	nn = SimpleNeuralClassifier(layers = [784, 20, 10], eta = 1.0,
								batchSize = 50, epochs = 30)
	nn.train(trainingData)
	nn.test(testingData)
	nn.save('mnistNetworkInfo-784-20-10.csv')

if __name__ == '__main__':

	# Train & test neural network to classify handwritten digits from MNIST Database
	run_mnist_network()
