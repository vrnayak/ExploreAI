# testClassifiers.py
# OVERVIEW: Tests ability of AI algorithms to classify handwritten digits from MNIST database

import numpy as np
import idx2numpy as idx
from Classifiers.FFNeuralNetwork import FFNeuralNetwork

# Loads images & labels from MNIST Database
def load_mnist_data(imgPath, lblPath):

	examples = idx.convert_from_file(imgPath)
	labels = idx.convert_from_file(lblPath)

	examples = examples.reshape((examples.shape[0], 784))
	labels = labels.reshape((labels.size, 1))
	data = np.concatenate((examples, labels), axis=1)
	return data

# If preload is True, weights & biases are preloaded instead of trained
def run_mnist_network(preload, filename = None):

	nn = FFNeuralNetwork(layers = [784, 25, 10], eta = 3.0, batchSize = 40, epochs = 30, verbose = True)

	if preload is True:
		nn.load(filename)
		print('Weights & biases have successfully been loaded from file.')
	else:
		mnistTrainingSet = 'DigitClassification/Datasets/mnistTrainingImgs-60k-idx1-ubyte'
		mnistTrainingLbls = 'DigitClassification/Datasets/mnistTrainingLbls-60k-idx1-ubyte'
		trainingData = load_mnist_data(mnistTrainingSet, mnistTrainingLbls)

		# Train Neural Network & Save Weights/Biases
		nn.train(trainingData)
		nn.save('mnistNN-784-25-10-3-40-30.csv')

	# Test Neural Network
	mnistTestingSet = 'DigitClassification/Datasets/mnistTestingImgs-10k-idx1-ubyte'
	mnistTestingLbls = 'DigitClassification/Datasets/mnistTestingLbls-10k-idx1-ubyte'
	testingData = load_mnist_data(mnistTestingSet, mnistTestingLbls)

	accuracy = nn.test(testingData)
	print('The neural network correctly classified {:.2f}% of all images.'.format(accuracy * 100))


if __name__ == '__main__':

	# Important files
	networkInfo1 = 'mnistNN-784-20-10-3-50-30.csv'

	# Train & test neural network to classify handwritten digits from MNIST Database
	#run_mnist_network(True, networkInfo)
	run_mnist_network(preload=False)
