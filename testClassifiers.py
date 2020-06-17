# testClassifiers.py
# OVERVIEW: Tests ability of AI algorithms to classify handwritten digits from MNIST database

import numpy as np
import idx2numpy as idx
from Classifiers import SimpleNeuralClassifier

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

	nn = SimpleNeuralClassifier(layers = [784, 20, 10], eta = 1.0, batchSize = 50, epochs = 30)

	if preload is True:
		nn.load(filename)
		print('Weights & biases have successfully been loaded from file.')
	else:
		mnistTrainingSet = 'Datasets/mnistTrainingImgs-60k-idx1-ubyte'
		mnistTrainingLbls = 'Datasets/mnistTrainingLbls-60k-idx1-ubyte'
		trainingData = load_mnist_data(mnistTrainingSet, mnistTrainingLbls)

		# Train Neural Network & Save Weights/Biases
		nn.train(trainingData)
		nn.save('mnistNetworkInfo-784-20-10.csv')

	# Test Neural Network
	mnistTestingSet = 'Datasets/mnistTestingImgs-10k-idx1-ubyte'
	mnistTestingLbls = 'Datasets/mnistTestingLbls-10k-idx1-ubyte'
	testingData = load_mnist_data(mnistTestingSet, mnistTestingLbls)

	accuracy = nn.test(testingData)
	print('The neural network correctly classified {:.2f}% of all images.'.format(accuracy * 100))


if __name__ == '__main__':
	
	# Important files
	networkInfo = 'mnistNetworkInfo-784-20-10.csv'

	# Train & test neural network to classify handwritten digits from MNIST Database
	run_mnist_network(True, networkInfo)
