# digitClassification.py
# OVERVIEW: Tests ability of AI algorithms to classify handwritten digits from MNIST database

import numpy as np
import idx2numpy as idx
from Classifiers import SimpleNeuralClassifier

def load_mnist_data(imgPath, lblPath):

	examples = idx.convert_from_file(imgPath).flatten()
	labels = idx.convert_from_file(lblPath).flatten()
	data = np.concatenate((examples, labels), axis = 1)
	return data

if __name__ == '__main__':

	nn = SimpleNeuralClassifier(layers = [784, 20, 10], eta = 1.0,
								batchSize = 50, epochs = 30)

