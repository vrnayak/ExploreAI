# Classifiers.py
# OVERVIEW: This file contains implementations of AI algorithms used for classification tasks

import numpy as np
import pandas as pd
from tqdm import tqdm

class SimpleNeuralClassifier(object):

	# OVERVIEW: Feedforward neural network using MSE loss

	# Initialize neurons, layers, weights, biases, and hyperparameters
	def __init__(self, layers = None, eta = 1.0, batchSize = 40, epochs = 30):

		# Initialize sizes of each layer
		self.sizes = np.array([784, 20, 10]) if layers is None else np.array(layers)
		self.layers = np.zeros([self.sizes.size, np.max(self.sizes)])
		self.numLayers = self.sizes.size

		# Initialize weights, biases, and gradient
		neuronWeightDims = np.hstack((np.array(self.layers.shape), np.array([np.max(self.sizes)])))
		self.biases = 2 * np.random.random(self.layers.shape) - 1 	# Biases initialized between -1 & 1
		self.weights = 2 * np.random.random(neuronWeightDims) - 1	# Weights initialized between -1 & 1

		# Initialize weight & bias gradients (matrices of partials)
		self.biasGradient = np.zeros(self.biases.shape)		# Same shape as self.biases, contains db
		self.weightGradient = np.zeros(self.weights.shape)	# Same shape as self.weights, contains dw

		# Initialize hyperparameters
		self.eta = eta					# Learning rate
		self.batchSize = batchSize		# Batch size (for stochastic gradient descent)
		self.epochs = epochs			# Number of training epochs

		# Initialize useful functions
		self.sigmoid = lambda z: 1 / (1 + np.exp(-1 * z))
		self.sigmoid_inv = lambda x: np.log(0.01 + x) - np.log(1.01 - x)
		self.sigmoid_deriv = lambda z: (self.sigmoid(z) * (1 - self.sigmoid(z)))
		self.vectorize = lambda label: np.eye(self.sizes[-1])[label]

	# Trains neural network on list of labelled examples
	def train(self, data):

		for epoch in tqdm(range(self.epochs)):

			accuracy = 0
			np.random.shuffle(data)
			numBatches = int(np.floor(data.shape[0] / self.batchSize))
			for batch in range(numBatches):

				start = batch * self.batchSize
				minibatch = data[start : start + self.batchSize]
				for example in minibatch:

					label = example[-1]
					pred, cost = self.feed(example[:-1], label)
					accuracy += 1 if pred == label else 0
					self.compute_gradient(label)

				self.learn()	# Adjust weights/biases

			print('Epoch {}: {} / {} images correctly classified'
				  .format(epoch + 1, accuracy, len(data)))

	# Tests neural network on list of labelled examples
	def test(self, data):

		accuracy = 0
		for example in data:
			pred, cost = self.feed(example[:-1], example[-1])
			accuracy += 1 if pred == example[-1] else 0

		return accuracy / len(data)	# % of classifications correctly answered

	# Store neural network's weights & biases in CSV file
	def save(self, filename):

		networkInfo = np.hstack((self.weights.flatten(), self.biases.flatten()))
		np.savetxt(filename, networkInfo, delimiter = ',')

	# Feeds example to neural network and generates prediction/loss
	def feed(self, example, label):

		self.layers[0, :self.sizes[0]] = example / 256
		for i in range(1, self.numLayers):		# For each layer

			prevLayer = self.layers[i - 1, :self.sizes[i - 1]]
			neuralMatrix = self.weights[i, :self.sizes[i], :self.sizes[i - 1]]
			weightSums = np.matmul(neuralMatrix, prevLayer) + self.biases[i, :self.sizes[i]]
			self.layers[i, :self.sizes[i]] = self.sigmoid(weightSums)

		pred = np.argmax(self.layers[-1, :self.sizes[-1]])
		return pred, self.loss(label)

	# Uses backpropagation to compute gradient (used to adjust weights/biases)
	def compute_gradient(self, label):

		# Compute error for output layer
		outputLayer = self.layers[-1, :self.sizes[-1]]
		lossDerivs = self.sigmoid_deriv(self.sigmoid_inv(outputLayer))
		outputErr = self.loss_deriv(label) * lossDerivs

		# Adjust weights/biases for output layer
		self.biasGradient[-1, :self.sizes[-1]] -= self.eta * outputErr
		for i in range(self.sizes[-1]):		# For each neuron in the output layer
			delta = self.eta * outputErr[i] * self.layers[-2, :self.sizes[-2]]
			self.weightGradient[-1, i, :self.sizes[-2]] -= delta

		# Compute errors for remaining layers
		prevError = outputErr
		for layer in range(-2, -1 * self.numLayers, -1):

			# Compute error
			currLayer = self.layers[layer, :self.sizes[layer]]
			lossDerivs = self.sigmoid_deriv(self.sigmoid_inv(currLayer))
			neuralMatrix = self.weights[layer + 1, :self.sizes[layer + 1], :self.sizes[layer]]
			error = np.matmul(neuralMatrix.T, prevError) * lossDerivs

			# Adjust weights/biases for current layer
			self.biasGradient[layer, :self.sizes[layer]] -= self.eta * error
			for i in range(self.sizes[layer]):
				delta = self.eta * error[i] * self.layers[layer - 1, :self.sizes[layer - 1]]
				self.weightGradient[layer, i, :self.sizes[layer - 1]] -= delta

			prevError = error

	# Uses computed gradient to adjust weights/biases
	def learn(self):

		# Scale gradients by number of examples seen
		self.biasGradient /= self.batchSize
		self.weightGradient /= self.batchSize

		# Add gradients (partials) to adjust weights/biases
		self.biases += self.biasGradient
		self.weights += self.weightGradient

		# Reset weight & bias gradients
		self.biasGradient = np.zeros(self.biases.shape)
		self.weightGradient = np.zeros(self.weights.shape)

	# Computes MSE loss function (returns scalar)
	def loss(self, label):
		expected = self.vectorize(label)
		actual = self.layers[-1, :self.sizes[-1]]
		lossMSE = np.linalg.norm(expected - actual) ** 2
		return lossMSE

	# Computes derivative of loss function (returns vector)
	def loss_deriv(self, label):
		expected = self.vectorize(label)
		actual = self.layers[-1, :self.sizes[-1]]
		return actual - expected
