# createHyperplane.py

from Classifiers.Perceptron import Perceptron


data = [[1, 2], [3, 4], [0, 3], [6, 6], [-1, -4], [-2, -2], [-4, -6]]
labels = [1, 1, 1, 1, -1, -1, -1]

classifier = Perceptron(data, labels)
classifier.train()
