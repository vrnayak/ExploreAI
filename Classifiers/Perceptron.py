# Perceptron.py
# OVERVIEW: Implements a linear perceptron algorithm for binary classification

import numpy as np

# A Linear Binary Classification Perceptron Algorithm 
class Perceptron(object):

  # Initializes normal vector to hyperplane and offset to 0
  def __init__(self, examples, labels):

    # Initialize hyperplane parameters
    self.offset = 0.0
    self.theta = np.zeros(len(examples[0]))

    # Initialize dataset
    self.size = len(examples)
    self.labels = np.array(labels)
    self.datapoints = np.array(examples)
    #print(self.datapoints)
  # Train algorithm to learn a decision boundary
  def train(self):

    while not self.converged():

      for point in range(self.size):
        if not self.classify(point):
          self.theta += (self.labels[point] * self.datapoints[point])
          self.offset += self.labels[point]
          point += 1

    print(self.theta)
    print(self.offset)
  

  def classify(self, point):
    actual = self.labels[point]
    predicted = 1
    #print(self.datapoints[point])
    if (np.dot(self.theta, self.datapoints[point]) + self.offset <= 0):
      predicted = -1
    return actual == predicted

  
  # Checks if any points are misclassified
  def converged(self):

    for i in range(self.size):
      if not self.classify(i):
        return False
    
    return True
