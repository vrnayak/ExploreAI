# Perceptron.py
# OVERVIEW: Implements a linear perceptron algorithm for binary classification

import numpy as np

# A Linear Binary Classification Perceptron Algorithm 
class Perceptron(object):

  # Initializes normal vector to hyperplane and offset to 0
  def __init__(self, dimension, verbose=False, separable=False, loss_function='0-1'):

    # Initialize hyperplane parameters
    self.offset = 0.0
    self.theta = np.zeros(dimension)

    # Initialize dataset variables
    self.numDataPoints = 0
    self.labels = np.array([])
    self.datapoints = np.array([])
    
    # Initialize other parameters 
    self.verbose = verbose
    self.separable = separable  # whether data is linearly separable
    self.loss_function = loss_function 

  # Train algorithm to learn a decision boundary
  def train(self, datapoints, labels):

    # Create numpy arrays using parameters
    self.labels = labels
    self.numDataPoints = self.labels.shape[0]
    self.datapoints = datapoints

    # Actual perceptron algorithm
    while not self.hasConverged():

      for point in range(self.numDataPoints):
        if self.misclassified(point):
          self.theta += (self.labels[point] * self.datapoints[point])
          self.offset += self.labels[point]

    return self.theta, self.offset
  
  # Determine whether current hyperplane misclassifies data point
  def misclassified(self, point):

    predicted = 1
    actual = self.labels[point]
    if (np.dot(self.theta, self.datapoints[point]) + self.offset <= 0):
      predicted = -1
    return actual != predicted

  
  # Checks convergence condition based on loss function
  def hasConverged(self):
  
    if self.loss_function == '0-1':
      return self.zeroOneLoss() == 0

  # 0-1 Loss Function
  def zeroOneLoss(self):

    results = self.labels * (np.dot(self.datapoints, self.theta) + self.offset)
    totalLoss = results[results <= 0].shape[0] / self.numDataPoints
    return totalLoss