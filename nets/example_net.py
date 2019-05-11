'''
This file contains an example of a simple model.
This is not the real net to use.
'''

import tensorflow as tf


def model(data):
  '''\
  Example of a feedforward net definition.
  Logistic model (softmax not applied).

  Args:
    data: tensor of input features (batch_size,) + features.shape

  Returns:
    array of shape (batch_size, classes) with logits of each class
  '''

  # Sizes
  n_features = int(data.shape[1]) # We know it's 4
  n_classes = 3

  # Retrieve the weights
  W = tf.get_variable('W', shape=(n_features, n_classes)) # Transpose
  b = tf.get_variable('b', shape=(n_classes,))

  # Function
  logits = tf.matmul(data, W) + b
  logits = tf.identity(logits, name='logits')

  return logits
