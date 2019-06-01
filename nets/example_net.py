'''
This file contains an example of a simple model.
This is not the real net to use.
'''

# NOTE: just testing maxout units. This model should not be used.
#   The tiny iris dataset can be solved with a simple logistic model. Here,
#   we're just debugging the maxout_layer(). Also, it may be incorrect to
#   pass maxout to softmax directly, but it would be excessive here.


import tensorflow as tf

from . import units


def model(data, dropouts, seed=None):
  '''\
  Example of a feedforward net definition.
  Single maxout layer (no dropout).

  Args:
    data: tensor of input features (batch_size, n_features)
    dropout: not used in this model
    seed: seed for deterministic initialization of variables.

  Returns:
    logits: array of shape (batch_size, classes) with logits of each class
    n: size of the net. Number of weights.
  '''

  # Sizes
  n_features = int(data.shape[1]) # We know it's 4
  n_classes = 3
  n_channels = 2 # The minimum
  n = n_features*n_channels*n_classes

  with tf.variable_scope('maxout1'):
    logits = units.maxout_layer(data, n_classes, n_channels, seed)

  logits = tf.identity(logits, name='logits')

  return logits, n
