'''\
NN for the MNIST dataset.
'''


# NOTE: this is not the final net. Just testing the MNIST dataset


import tensorflow as tf

from . import units


def model(data, dropouts, seed=None):
  # TODO: wrong model and missing docstring

  # Sizes
  n_pixels = int(data.shape[1])
  n_classes = 10
  n_channels = 2 # The minimum

  with tf.variable_scope('maxout1'):
    logits = units.maxout_layer(data, n_classes, n_channels, seed)

  logits = tf.identity(logits, name='logits')

  return logits
