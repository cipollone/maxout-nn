'''
This file contains an example of a simple model.
This is not the real net to use.
'''

import tensorflow as tf


def model(data, dropouts):
  '''\
  Example of a feedforward net definition.

  Args:
    data: tensor of input features (batch_size,) + features.shape
    dropouts: sequence of two scalar tensors that contains the dropout rate
        for input and hidden units.

  Returns:
    array of shape (batch_size, classes) with logits of each class
  '''

  # Sizes
  n_features = int(data.shape[1]) # We know it's 4
  n_classes = 3
  n_hidden = 20 # NOTE: Hidden layer added just to overfit. Testing dropout.

  ## Model

  # Dropout
  tensor = tf.nn.dropout(data, keep_prob=1-dropouts[0])

  # Layer 1
  with tf.variable_scope('l1'):
    W = tf.get_variable('W', shape=(n_features, n_hidden))
    b = tf.get_variable('b', shape=(n_hidden,))
    tensor = tf.sigmoid(tf.matmul(tensor, W) + b)

    # Dropout
    tensor = tf.nn.dropout(tensor, keep_prob=1-dropouts[1])

  # Layer 2
  with tf.variable_scope('l2'):
    W = tf.get_variable('W', shape=(n_hidden, n_classes))
    b = tf.get_variable('b', shape=(n_classes,))
    logits = tf.matmul(tensor, W) + b

  logits = tf.identity(logits, name='logits')
  return logits
