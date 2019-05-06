# TODO: remove this file and use the final model

import tensorflow as tf


def model(data):
  '''\
  Temporary placeholder for the real net. Example of a feedforward net
  definition. Logistic model.

  Args:
    data: input features (batch_size,) + features.shape

  Returns:
    array of shape (batch_size, classes) with probability of each class
  '''

  w = tf.get_variable(name='w', initializer=tf.constant(3.0))
  out = w*data

  return out
