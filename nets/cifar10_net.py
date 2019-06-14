'''\
NN for the CIFAR-10 dataset.
'''

import tensorflow as tf
import numpy as np

from . import units


def model(data, dropouts, seed=None):
  '''\
  FF Neural network for the CIFAR-10 dataset. Using maxout units and dropouts.

  Args:
    data: tensor of input images (batch_size, 32 rows, 32 cols, 3 channels)
    dropouts: two scalar tensors that contains the dropout rate for input and
      hidden units.
    seed: seed for deterministic initialization of variables.

  Return:
    logis: (batch_size, n_classes). n_classes=10
  '''
  
  # Input dropout
  tensor = tf.nn.dropout(data, rate=dropouts[0])

  # Conv
  with tf.variable_scope('1-conv_maxout'):
    tensor = units.conv_maxout_layer(tensor, filter_shape=(5, 5, 50),
        ch_size=5, strides=(1,1,1,1), padding='SAME', seed=seed)
    tensor_visualize = tensor

  # Conv
  with tf.variable_scope('2-conv_maxout'):
    tensor = units.conv_maxout_layer(tensor, filter_shape=(5, 5, 50),
        ch_size=5, strides=(1,1,1,1), padding='SAME', seed=seed)

  # Maxpooling
  tensor = tf.nn.max_pool(tensor, ksize=[1,3,3,1], strides=[1,2,2,1],
    padding='SAME')

  # Conv
  with tf.variable_scope('3-conv_maxout'):
    tensor = units.conv_maxout_layer(tensor, filter_shape=(5, 5, 20),
        ch_size=5, strides=(1,1,1,1), padding='SAME', seed=seed)

  # Maxpooling
  tensor = tf.nn.max_pool(tensor, ksize=[1,3,3,1], strides=[1,2,2,1],
    padding='SAME')

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1], seed=seed)

  # Maxout
  with tf.variable_scope('4-maxout'):
    tensor = tf.reshape(tensor, shape=(-1, np.prod(tensor.shape[1:])))
    tensor = units.maxout_layer(tensor, out_size=200, ch_size=10, seed=seed)

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1], seed=seed)

  # Dense
  with tf.variable_scope('5-dense'):
    tensor = units.dense_layer(tensor, out_size=10, seed=seed)

  logits = tf.identity(tensor, name='logits')

  # Debug
  with tf.name_scope('visualizations'):
    # NOTE: assuming batch_size >= 10
    tensor_visualize = tensor_visualize[:10,:,:,:3]
    tf.add_to_collection('VISUALIZATIONS', tensor_visualize[:10,:,:,:])

  return logits
