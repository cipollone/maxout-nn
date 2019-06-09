'''\
NN for the MNIST dataset.
'''

# TODO: testing convolutional layer. Do not save this net


import tensorflow as tf
import numpy as np

from . import units


def model(data, dropouts, seed=None):
  '''\
  FF Neural network for the MNIST dataset. Fully connected layers + softmax.
  Using maxout units and dropouts.

  Args:
    data: tensor of input features (batch_size, 784 pixels/features)
    dropouts: two scalar tensors that contains the dropout rate for input and
      hidden units.
    seed: seed for deterministic initialization of variables.

  Return:
    logis: (batch_size, n_classes). n_classes=10
  '''
  
  # Input dropout
  tensor = tf.nn.dropout(data, rate=dropouts[0])

  # Layer 1
  with tf.variable_scope('1-conv_maxout'):
    tensor = tf.reshape(tensor, shape=(-1, 28, 28, 1))
    tensor = units.conv_maxout_layer(tensor, filter_shape=(10, 10, 50),
        ch_size=2, strides=(1,1,1,1), seed=seed)
    tensor_visualize = tensor

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1], seed=seed)

  # Layer 2
  with tf.variable_scope('2-conv_maxout'):
    tensor = units.conv_maxout_layer(tensor, filter_shape=(5, 5, 20),
        ch_size=5, strides=(1,2,2,1), seed=seed)

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1], seed=seed)

  # Layer 3
  with tf.variable_scope('3-conv_maxout'):
    tensor = units.conv_maxout_layer(tensor, filter_shape=(5, 5, 20),
        ch_size=20, strides=(1,2,2,1), seed=seed)

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1], seed=seed)

  # Layer 3
  with tf.variable_scope('3-linear'):
    tensor = tf.reshape(tensor, shape=(-1, np.prod(tensor.shape[1:])))
    tensor = units.dense_layer(tensor, out_size=10, seed=seed)

  logits = tf.identity(tensor, name='logits')

  # Debug
  with tf.name_scope('visualizations'):
    # NOTE: assuming batch_size >= 10
    tensor_visualize = tf.reduce_mean(tensor_visualize[:10,:,:,:], axis=-1)
    tensor_visualize = tf.expand_dims(tensor_visualize, -1)
    tf.add_to_collection('VISUALIZATIONS', tensor_visualize[:10,:,:,:])

  return logits
