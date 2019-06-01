'''\
NN for the MNIST dataset.
'''

import tensorflow as tf

from . import units


def model(data, dropouts, seed=None):
  '''\
  FF Neural network for the MNIST dataset. Fully connected layers + softmax.
  Using maxout units and dropouts.

  Args:
    data: tensor of input features (batch_size, n_features)
    dropouts: two scalar tensors that contains the dropout rate for input and
      hidden units.
    seed: seed for deterministic initialization of variables.
  '''

  # Input dropout
  tensor = tf.nn.dropout(data, keep_prob=1-dropouts[0])

  # Layer 1
  with tf.variable_scope('1-maxout'):
    (tensor,W1) = units.maxout_layer(tensor, out_size=100, ch_size=20,
        seed=seed, return_W=True)

  # Dropout
  tensor = tf.nn.dropout(tensor, keep_prob=1-dropouts[1])

  # Layer 2
  with tf.variable_scope('2-maxout'):
    tensor = units.maxout_layer(tensor, out_size=10, ch_size=10, seed=seed)

  logits = tf.identity(tensor, name='logits')

  # Debug
  tensor = tf.reshape(tf.reduce_mean(W1, axis=0), shape=(28,28,-1,1))
  tensor = tf.transpose(tensor, perm=(2,0,1,3))
  tf.add_to_collection('W_visualization', tensor)

  return logits
