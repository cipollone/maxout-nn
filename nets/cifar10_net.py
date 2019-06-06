'''\
NN for the CIFAR-10 dataset.
'''

import tensorflow as tf

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

  # NOTE: this is not the real net: just testing the dataset for now.

  # Reshape to use the classic 1D maxout layer
  tensor = tf.reshape(data, (-1, 32*32*3))

  # Sizes
  d = 32*32*3
  k1 = 30
  m1 = 20
  k2 = 2
  m2 = 10

  # Input dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[0])

  # Layer 1
  with tf.variable_scope('1-maxout'):
    (tensor,W1) = units.maxout_layer(tensor, out_size=m1, ch_size=k1,
        seed=seed, return_W=True)

  # Dropout
  tensor = tf.nn.dropout(tensor, rate=dropouts[1])

  # Layer 2
  with tf.variable_scope('2-maxout'):
    tensor = units.maxout_layer(tensor, out_size=m2, ch_size=k2, seed=seed)

  logits = tf.identity(tensor, name='logits')

  # Debug
  with tf.name_scope('visualizations'):
    tensor = tf.reshape(tf.reduce_mean(W1, axis=1), shape=(32, 32, 3, -1))
    tensor = tf.transpose(tensor, perm=(3, 0, 1, 2))
    tf.add_to_collection('W_visualization', tensor)

  return logits
