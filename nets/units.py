'''\
Definition of the maxout layer.
'''

import tensorflow as tf


def maxout_layer(x, out_size, ch_size):
  '''\
  Computes the maxout units for inputs x. This function defines tf opterations
  that compute the maxout layer: linear + max.

  Args:
    x: input vectors. Shape (batch_size, n_features) (N x d in paper)
    out_size: output length (m in paper)
    ch_size: number of channels to use (k in paper)

  Returns:
    a tensor in output
  '''

  in_size = int(x.shape[1])

  # Parameters
  W = tf.get_variable('W', shape=(ch_size, in_size, out_size))
  b = tf.get_variable('b', shape=(ch_size, out_size))

  # Affine maps (multiply a whole batch in one step)
  z = tf.einsum('id,kdm->ikm', x, W) + b

  # Max
  out = tf.reduce_max(z, axis=1)

  return out
