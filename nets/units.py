'''\
Definition of the maxout layer.
'''

# NOTE: using variables' default initializer


import tensorflow as tf


def maxout_layer(x, out_size, ch_size, seed=None, return_W=None):
  '''\
  Computes the maxout units for inputs x. This function defines tf opterations
  that compute the maxout layer: linear + max.
  It puts variables to be regularized in collection 'REGULARIZABLE_VARS'.

  Args:
    x: input vectors. Shape (batch_size, n_features) (N x d in paper)
    out_size: output length (m in paper)
    ch_size: number of channels to use (k in paper)
    seed: seed for deterministic initialization of variables.
    return_W: if true, returns both the output and the W parameters.

  Returns:
    a tensor in output
  '''

  in_size = int(x.shape[1])
  
  # Initializer
  init = tf.glorot_uniform_initializer(seed) if seed else None

  # Parameters
  W = tf.get_variable('W', shape=(in_size, ch_size, out_size),
      initializer=init)
  b = tf.get_variable('b', shape=(ch_size, out_size), initializer=init)

  # Regularization should only affect W
  tf.add_to_collection('REGULARIZABLE_VARS', W)

  # Affine maps (multiply a whole batch in one step)
  z = tf.einsum('id,dkm->ikm', x, W) + b

  # Max
  out = tf.reduce_max(z, axis=1)

  # Ret
  if return_W:
    return (out, W)
  else:
    return out


def dense_layer(x, out_size, seed=None):
  '''\
  Affine function of given dimensions. No activation function.
  We're not using keras.layers.Dense because it's easy to write, and to be
  consistent with maxout_layer.
  It puts variables to be regularized in collection 'REGULARIZABLE_VARS'.

  Args:
    x: input vectors. Shape (batch_size, n_features)
    out_size: number of units
    seed: seed for deterministic initialization of variables.

  Returns:
    a tensor in output
  '''

  in_size = int(x.shape[1])
  
  # Initializer
  init = tf.glorot_uniform_initializer(seed) if seed else None

  # Parameters
  W = tf.get_variable('W', shape=(in_size, out_size), initializer=init)
  b = tf.get_variable('b', shape=(out_size), initializer=init)

  # Regularization should only affect W
  tf.add_to_collection('REGULARIZABLE_VARS', W)

  # Affine maps (for whole batch)
  y = tf.matmul(x, W) + b

  return y
