'''\
Definition of the maxout layer.
'''

# NOTE: using variables' default initializer


import tensorflow as tf


def maxout_layer(x, out_size, ch_size, seed=None, return_W=None):
  '''\
  Computes the maxout units for inputs x. This function defines tf opterations
  that compute the maxout layer: linear + max.
  It puts variables to be regularized in collection 'REGULARIZABLE_VARS' and
  'RENORMALIZABLE_VARS'.

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

  # Regularization and renormalization
  tf.add_to_collection('REGULARIZABLE_VARS', W)
  tf.add_to_collection('RENORMALIZABLE_VARS', W)

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
  It puts variables to be regularized in collection 'REGULARIZABLE_VARS' and
  'RENORMALIZABLE_VARS'.

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
  tf.add_to_collection('RENORMALIZABLE_VARS', W)

  # Affine maps (for whole batch)
  y = tf.matmul(x, W) + b

  return y


def conv_maxout_layer(x, filter_shape, ch_size, strides=None, padding=None,
        seed=None, return_W=None):
  '''\
  Performs a 2D convolution with 'filter' for each image in x, then applies 
  maxout. This function wraps tf.nn.conv2d.  However, the output tensor
  consists of the max of all 'ch_size' convolutions. No padding applied.
  It puts variables to be regularized in collection 'REGULARIZABLE_VARS' and
  'RENORMALIZABLE_VARS'.

  Args:
    x: input tensor of shape (batch, height, width, channels)
    filter_shape: shape of the convolution filter. Interpreted as:
      (f_height, f_width, out_channels).
      This creates f_height*f_width*channels*out_channels*ch_size parameters.
    strides: 1-D tensor of length 4. Stride along each dimension of 'x'.
      It should be like [1,*,*,1]. 'None' means [1,1,1,1].
    padding: 'SAME' or 'VALID'. see tf.nn.conv2d. 'SAME' by default.
    seed: seed for deterministic initialization of variables.
    return_W: if true, returns both the output and the W parameters.

  Returns:
    a tensor in output of shape (batch, out_height, out_width, out_channels)
  '''

  # Defaults
  if not strides:
    strides = [1,1,1,1]
  if not padding:
    padding = 'SAME'

  # Initializer
  init = tf.glorot_uniform_initializer(seed) if seed else None

  # Parameters
  filter_shape = [filter_shape[0], filter_shape[1], int(x.shape[-1]),
      filter_shape[2]*ch_size]
  W = tf.get_variable('W', shape=filter_shape, initializer=init)
  b = tf.get_variable('b', shape=(filter_shape[-1],), initializer=init)

  # Regularization
  tf.add_to_collection('REGULARIZABLE_VARS', W)
  tf.add_to_collection('RENORMALIZABLE_VARS', W)

  # Convolution
  z = tf.nn.conv2d(x, filter=W, strides=strides, padding=padding,
      data_format='NHWC')
  z = z + b

  # Max along ch_size axis
  shape = ([-1] + z.shape[1:3].as_list() +
      [int(z.shape[3]) // ch_size] + [ch_size])
  z = tf. reshape(z, shape=shape)

  out = tf.reduce_max(z, axis=-1)

  # Ret
  if return_W:
    return (out, W)
  else:
    return out
