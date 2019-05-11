'''\
Dataset loading module. Dataset suppported MNIST ans CIFAR-10.
'''
# TODO: modify this file to load the real dataset.
# See the online documentation of tf.data module to see how to do this
# There are mainly two alternatives: placeholders and iterators.
# This tiny dataset can be used all at once. However, bigger datasets often
# needs to be loaded and used in blocks `batches' because they are too big for
# RAM (or maybe because training is improved). You should add an option to
# specify batch size. Also, we could use batchsize=Null to indicate the
# whole dataset.
#
# First download the MNIST and CIFAR-10 dataet in two folders:
# ./datasets/mnist/ and ./datasets/cifar-10/. Please, write what you did to
# prepare them, because i'll need to do the same.
# Feel free do delete any funcion in this file and modify as you need
# (everything is saved, anyway).
#
# We could use functions, as I did here in this simple case, or use classes.
# For example:
# --
# get_dataset(dataset): return Dataset(dataset)
# class Dataset:
#   def __init__(self,dataset,batchsize):
#     # Set the information for the correct dataset
#   def get_next_batch(self):
#     '''return next batch of data'''
#   def placeholder(self):
#     '''return placeholder'''
# --
# or similar.. (this would be different with iterators)
#
# Use docstrings comments with similar format:
#   ''' The strings just below every def or class'''
# Avoid global variables, use classes if they need to change


import numpy as np
import tensorflow as tf


def load(dataset):
  '''\
  Returns the dataset.

  Args:
    dataset: name of the dataset to load.

  Returns:
    ((train_x, train_y), (test_x, test_y)): numpy arrays
  '''
  if dataset == 'example':
    return ((iris_train_x, iris_train_y), (iris_test_x, iris_test_y))
  else:
    raise ValueError(dataset + ' is not a valid dataset')


def placeholder(dataset):
  '''\
  Returns a tf.placeholder for the input and output tensors for each dataset.
  The first dimension represents the batch size and it's left unspecified.

  Args:
    dataset: name of the dataset to load.

  Returns:
    (input_placeholder, output_placeholder)
  '''
  if dataset == 'example':
    input_ph = tf.placeholder(tf.float32, shape=(None,)+iris_test_x.shape[1:],
        name='input_features')
    output_ph = tf.placeholder(tf.int32, shape=(None,), name='output_labels')
    return (input_ph, output_ph)
  else:
    raise ValueError(dataset + ' is not a valid dataset')


# NOTE: demo dataset, written here for simplicity.
# Test dataset
iris_test_x = np.array([
  [5.9,3.0,4.2,1.5],
  [6.9,3.1,5.4,2.1],
  [5.1,3.3,1.7,0.5],
  [6.0,3.4,4.5,1.6],
  [5.5,2.5,4.0,1.3],
  [6.2,2.9,4.3,1.3],
  [5.5,4.2,1.4,0.2],
  [6.3,2.8,5.1,1.5],
  [5.6,3.0,4.1,1.3],
  [6.7,2.5,5.8,1.8],
  [7.1,3.0,5.9,2.1],
  [4.3,3.0,1.1,0.1],
  [5.6,2.8,4.9,2.0],
  [5.5,2.3,4.0,1.3],
  [6.0,2.2,4.0,1.0],
  [5.1,3.5,1.4,0.2],
  [5.7,2.6,3.5,1.0],
  [4.8,3.4,1.9,0.2],
  [5.1,3.4,1.5,0.2],
  [5.7,2.5,5.0,2.0],
  [5.4,3.4,1.7,0.2],
  [5.6,3.0,4.5,1.5],
  [6.3,2.9,5.6,1.8],
  [6.3,2.5,4.9,1.5],
  [5.8,2.7,3.9,1.2],
  [6.1,3.0,4.6,1.4],
  [5.2,4.1,1.5,0.1],
  [6.7,3.1,4.7,1.5],
  [6.7,3.3,5.7,2.5],
  [6.4,2.9,4.3,1.3]
])
iris_test_y = np.array([
  1, 2, 0, 1, 1, 1, 0, 2, 1, 2, 2, 0, 2, 1, 1, 0, 1, 0, 0, 2, 0, 1, 2, 1, 1, 1,
  0, 1, 2, 1
])

# Training dataset
iris_train_x = np.array([
  [6.4,2.8,5.6,2.2],
  [5.0,2.3,3.3,1.0],
  [4.9,2.5,4.5,1.7],
  [4.9,3.1,1.5,0.1],
  [5.7,3.8,1.7,0.3],
  [4.4,3.2,1.3,0.2],
  [5.4,3.4,1.5,0.4],
  [6.9,3.1,5.1,2.3],
  [6.7,3.1,4.4,1.4],
  [5.1,3.7,1.5,0.4],
  [5.2,2.7,3.9,1.4],
  [6.9,3.1,4.9,1.5],
  [5.8,4.0,1.2,0.2],
  [5.4,3.9,1.7,0.4],
  [7.7,3.8,6.7,2.2],
  [6.3,3.3,4.7,1.6],
  [6.8,3.2,5.9,2.3],
  [7.6,3.0,6.6,2.1],
  [6.4,3.2,5.3,2.3],
  [5.7,4.4,1.5,0.4],
  [6.7,3.3,5.7,2.1],
  [6.4,2.8,5.6,2.1],
  [5.4,3.9,1.3,0.4],
  [6.1,2.6,5.6,1.4],
  [7.2,3.0,5.8,1.6],
  [5.2,3.5,1.5,0.2],
  [5.8,2.6,4.0,1.2],
  [5.9,3.0,5.1,1.8],
  [5.4,3.0,4.5,1.5],
  [6.7,3.0,5.0,1.7],
  [6.3,2.3,4.4,1.3],
  [5.1,2.5,3.0,1.1],
  [6.4,3.2,4.5,1.5],
  [6.8,3.0,5.5,2.1],
  [6.2,2.8,4.8,1.8],
  [6.9,3.2,5.7,2.3],
  [6.5,3.2,5.1,2.0],
  [5.8,2.8,5.1,2.4],
  [5.1,3.8,1.5,0.3],
  [4.8,3.0,1.4,0.3],
  [7.9,3.8,6.4,2.0],
  [5.8,2.7,5.1,1.9],
  [6.7,3.0,5.2,2.3],
  [5.1,3.8,1.9,0.4],
  [4.7,3.2,1.6,0.2],
  [6.0,2.2,5.0,1.5],
  [4.8,3.4,1.6,0.2],
  [7.7,2.6,6.9,2.3],
  [4.6,3.6,1.0,0.2],
  [7.2,3.2,6.0,1.8],
  [5.0,3.3,1.4,0.2],
  [6.6,3.0,4.4,1.4],
  [6.1,2.8,4.0,1.3],
  [5.0,3.2,1.2,0.2],
  [7.0,3.2,4.7,1.4],
  [6.0,3.0,4.8,1.8],
  [7.4,2.8,6.1,1.9],
  [5.8,2.7,5.1,1.9],
  [6.2,3.4,5.4,2.3],
  [5.0,2.0,3.5,1.0],
  [5.6,2.5,3.9,1.1],
  [6.7,3.1,5.6,2.4],
  [6.3,2.5,5.0,1.9],
  [6.4,3.1,5.5,1.8],
  [6.2,2.2,4.5,1.5],
  [7.3,2.9,6.3,1.8],
  [4.4,3.0,1.3,0.2],
  [7.2,3.6,6.1,2.5],
  [6.5,3.0,5.5,1.8],
  [5.0,3.4,1.5,0.2],
  [4.7,3.2,1.3,0.2],
  [6.6,2.9,4.6,1.3],
  [5.5,3.5,1.3,0.2],
  [7.7,3.0,6.1,2.3],
  [6.1,3.0,4.9,1.8],
  [4.9,3.1,1.5,0.1],
  [5.5,2.4,3.8,1.1],
  [5.7,2.9,4.2,1.3],
  [6.0,2.9,4.5,1.5],
  [6.4,2.7,5.3,1.9],
  [5.4,3.7,1.5,0.2],
  [6.1,2.9,4.7,1.4],
  [6.5,2.8,4.6,1.5],
  [5.6,2.7,4.2,1.3],
  [6.3,3.4,5.6,2.4],
  [4.9,3.1,1.5,0.1],
  [6.8,2.8,4.8,1.4],
  [5.7,2.8,4.5,1.3],
  [6.0,2.7,5.1,1.6],
  [5.0,3.5,1.3,0.3],
  [6.5,3.0,5.2,2.0],
  [6.1,2.8,4.7,1.2],
  [5.1,3.5,1.4,0.3],
  [4.6,3.1,1.5,0.2],
  [6.5,3.0,5.8,2.2],
  [4.6,3.4,1.4,0.3],
  [4.6,3.2,1.4,0.2],
  [7.7,2.8,6.7,2.0],
  [5.9,3.2,4.8,1.8],
  [5.1,3.8,1.6,0.2],
  [4.9,3.0,1.4,0.2],
  [4.9,2.4,3.3,1.0],
  [4.5,2.3,1.3,0.3],
  [5.8,2.7,4.1,1.0],
  [5.0,3.4,1.6,0.4],
  [5.2,3.4,1.4,0.2],
  [5.3,3.7,1.5,0.2],
  [5.0,3.6,1.4,0.2],
  [5.6,2.9,3.6,1.3],
  [4.8,3.1,1.6,0.2],
  [6.3,2.7,4.9,1.8],
  [5.7,2.8,4.1,1.3],
  [5.0,3.0,1.6,0.2],
  [6.3,3.3,6.0,2.5],
  [5.0,3.5,1.6,0.6],
  [5.5,2.6,4.4,1.2],
  [5.7,3.0,4.2,1.2],
  [4.4,2.9,1.4,0.2],
  [4.8,3.0,1.4,0.1],
  [5.5,2.4,3.7,1.0]
])
iris_train_y = np.array([
  2, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 1, 0, 0, 2, 1, 2, 2, 2, 0, 2, 2, 0, 2, 2, 0,
  1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 1,
  1, 0, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 0, 2, 2, 0, 0, 1, 0, 2, 2, 0, 1, 1,
  1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 0, 2, 1, 0, 0, 2, 0, 0, 2, 1, 0, 0, 1, 0, 1,
  0, 0, 0, 0, 1, 0, 2, 1, 0, 2, 0, 1, 1, 0, 0, 1
])
