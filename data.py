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
    return _load_iris()
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
    return _placeholder_iris()
  else:
    raise ValueError(dataset + ' is not a valid dataset')


def _load_iris():
  '''\
  Loads and returns the iris dataset. See load().
  '''

  train_x = np.genfromtxt('datasets/IRIS/iris_train_features.csv',delimiter=',')
  train_y = np.genfromtxt('datasets/IRIS/iris_train_labels.csv',delimiter=',')
  test_x = np.genfromtxt('datasets/IRIS/iris_test_features.csv',delimiter=',')
  test_y = np.genfromtxt('datasets/IRIS/iris_test_labels.csv',delimiter=',')

  return ((train_x, train_y), (test_x, test_y))


def _placeholder_iris():
  '''\
  Returns input and output placeholders. See load().
  '''

  input_ph = tf.placeholder(tf.float32, shape=(None,4), name='input_features')
  output_ph = tf.placeholder(tf.int32, shape=(None,), name='output_labels')

  return (input_ph, output_ph)

