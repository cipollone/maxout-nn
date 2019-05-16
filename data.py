'''\
Dataset loading module. See maxout -h for a list of the supported datasets.
Using the same reinitializable iterator with train and test datasets.
'''

import numpy as np
import tensorflow as tf


def dataset(name, group):
  '''\
  Returns the Dataset. Both 'train' and 'test' dataset contain a group of
  elements: 'train' is repeated and shuffled, 'test' is the whole test set.

  Args:
    name: name of the dataset to use.
    group: 'train' or 'test'.

  Returns:
    a tf Dataset that contains groups of (features, label) pairs
  '''

  # Check
  if not group in ('train','test'):
    raise ValueError("Group must be 'train' or 'test'")

  # Create
  if name == 'example':
    (data, size) = _iris_dataset(group)
  else:
    raise ValueError(name + ' is not a valid dataset')

  # Input pipeline
  if group == 'test':
    data = data.batch(size)     # All toghether
  else:
    data = data.batch(120).repeat()

  return data


def iterator(name):
  '''\
  Returns a reinitializable tf Iterator for batches of the dataset 'name'. The
  shape of the objects returned depends on 'name'. The iterator has the first
  dimension unspecified, because that is the batch size. The iterator is
  created on a generic dataset of the correct type. It needs to be initialized
  with a dataset before use.

  Args:
    name: name of the dataset to use.

  Returns:
    a tf Iterator
  '''

  if name == 'example':
    return _iris_iterator()
  else:
    raise ValueError(name + ' is not a valid dataset')


def _iris_dataset(group):
  '''\
  See dataset().

  Returns:
    (dataset, n): dataset and number of samples available
  '''

  # Read files
  if group == 'train':
    x = np.genfromtxt('datasets/IRIS/iris_train_features.csv',delimiter=',')
    y = np.genfromtxt('datasets/IRIS/iris_train_labels.csv',delimiter=',')
  else:
    x = np.genfromtxt('datasets/IRIS/iris_test_features.csv',delimiter=',')
    y = np.genfromtxt('datasets/IRIS/iris_test_labels.csv',delimiter=',')

  # To TF
  n = y.size
  x = tf.constant(x, dtype=tf.float32, name='features')
  y = tf.constant(y, dtype=tf.int32, name='labels')
  
  # Return dataset
  data = tf.data.Dataset.from_tensor_slices((x,y))
  return (data, n)


def _iris_iterator():
  '''\
  See iterator().
  '''

  return tf.data.Iterator.from_structure(
      output_types = (tf.float32, tf.int32),
      output_shapes = ((None,4), (None,)),
      shared_name='Iris_iterator')

