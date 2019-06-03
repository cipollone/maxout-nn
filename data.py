'''\
Dataset loading module. See maxout -h for a list of the supported datasets.
Using the same reinitializable iterator for all splits.
'''

import numpy as np
import tensorflow as tf
import struct

from tools import with_persistent_vars


def dataset(name, group, batch=None, seed=None):
  '''\
  Returns the Dataset. 'train', 'val' and 'test' datasets contain the same type
  of elements. However, 'train' is also repeated and shuffled. 'val' and 'test'
  are always returned as whole, 'train' can be divided in batches.

  Args:
    name: name of the dataset to use.
    group: 'train' 'val' or 'test'.
    batch: how many samples to return. If None, the entire dataset is returned.
    seed: seed for deterministic dataset permutation.

  Returns:
    a tf Dataset that contains groups of (features, label) pairs
  '''

  # Check
  if not group in ('train','val','test'):
    raise ValueError("Group must be 'train', 'val' or 'test'")

  # Create
  if name == 'example':
    (data, size) = _iris_dataset(group)
  elif name == 'mnist':
    (data, size) = _mnist_dataset(group)
  elif name == 'cifar10':
    (data, size) = _cifar10_dataset(group)
  else:
    raise ValueError(name + ' is not a valid dataset')
  
  # Select batch
  if not batch or batch < 1 or batch > size :
    batch = size

  # Input pipeline
  if group == 'train':
    data = data.shuffle(min(size, 10000), seed=seed) # shuffle up to 10000
    data = data.repeat()                  # infinite repetition
    data = data.batch(batch)              # batches
    data = data.prefetch(1)               # also fetch next batch
  else:
    data = data.batch(size)               # all toghether

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
  elif name == 'mnist':
    return _mnist_iterator()
  elif name == 'cifar10':
    return _cifar10_iterator()
  else:
    raise ValueError(name + ' is not a valid dataset')


@with_persistent_vars(mean=None, std=None)
def _mnist_dataset(group):

  # This dataset is small enough to be loaded all at once
  
  # Read files
  if group == 'train':
    images, labels = _read_mnist(
        "datasets/MNIST/train-55k-images-idx3-ubyte",
        "datasets/MNIST/train-55k-labels-idx1-ubyte")
  elif group == 'val':
    images, labels = _read_mnist(
        "datasets/MNIST/val-5k-images-idx3-ubyte",
        "datasets/MNIST/val-5k-labels-idx1-ubyte")
  else:
    images, labels = _read_mnist(
        "datasets/MNIST/test-10k-images-idx3-ubyte",
        "datasets/MNIST/test-10k-labels-idx1-ubyte")

  # Standardization
  if not _mnist_dataset.mean: # compute same for all splits
    _mnist_dataset.mean = np.mean(images)
    _mnist_dataset.std = np.std(images)

  images = (images - _mnist_dataset.mean) / _mnist_dataset.std

  # To TF
  n = labels.shape[0]
  images = tf.constant(images, dtype=tf.float32, name='features')
  labels = tf.constant(labels, dtype=tf.int32, name='labels')

  data = tf.data.Dataset.from_tensor_slices((images, labels))
  return (data, n)


def _read_mnist(images_name, labels_name):

  with open(labels_name, 'rb') as lab:
    magic, n = struct.unpack('>II',lab.read(8))
    labels = np.fromfile(lab, dtype=np.uint8)

  with open(images_name, "rb") as img:
    magic, num, rows, cols = struct.unpack(">IIII",img.read(16))
    images = np.fromfile( img, dtype=np.uint8).reshape(len(labels), 784)

  ## image plot example
  #img = np.reshape(img,(28,28))
  #plt.imshow(img)
  #plt.show()

  return images,labels


def _mnist_iterator():
  '''\
  See iterator().
  '''

  return tf.data.Iterator.from_structure(
      output_types = (tf.float32, tf.int32),
      output_shapes = ((None,784), (None,)),
      shared_name='MNIST_iterator')


@with_persistent_vars(mean=None, std=None)
def _cifar10_dataset(group):

  # This dataset is small enough to be loaded all at once
  
  # Read files
  if group == 'train':
    images, labels = _read_cifar10("datasets/CIFAR-10/train_data.bin")
  elif group == 'val':
    images, labels = _read_cifar10("datasets/CIFAR-10/val_data.bin")
  else:
    images, labels = _read_cifar10("datasets/CIFAR-10/test.bin")

  # Standardization
  if not _cifar10_dataset.mean: # compute same for all splits
    _cifar10_dataset.mean = np.mean(images)
    _cifar10_dataset.std = np.std(images)

  images = (images - _cifar10_dataset.mean) / _cifar10_dataset.std

  # To TF
  n = labels.shape[0]
  images = tf.constant(images, dtype=tf.float32, name='features')
  labels = tf.constant(labels, dtype=tf.int32, name='labels')

  data = tf.data.Dataset.from_tensor_slices((images, labels))
  return (data, n)


def _read_cifar10(data_path):

  # Load dataset
  dataset = np.fromfile(data_path, dtype=np.uint8)
  dataset = np.reshape(dataset, (-1, 1+3*32*32))   # label+channel*row*col
  images = dataset[:,1:]
  labels = dataset[:,0]

  # Reshape images as 3D tensors
  images = np.reshape(images, (-1, 3, 32, 32))     # batch, chs, rows, cols
  images = np.transpose(images, (0, 2, 3, 1))      # batch, rows, cols, chs
  
  return images, labels


def _cifar10_iterator():
  '''\
  See iterator().
  '''

  return tf.data.Iterator.from_structure(
      output_types = (tf.float32, tf.int32),
      output_shapes = ((None,32,32,3), (None,)),
      shared_name='cifar10_iterator')


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
  elif group == 'val':
    x = np.genfromtxt('datasets/IRIS/iris_val_features.csv',delimiter=',')
    y = np.genfromtxt('datasets/IRIS/iris_val_labels.csv',delimiter=',')
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
