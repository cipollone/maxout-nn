'''\
Dataset loading module. See maxout -h for a list of the supported datasets.
Using the same reinitializable iterator for all splits.
'''

import numpy as np
import tensorflow as tf
import struct


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
  else:
    raise ValueError(name + ' is not a valid dataset')


def _mnist_dataset(group):

  # This dataset is small enough to be loaded all at once
  
  # Read files
  if group == 'train':
    my_images, my_labels = _read_mnist(
        "datasets/MNIST/train-55k-images-idx3-ubyte",
        "datasets/MNIST/train-55k-labels-idx1-ubyte")
  elif group == 'val':
    my_images, my_labels = _read_mnist(
        "datasets/MNIST/val-5k-images-idx3-ubyte",
        "datasets/MNIST/val-5k-labels-idx1-ubyte")
  else:
    my_images, my_labels = _read_mnist(
        "datasets/MNIST/test-10k-images-idx3-ubyte",
        "datasets/MNIST/test-10k-labels-idx1-ubyte")

  # Normalize images in [0,1]
  my_images = my_images/255.0

  # To TF
  n = my_labels.shape[0]
  my_images = tf.constant(my_images, dtype=tf.float32, name='features')
  my_labels = tf.constant(my_labels, dtype=tf.int32, name='labels')

  data = tf.data.Dataset.from_tensor_slices((my_images, my_labels))
  return (data, n)


def _read_mnist(images_name, labels_name):

  with open(labels_name, 'rb') as lab:
    magic, n = struct.unpack('>II',lab.read(8))
    labels = np.fromfile(lab, dtype=np.uint8)

  with open(images_name, "rb") as img:
    magic, num, rows, cols = struct.unpack(">IIII",img.read(16))
    images = np.fromfile( img, dtype=np.uint8).reshape(len(labels), 784)

  ## image plot example
  #my_img = np.reshape(my_img,(28,28))
  #plt.imshow(my_img)
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
