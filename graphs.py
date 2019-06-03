'''\
This module defines a tf.Graph for classification. Depending on the dataset,
inputs and net change. The models are clear in Tensorboard.
'''

import tensorflow as tf

import data
import nets.example_net
import nets.mnist_net
import nets.cifar10_net
from tools import namespacelike


@namespacelike
class CGraph:
  '''\
  An object of this class, when instantiated, creates at tf.Graph for
  classification. Most parts of the net may change, but the outer structure
  is the same. The members are useful tf Tensors:
    graph: a tf.Graph
    output: the predicted output
    loss: the loss
    regular_loss: regulatization loss (this is also included in loss)
    errors: number of wrong predictions
    dropouts: list two dropout-rate placeholders for input and hidden units
    accuracy: accuracy tensor
    use_train_data: use this op to read the training set
    use_val_data: use this op to read the validation set
    use_test_data: use this op to read the test set
  '''

  def __init__(self, dataset, batch=None, seed=None, regularization=None):
    '''\
    Create the graph and save useful tensors.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.
      batch: Batch size in int, or None to use the full dataset. None by
        default.
      seed: constant seed for repeatable results.
      regularization: constant that is multuplied to the total variables
        regularization loss. None means no regularization.
    '''
    
    # Create new
    graph = tf.Graph()
    with graph.as_default():

      # Input block
      with tf.name_scope('input'):

        # Dataset objects
        data_train = data.dataset(dataset, 'train', batch, seed)
        data_val = data.dataset(dataset, 'val')
        data_test = data.dataset(dataset, 'test')

        # Iterator for both datasets
        iterator = data.iterator(dataset)
        use_train_data = iterator.make_initializer(data_train,name='use_train')
        use_val_data = iterator.make_initializer(data_val,name='use_val')
        use_test_data = iterator.make_initializer(data_test,name='use_test')

        (features, labels) = iterator.get_next(name='GetInput')

      features = tf.identity(features, name='features')
      labels = tf.identity(labels, name='labels')
    
      # Net
      with tf.variable_scope('net'):

        # Dropout placeholders
        dropouts = [tf.placeholder(tf.float32, shape=[], name=i+'_dropout')
            for i in ('input','hidden')]

        # Model
        if dataset == 'example':
          logits, size = nets.example_net.model(features, dropouts, seed)
        elif dataset == 'mnist':
          logits, size = nets.mnist_net.model(features, dropouts, seed)
        elif dataset == 'cifar10':
          logits, size = nets.cifar10_net.model(features, dropouts, seed)
        else:
          raise ValueError(dataset + ' is not a valid dataset')
      
      # Output
      with tf.name_scope('out'):

        # Prediction
        probabilities = tf.nn.softmax(logits, axis=1)
        output = tf.argmax(probabilities, axis=1, output_type=tf.int32)

        # Regularization
        with tf.name_scope('regularization'):
          if regularization == None: regularization = 0
          regular_const = regularization / size
          regular_loss = tf.reduce_sum(
              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
          regular_loss = regular_loss * regular_const

        # Loss
        with tf.name_scope('loss'):
          loss = tf.losses.sparse_softmax_cross_entropy(
              labels=labels, logits=logits)
          loss = loss + regular_loss

        # Errors and accuracy
        diff = tf.not_equal(output, labels)
        errors = tf.reduce_sum(tf.cast(diff, tf.int32))
        accuracy = 1 - errors/tf.size(diff)

      output = tf.identity(output, name='predicted_label')
      loss = tf.identity(loss, name='loss')
      errors = tf.identity(errors, name='errors')
      accuracy = tf.identity(accuracy, name='accuracy')

    # Save
    self.graph = graph
    self.output = output
    self.loss = loss
    self.regular_loss = regular_loss
    self.errors = errors
    self.dropouts = dropouts
    self.accuracy = accuracy
    self.use_train_data = use_train_data
    self.use_val_data = use_val_data
    self.use_test_data = use_test_data
