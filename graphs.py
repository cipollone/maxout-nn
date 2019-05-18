'''\
This module defines a tf.Graph for classification. Depending on the dataset,
inputs and net change. The models are clear in Tensorboard.
'''

import tensorflow as tf

import data
import nets.example_net
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
    errors: number of wrong predictions
    dropouts: list two dropout-rate placeholders for input and hidden units
    accuracy: accuracy tensor
    use_train_data: use this op to read the training set
    use_test_data: use this op to read the test set
  '''

  def __init__(self, dataset, batch=None):
    '''\
    Create the graph and save useful tensors.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.
      batch: Batch size in int, or None to use the full dataset. None by
        default.
    '''
    
    # Create new
    graph = tf.Graph()
    with graph.as_default():
    
      # Input block
      with tf.name_scope('input'):

        # Dataset objects
        data_train = data.dataset(dataset, 'train', batch)
        data_test = data.dataset(dataset, 'test')

        # Iterator for both datasets
        iterator = data.iterator(dataset)
        use_train_data = iterator.make_initializer(data_train,name='use_train')
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
          logits = nets.example_net.model(features, dropouts)
        else:
          raise ValueError(dataset + ' is not a valid dataset')
      
      # Output
      with tf.name_scope('out'):

        # Prediction
        probabilities = tf.nn.softmax(logits, axis=1)
        output = tf.argmax(probabilities, axis=1, output_type=tf.int32)

        # Loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits)

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
    self.errors = errors
    self.dropouts = dropouts
    self.accuracy = accuracy
    self.use_train_data = use_train_data
    self.use_test_data = use_test_data
