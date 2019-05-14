'''\
This module defines different tf.Graph(s) for classification, depending on the
dataset.
'''

import tensorflow as tf

import data
import nets.example_net


class CGraph:
  '''\
  An object of this class, when instantiated, creates at tf.Graph for
  classification. Most parts of the net may change, but the outer structure
  is the same. The members are useful tf objects:
    graph: a tf.Graph
    input_ph: the input placeholder
    labels_ph: the labels placeholder
    output: the predicted output
    loss: the loss
    errors: number of wrong predictions
    dropouts: list two dropout-rate placeholders for input and hidden units
    accuracy: accuracy tensor
  '''

  def __init__(self, dataset):
    '''\
    Create the graph and save useful tensors.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.
    '''
    
    # Create new
    graph = tf.Graph()
    with graph.as_default():
    
      # Input
      (input_ph, labels_ph) = data.placeholder(dataset)
    
      # Net
      with tf.variable_scope('net'):

        # Dropout placeholders
        dropouts = [tf.placeholder(tf.float32, shape=[], name=i+'_dropout')
            for i in ('input','hidden')]

        # Model
        if dataset == 'example':
          logits = nets.example_net.model(input_ph, dropouts)
        else:
          raise ValueError(dataset + ' is not a valid dataset')
      
      # Output
      with tf.name_scope('out'):

        # Prediction
        probabilities = tf.nn.softmax(logits, axis=1)
        output = tf.argmax(probabilities, axis=1, output_type=tf.int32)

        # Loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_ph, logits=logits)

        # Errors and accuracy
        diff = tf.not_equal(output, labels_ph)
        errors = tf.reduce_sum(tf.cast(diff, tf.int32))
        accuracy = 1 - errors/tf.size(diff)

      output = tf.identity(output, name='predicted_label')
      loss = tf.identity(loss, name='loss')
      errors = tf.identity(errors, name='errors')
      accuracy = tf.identity(accuracy, name='accuracy')

    # Save
    self.graph = graph
    self.input_ph = input_ph
    self.labels_ph = labels_ph
    self.output = output
    self.loss = loss
    self.errors = errors
    self.dropouts = dropouts
    self.accuracy = accuracy

