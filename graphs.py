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
    input: the input placeholder
    output: the predicted output
    loss: the loss
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
        if dataset == 'example':
          logits = nets.example_net.model(input_ph)
        else:
          raise ValueError(dataset + ' is not a valid dataset')
      
      # Output
      with tf.name_scope('out'):

        # Prediction
        probabilities = tf.nn.softmax(logits, axis=1)
        output = tf.argmax(probabilities, axis=1)

        # Loss
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels_ph, logits=logits)

      output = tf.identity(output, name='predicted_label')
      loss = tf.identity(loss, name='loss')

    # Save
    self.graph = graph
    self.input = input_ph
    self.output = output
    self.loss = loss

