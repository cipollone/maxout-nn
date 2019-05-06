'''\
This module contains the definition of the tf.Graph used for training.
'''

import tensorflow as tf

import nets.example_net
import data


class MaxoutGraph:
  '''\
  An object of this class, when instantiated, creates all ops required for
  the network.
  The graph can be accessed from:
    self.training_graph
  self.input is the input placeholder, self.output is the output tensor
  '''

  def __init__(self, dataset):
    '''\
    Create the graph (for training) and store it internally.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.
    '''
    
    # Training graph
    self.training_graph, self.input, self.output = \
        self._create_traning(dataset)


  def _create_traning(self, dataset):
    '''\
    Internal method. Defines all ops required for training.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.

    Returns:
      (graph, in, out): tuple with the tf.Graph, intput and output tensors.
    '''

    # Create new
    graph = tf.Graph()
    with graph.as_default():
    
      # Input
      input_ph = data.placeholder(dataset)
    
      # Net
      with tf.variable_scope('net'):
        output = nets.example_net.model(input_ph)
    
    # Interface of this graph
    input_tensor = input_ph
    output_tensor = tf.identity(output, name='output')

    return (graph, input_tensor, output_tensor)

