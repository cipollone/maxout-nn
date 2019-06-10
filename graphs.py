'''\
This module defines a tf.Graph for classification. Depending on the dataset,
inputs and net change. The models are clear in Tensorboard.
'''

import tensorflow as tf
import numpy as np
import os

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
    normalization_ops: run these ops to normalize vectors in weight matrices
  '''

  def __init__(self, dataset, batch=None, seed=None, regularization=None,
      renormalization=None):
    '''\
    Create the graph and save useful tensors.

    Args:
      dataset: The name of the dataset to use. Each dataset has its own net.
      batch: Batch size in int, or None to use the full dataset. None by
        default.
      seed: constant seed for repeatable results.
      regularization: constant that is multuplied to the total variables
        regularization loss. None means no regularization.
      renormalization: if not none, normalization_ops renormalize all weight
        vectors if their norms exceed this value.
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
          logits = nets.example_net.model(features, dropouts, seed)
        elif dataset == 'mnist':
          logits = nets.mnist_net.model(features, dropouts, seed)
        elif dataset == 'cifar10':
          logits = nets.cifar10_net.model(features, dropouts, seed)
        else:
          raise ValueError(dataset + ' is not a valid dataset')
      
      # Output
      with tf.name_scope('out'):

        # Prediction
        probabilities = tf.nn.softmax(logits, axis=1)
        output = tf.argmax(probabilities, axis=1, output_type=tf.int32)

        # Loss block
        with tf.name_scope('loss'):

          # Regularization: l2
          regular_loss = 0
          size = 0
          for var in tf.get_collection('REGULARIZABLE_VARS'):
            size += np.prod(var.shape.as_list())
            regular_loss += tf.nn.l2_loss(var)

          regular_const = regularization / size if regularization else 0
          regular_loss = tf.multiply(regular_loss,regular_const,
              name='regularization')

          # Loss
          loss = tf.losses.sparse_softmax_cross_entropy(
              labels=labels, logits=logits)
          loss = loss + regular_loss

        # Errors and accuracy
        diff = tf.not_equal(output, labels)
        errors = tf.reduce_sum(tf.cast(diff, tf.int32))
        accuracy = 1 - errors/tf.size(diff)

      # Out
      output = tf.identity(output, name='predicted_label')
      loss = tf.identity(loss, name='loss')
      errors = tf.identity(errors, name='errors')
      accuracy = tf.identity(accuracy, name='accuracy')

      # Regularization: normalization
      normalization_ops = []
      for var in tf.get_collection('RENORMALIZABLE_VARS'):

        # Put these op next to vars
        scope = os.path.split(var.name)[0] + '/renormalization/'
        with tf.name_scope(scope):

          # Reshape if convolutional kernels
          is_convolutional = len(var.shape) > 3
          vectors = (var if not is_convolutional else tf.reshape(var,
              shape=(var.shape[0]*var.shape[1], *var.shape[2:]))) # collapse ij

          # Check length of vectors
          norms = tf.norm(vectors, axis=0)  # Vectors along the first dimension
          norms = tf.expand_dims(norms, axis=0)

          # Scale only if exceeding limit
          if not renormalization: renormalization = 1
          scale = tf.where(tf.greater(norms, renormalization),
              norms, tf.broadcast_to(1.0, norms.shape))

          # Apply
          scaling_op = var.assign(var / scale)
          normalization_ops.append(scaling_op)

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
    self.normalization_ops = normalization_ops


@namespacelike
class EmaVariables:
  '''\
  This class creates and ExponentialMovingAverage for all trainable variables
  (the model parameters). It also creates backup copies of all vars,
  so to swap the training values and the EMA estimages.  Both the original
  variables and the ExponentialMovingAverages(ema) should be saved in
  checkpoints (not the backups). Remember to call self.update_op to update
  averages every time the variables change. use_ema and use_original are two
  ops to switch beween averaged variables and original ones.
  initialize_backups is the initialization op to call once
  (global_variables_initializer does not work).
  NOTE: never call use_ema two times in a row: this would overwrite the backups
  and the original variables would be lost.
  '''

  def __init__(self, decay):
    '''\
    Creates copies and defines restore/save ops for all trainable Variables.

    Args:
      decay: decay rate of the average, in [0,1).
    '''

    # Create a moving average and save estimates
    self.vars = tf.trainable_variables()
    self.ema = tf.train.ExponentialMovingAverage(decay=decay)
    self.update_op = self.ema.apply(self.vars)
    self.averages = [self.ema.average(v) for v in self.vars]
    self.backup_vars = []

    # Ops
    use_ema_ops = []
    use_original_ops = []
    initializers = []

    # For each var in the model
    with tf.variable_scope('backups', initializer=tf.zeros_initializer):
      for var, average in zip(self.vars, self.averages):

        # Make a copy
        backup_var = tf.get_variable(name=var.name.split(':')[0],
            shape=var.shape, dtype=var.dtype, trainable=False, collections=[])
        self.backup_vars.append(backup_var)

        initializers.append(backup_var.assign(var))

        # Assign op
        with tf.control_dependencies([backup_var.assign(var)]):
          use_ema_ops.append(var.assign(average))

        # Restore op
        use_original_ops.append(var.assign(backup_var))

    # Output
    self.use_ema_variables = tf.group(use_ema_ops,
        name='use_ema_variables')
    self.use_training_variables = tf.group(use_original_ops,
        name='use_training_variables')
    self.initialize_backups = tf.group(initializers,
        name='backups_initializer')
