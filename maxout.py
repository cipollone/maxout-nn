#!/usr/bin/env python3

'''\
Main script file. Depending on the parameters, we can train or test with
different Maxout networks.
'''

# NOTE: using default variables initialization
# NOTE: using default variables regularization
# NOTE: not explaining how the dataset should be saved
# NOTE: what about validation set?

# TODO: debug batch. Training do not work well if batch < size. Are all batches
# of the same size?

import os
import argparse
import shutil
import tensorflow as tf

from graphs import CGraph


def training(args):
  '''\
  Training function. Saves checkpoints in models/<args.dataset>/model.

  Args:
    args: a namespace object. Run `maxout.py -h' for a list of the available
      options.
  '''

  print('| Training')

  # Start or continue? Check directories and set iteration range
  if not args.cont:
    # Start from scratch
    clear_saved(args.dataset)
    steps_range = range(1, args.steps + 1)
  else:
    # Continue
    if os.path.exists('logs/train') or os.path.exists('logs/test'):
      raise FileExistsError(
        "Move 'logs/train' and 'logs/test' if you want to continue.")
    with open('logs/last_step.txt') as step_f:
      last_step = int(step_f.read())
    steps_range = range(last_step+1, last_step+args.steps+1)

  # Instantiate the graph
  graph = CGraph(args.dataset, args.batch)

  # Use it
  with graph.graph.as_default():

    # Optimizer
    optimizer = select_optimizer(args)
    minimize = optimizer.minimize(graph.loss)

    # Logger
    tf.summary.scalar('loss', graph.loss)
    tf.summary.scalar('accuracy', graph.accuracy)
    summaries_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs/train', graph=graph.graph)
    test_writer = tf.summary.FileWriter('logs/test', graph=graph.graph)

    # Saver
    saver = tf.train.Saver(max_to_keep=3)

    # Run
    with tf.Session() as sess:

      # Initialize variables
      if not args.cont: # First time
        sess.run(tf.global_variables_initializer())
      else:             # Continue
        checkpoint = tf.train.latest_checkpoint(
            checkpoint_dir=os.path.join('models',args.dataset))
        saver.restore(sess, checkpoint)
        print('| Variables restored.')

      # Create contexts
      contexts = RunContexts(sess, train_set=graph.use_train_data,
          test_set=graph.use_test_data)

      # Main loop
      for step in steps_range:

        # Train
        with contexts.train_set:
          sess.run( minimize,
              feed_dict={
                graph.dropouts[0]: args.dropout[0],
                graph.dropouts[1]: args.dropout[1],
              })

        # Every log_every steps or at the end
        if step % args.log_every == 0 or step == steps_range.stop-1:

          # Test on train set and test set
          with contexts.train_set:
            train_loss, train_summaries = sess.run( (graph.loss, summaries_op),
                feed_dict={
                  graph.dropouts[0]: 0,
                  graph.dropouts[1]: 0,
                })
          with contexts.test_set:
            test_loss, test_summaries = sess.run( (graph.loss, summaries_op),
                feed_dict={
                  graph.dropouts[0]: 0,
                  graph.dropouts[1]: 0,
                })

          # Log
          print('| Step: ' + str(step) + ', train loss: ' + str(train_loss))
          train_writer.add_summary(train_summaries, step)
          test_writer.add_summary(test_summaries, step)

          # Save parameters
          model_name = 'model-step{}'.format(step)
          saver.save(sess, os.path.join('models',args.dataset,model_name))

          # Save step number
          with open('logs/last_step.txt', 'wt') as step_f:
            step_f.write(str(step))
          


def testing(args):
  '''\
  Test the performances of the net on the test set. Creates the tf Graph for
  testing, loads the weights and evaluates the predictions. Parameters are
  loaded from last checkpoint: models/<args.dataset>/model.
  
  Args:
    args: a namespace object. Run `maxout.py -h' for a list of the available
      options.
  '''

  print('| Testing')

  # Instantiate the graph
  graph = CGraph(args.dataset)

  # Use it
  with graph.graph.as_default():
    
    # Create a Saver
    saver = tf.train.Saver()

    # Run
    with tf.Session() as sess:

      # Restore parameters
      checkpoint = tf.train.latest_checkpoint(
          checkpoint_dir=os.path.join('models',args.dataset))
      saver.restore(sess, checkpoint)

      # Create context
      contexts = RunContexts(sess, test_set=graph.use_test_data)

      # Predict
      with contexts.test_set:
        output,loss,errors = sess.run( (graph.output,graph.loss,graph.errors),
            feed_dict={
              graph.dropouts[0]: 0,
              graph.dropouts[1]: 0,
            })

      # Out
      print('| Predicted:', output)
      print('| Loss:', loss)
      print('| Errors:', errors)


def debug(args):
  '''\
  Debugging
  '''

  # Prints
  print('| Debug')


  import pdb
  pdb.set_trace()


def clear_saved(dataset):
  '''\
  Removes all files from 'models/<dataset>/', 'logs/train' and 'logs/test'.
  Ask for confirmation at terminal.

  Args:
    dataset: The name of the dataset to use.
  '''

  # Confirm
  print('| Clearing previous savings. Hit enter to confirm.')
  input()
  
  # To remove
  join = os.path.join
  model = join('models', dataset)
  logs = [join('logs',x) for x in ['train','test','debug']]
  paths = [model] + logs

  # Rm
  for p in paths:
    if os.path.exists(p):
      shutil.rmtree(p)


def select_optimizer(args):
  '''\
  Returns a tf optimizer, initialized with options. See the tf api of these
  optimizers to see what options are available for each one.
  
  Args:
    args: namespace of options with these fields:
      rate: learning rate
      optimizer: identifier of the optimizer to use. Choices:
          gd: GradientDescentOptimizer
          rms: RMSPropOptimizer
          adagrad: AdagradOptimizer
          adadelta: AdadeltaOptimizer. Use rate=1 and other options.
          adam: AdamOptimizer
      parameters: a list of `opt=val' options to pass to the constructor.
        val is numeric.

  Returns:
    a tf.train.Optimizer
  '''

  # Extract parameters
  params = dict()
  if args.parameters:
    for p in args.parameters:
      key, val = p.split('=',maxsplit=1)
      params[key.strip()] = float(val)

  # Select optimizer
  if args.optimizer == 'gd':
    opt = tf.train.GradientDescentOptimizer(args.rate, **params)
  elif args.optimizer == 'rms':
    opt = tf.train.RMSPropOptimizer(args.rate, **params)
  elif args.optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(args.rate, **params)
  elif args.optimizer == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(args.rate, **params)
  elif args.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(args.rate, **params)
  else:
    raise ValueError(args.optimizer+
      ' is not an optimizer. See help(maxout.select_optimizer)')

  print('| Using', opt.get_name())
  return opt


class RunContexts:
  '''\
  Create context managers for different runs of a Session. This class can be
  used as:
    cs = RunContexts(sess, train=init_training, test=init_testing, etc...)
    with cs.train:
      # Training
  It generates a context manager for each keyword argument in input. Each 
  context is entered but never left, so that successive `with cs.train' do not
  call the init_training op.
  '''

  def __init__(self, sess, **contexts):
    '''\
    See class description.

    Args:
      sess: tf Session. Must be active when using the contexts.
      key=val: context named 'key' with initialization op 'val'.
    '''

    self._sess = sess
    for name in contexts:
      self.__dict__[name] = self._RunContext(self, name, contexts[name])
    self._current = None

  class _RunContext:
    '''\
    The real context manager class. Internal class: do not use it directly.
    '''

    def __init__(self, allContexts, name, op):
      self.all = allContexts
      self.op = op
      self.name = name

    def __enter__(self):
      if self.all._current != self:
        self.all._sess.run(self.op)
        self.all._current = self

    def __exit__(self, exc_type, exc_value, exc_tb):
      pass


def main():
  '''\
  Main function. Called when this file is executed as script.
  '''

  # Defaults
  learning_rate = 0.05
  n_steps = 200
  log_every = 20
  optimizer = 'rms'

  ## Parsing arguments
  parser = argparse.ArgumentParser(description='Training and testing with\
      the Maxout network')
  parser.add_argument('op', choices=['train','test','debug'],
      help='What to do with the net. Most options only affect training.')
  parser.add_argument('-d', '--dataset', default='example',
      choices=['example'], help='Which dataset to load')
  parser.add_argument('-r', '--rate', type=float, default=learning_rate,
      help='Learning rate / step size. Depends on the optimizer.')
  parser.add_argument('-s', '--steps', type=int, default=n_steps,
      help='Number of steps of the optimization')
  parser.add_argument('-l', '--log_every', type=int, default=log_every,
      help='Interval of number of steps between logs/saved models')
  parser.add_argument('-o', '--optimizer', default=optimizer,
      choices=['gd', 'rms', 'adagrad', 'adadelta', 'adam'],
      help='Name of the optimizer to use.\
          See `help(maxout.select_optimizer)\' to know more.')
  parser.add_argument('-p', '--parameters',
      nargs='+', metavar='PARAMETER',
      help='If the optimizer needs other arguments than just --rate,\
      use this option. One or more `opt=val\' for any opt argument of the\
      optimizer selected (see tf doc). val is assumed numeric.')
  parser.add_argument('-c', '--continue', action='store_true', dest='cont',
      help='Loads most recent saved model and resumes training from there.\
          Continue with the same optimizer.')
  parser.add_argument('--dropout', type=float, nargs=2, metavar=('input_rate',
      'hidden_rate'),
      help='Dropout probability: drop probability for input and hidden units.')
  parser.add_argument('-b', '--batch', type=int, 
      help='Batch size. Without this parameter, the whole dataset is used.')

  args = parser.parse_args()

  # Small checks for dropout
  if not args.dropout:
    args.dropout = (0,0) # Keep everything
  else:
    if not (0 <= args.dropout[0] <= 1 and 0 <= args.dropout[1] <= 1):
      raise ValueError(
        '--dropout arguments is not a probability. It must be in [0, 1].')
    if args.op == 'test':
      print('Warning: dropout argument is not used in testing.')


  # Go
  if args.op == 'train':
    training(args)
  elif args.op == 'test':
    testing(args)
  elif args.op == 'debug':
    debug(args)



if __name__ == '__main__':
  main()
