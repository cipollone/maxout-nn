#!/usr/bin/env python3

'''\
Main script file. Depending on the parameters, we can train or test with
different Maxout networks.
'''


import os
import argparse
import shutil
import tensorflow as tf
import numpy as np

from graphs import CGraph, EmaVariables
from tools import RunContexts


def training(args):
  '''\
  Training function. Saves checkpoints in models/<args.dataset>/model.

  Args:
    args: a namespace object. Run `maxout.py -h' for a list of the available
      options.
  '''

  # Prints
  print('| Training')
  print('| Dataset:', args.dataset, flush=True)

  # Start or continue? Check directories and set iteration range
  if not args.cont:
    # Start from scratch
    _clear_saved(args.dataset)
    steps_range = range(1, args.steps + 1)
  else:
    # Continue
    if os.path.exists('logs/train') or os.path.exists('logs/val'):
      raise FileExistsError(
        "Move 'logs/train' and 'logs/val' if you want to continue.")
    with open('logs/last_step.txt') as step_f:
      last_step = int(step_f.read())
    steps_range = range(last_step+1, last_step+args.steps+1)

  # Instantiate the graph
  graph = CGraph(args.dataset, args.batch, args.seed, args.regularization,
      args.renormalization)

  # Use it
  with graph.graph.as_default():

    # Constant seed for debugging
    if args.seed:
      tf.set_random_seed(args.seed)

    # Optimizer
    optimizer = _select_optimizer(args)
    minimize = optimizer.minimize(graph.loss)

    # Tensorboard summaries
    tf.summary.scalar('loss', graph.loss)
    tf.summary.scalar('accuracy', graph.accuracy)
    val_summaries_op = tf.summary.merge_all()
    images = tf.get_collection('VISUALIZATIONS')
    if images:
      tf.summary.image('tensor_images', images[0], max_outputs=10)
    train_summaries_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs/train', graph=graph.graph)
    val_writer = tf.summary.FileWriter('logs/val')

    # Predict with running averages
    variables = EmaVariables(args.ema)

    # Variables initializer and saver
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=3)

    # Run
    with tf.Session() as sess:

      # Graph is complete
      tf.get_default_graph().finalize()
      var_sizes = [np.prod(var.shape.as_list()) for var in variables.vars]
      print('| Number of parameters:', np.sum(var_sizes), flush=True)

      # Initialize variables
      if not args.cont: # First time
        sess.run(init)
      else:             # Continue
        checkpoint = tf.train.latest_checkpoint(
            checkpoint_dir=os.path.join('models',args.dataset))
        saver.restore(sess, checkpoint)
        print('| Variables restored.', flush=True)

      # Initialize average
      sess.run(variables.initialize_backups)

      # Create contexts
      contexts = RunContexts(sess,
          train=(graph.use_train_data, variables.use_training_variables),
          val=(graph.use_val_data, variables.use_ema_variables))

      # Main loop
      for step in steps_range:

        # Train
        with contexts.train:
          sess.run( minimize,
              feed_dict={
                graph.dropouts[0]: args.dropout[0],
                graph.dropouts[1]: args.dropout[1],
              })

        # Renormalization
        if args.renormalization:
          sess.run( graph.normalization_ops )

        # Update average
        sess.run( variables.update_op )

        # Every log_every steps or at the end
        if step % args.log_every == 0 or step == steps_range.stop-1:

          # Test on train set and validation set
          with contexts.train:
            train_loss, train_regular_loss, train_summaries = sess.run(
                (graph.loss, graph.regular_loss, train_summaries_op),
                feed_dict={
                  graph.dropouts[0]: 0,
                  graph.dropouts[1]: 0,
                })
          with contexts.val:
            val_loss, val_summaries = sess.run(
                (graph.loss, val_summaries_op),
                feed_dict={
                  graph.dropouts[0]: 0,
                  graph.dropouts[1]: 0,
                })

          # Log
          print('| Step: ' + str(step) + ', train loss: ' + str(train_loss) +
              ' ( reg_loss: ' + str(train_regular_loss) + ' )', flush=True)
          train_writer.add_summary(train_summaries, step)
          val_writer.add_summary(val_summaries, step)
          train_writer.flush()
          val_writer.flush()

          # Save parameters
          model_name = 'model-step{}'.format(step)
          with contexts.train:
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

  # Prints
  print('| Testing')
  print('| Dataset:', args.dataset, flush=True)

  # Instantiate the graph
  graph = CGraph(args.dataset, seed=args.seed)

  # Use it
  with graph.graph.as_default():
    
    # Constant seed for debugging
    if args.seed:
      tf.set_random_seed(args.seed)

    # Predict with running averages
    variables = EmaVariables(args.ema)

    # Create a Saver
    saver = tf.train.Saver()

    # Run
    with tf.Session() as sess:

      # Graph is complete
      tf.get_default_graph().finalize()

      # Restore parameters
      checkpoint = tf.train.latest_checkpoint(
          checkpoint_dir=os.path.join('models',args.dataset))
      saver.restore(sess, checkpoint)

      # Create context
      contexts = RunContexts(sess,
          test_set=(graph.use_test_data, variables.use_ema_variables))

      # Predict
      with contexts.test_set:
        output, loss, accuracy = sess.run(
            (graph.output, graph.loss, graph.accuracy),
            feed_dict={
              graph.dropouts[0]: 0,
              graph.dropouts[1]: 0,
            })

      # Out
      print('| Predicted:', output)
      print('| Accuracy:', accuracy)
      print('| Loss:', loss)


def debug(args):
  '''\
  Debugging
  '''

  # Prints
  print('| Debug')


  import pdb
  pdb.set_trace()


def _clear_saved(dataset):
  '''\
  Removes all files from 'models/<dataset>/', 'logs/train' and 'logs/val'.
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
  logs = [join('logs',x) for x in ['train','val','debug']]
  paths = [model] + logs

  # Rm
  for p in paths:
    if os.path.exists(p):
      shutil.rmtree(p)


def _select_optimizer(args):
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
      ' is not an optimizer. See help(maxout._select_optimizer)')

  print('| Using', opt.get_name())
  return opt


def main():
  '''\
  Main function. Called when this file is executed as script.
  '''

  # Defaults
  learning_rate = 0.05
  n_steps = 200
  log_every = 20
  optimizer = 'adam'
  seed = 4134631
  dataset = 'cifar10'
  ema = 0.99

  ## Parsing arguments
  parser = argparse.ArgumentParser(description='Training and testing with\
      the Maxout network')
  parser.add_argument('op', choices=['train','test','debug'],
      help='What to do with the net. Most options only affect training.')
  parser.add_argument('-d', '--dataset', default=dataset,
      choices=['example','mnist','cifar10'], help='Which dataset to load')
  parser.add_argument('-r', '--rate', type=float, default=learning_rate,
      help='Learning rate / step size. Depends on the optimizer.')
  parser.add_argument('-s', '--steps', type=int, default=n_steps,
      help='Number of steps of the optimization')
  parser.add_argument('-l', '--log_every', type=int, default=log_every,
      help='Interval of number of steps between logs/saved models')
  parser.add_argument('-o', '--optimizer', default=optimizer,
      choices=['gd', 'rms', 'adagrad', 'adadelta', 'adam'],
      help='Name of the optimizer to use.\
          See `help(maxout._select_optimizer)\' to know more.')
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
  parser.add_argument('--pseudorand', action='store_const', const=seed,
      dest='seed', help='Always use the same seed for reproducible results')
  parser.add_argument('--regularization', type=float,
      help='Regularization scale. 0 means no regularization')
  parser.add_argument('--renormalization', type=float,
      help='If set, this is the maximum norm for all vectors in weigts\
          matrices')
  parser.add_argument('--ema', type=float, default=ema,
      help='Running average decay rate (something like 0.99).')

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

  if args.renormalization != None and args.renormalization <= 0:
    raise ValueError(
        '--renormalization must be a positive length.')

  if args.ema and not (0 <= args.ema < 1):
    raise ValueError(
        '--ema must be a decay rate in [0, 1).')

  # Go
  if args.op == 'train':
    training(args)
  elif args.op == 'test':
    testing(args)
  elif args.op == 'debug':
    debug(args)



if __name__ == '__main__':
  main()
