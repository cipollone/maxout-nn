#!/usr/bin/env python3

'''\
Main script file. Depending on the parameters, we can train or test with
different Maxout networks.
'''


import os
import argparse
import shutil
import tensorflow as tf

from graphs import CGraph
import data


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
  graph = CGraph(args.dataset)

  # Use it
  with graph.graph.as_default():

    # Add the optimizer
    optimizer = select_optimizer(args)
    minimize = optimizer.minimize(graph.loss)

    # Init
    init = tf.global_variables_initializer()
    
    # Logger
    tf.summary.scalar('loss', graph.loss)
    summaries_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('logs/train', graph=graph.graph)
    test_writer = tf.summary.FileWriter('logs/test', graph=graph.graph)

    # Saver
    saver = tf.train.Saver(max_to_keep=3)

    # Run
    with tf.Session() as sess:

      # Initialize variables
      if not args.cont: # First time
        sess.run(init)
      else:             # Continue
        checkpoint = tf.train.latest_checkpoint(
            checkpoint_dir=os.path.join('models',args.dataset))
        saver.restore(sess, checkpoint)
        print('| Variables restored.')

      # Load dataset once
      (data_train, data_test) = data.load(args.dataset)

      # Main loop
      for step in steps_range:

        # Train
        sess.run( minimize, feed_dict={
              graph.input_ph: data_train[0],
              graph.labels_ph: data_train[1] })

        # Every log_every steps or at the end
        if step % args.log_every == 0 or step == steps_range.stop-1:

          # Test on train set and test set
          train_loss, train_summaries = sess.run( (graph.loss, summaries_op),
              feed_dict={
                graph.input_ph: data_train[0],
                graph.labels_ph: data_train[1] })
          test_loss, test_summaries = sess.run( (graph.loss, summaries_op),
              feed_dict={
                graph.input_ph: data_test[0],
                graph.labels_ph: data_test[1] })

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
    args: namespace of options with these fields:
      dataset: dataset name
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

      # Run
      _, (features_test, labels_test) = data.load(args.dataset)
      output,loss,errors = sess.run( (graph.output, graph.loss, graph.errors),
          feed_dict={
            graph.input_ph: features_test,
            graph.labels_ph: labels_test })

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

  print(args)


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
  logs = (join('logs','train'), join('logs','test'))
  paths = (model,) + logs

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

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    training(args)
  elif args.op == 'test':
    testing(args)
  elif args.op == 'debug':
    debug(args)



if __name__ == '__main__':
  main()
