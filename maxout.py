#!/usr/bin/env python3

'''\
Main script file. Depending on the parameters, we can train or test with
different Maxout networks.
'''


import os
import argparse
import shutil
import time
import tensorflow as tf

from graphs import CGraph
import data


# TODO: training and testing loss
# TODO: add option to continue training
# TODO: subs delays with confirmations


def training(args):
  '''\
  Training function. Creates the training tf Graph and starts training. Saves
  checkpoints in models/<args.dataset>/model

  Args:
    args: namespace of options with these fields:
      dataset: dataset name
      rate: learning rate
      steps: total number of optimization steps
      log_every: interval of steps between log/saved models
  '''

  # Prints
  print('Training')
  time.sleep(1)

  # Clear old logs and models
  clear_saved(args.dataset)

  # Instantiate the graph
  graph = CGraph(args.dataset)

  # Use it
  with graph.graph.as_default():

    # Add the optimizer
    optimizer = tf.train.GradientDescentOptimizer(args.rate)
    minimize = optimizer.minimize(graph.loss)

    # Init
    init = tf.global_variables_initializer()
    
    # Logger
    tf.summary.scalar('loss', graph.loss)
    summaries_op = tf.summary.merge_all()

    logs_writer = tf.summary.FileWriter('logs')
    logs_writer.add_graph(graph.graph)
    logs_writer.flush()

    # Saver
    saver = tf.train.Saver(max_to_keep=3)

    # Run
    with tf.Session() as sess:

      # Initialize variables
      sess.run(init)

      # Load dataset once
      (features_train, labels_train), _ = data.load(args.dataset)

      # Main loop
      for step in range(1, args.steps+1):

        # Train
        sess.run( minimize, feed_dict={
              graph.input_ph: features_train,
              graph.labels_ph: labels_train })

        # Every log_every steps or at the end
        if step % args.log_every == 0 or step == args.steps:

          # Test
          loss, summaries = sess.run( (graph.loss, summaries_op),
              feed_dict={
                graph.input_ph: features_train,
                graph.labels_ph: labels_train })

          # Log
          print('Step: ' + str(step) + ', loss: ' + str(loss))
          logs_writer.add_summary(summaries, step)

          # Save parameters
          model_name = 'model-step{}'.format(step)
          saver.save(sess, os.path.join('models',args.dataset,model_name))


def testing(args):
  '''\
  Test the performances of the net on the test set. Creates the tf Graph for
  testing, loads the weights and evaluates the predictions. Parameters are
  loaded from last checkpoint: models/<args.dataset>/model.
  
  Args:
    args: namespace of options with these fields:
      dataset: dataset name
  '''

  # Prints
  print('Testing')
  time.sleep(1)

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
      print('Predicted:', output)
      print('Loss:', loss)
      print('Errors:', errors)


def debug():
  '''\
  Debugging
  '''

  # Prints
  print('Debug')


def clear_saved(dataset):
  '''\
  Removes all files from 'models/<dataset>/' and 'logs/'.

  Args:
    dataset: The name of the dataset to use.
  '''
  
  # List
  model = os.path.join('models', dataset)
  logs = [os.path.join('logs',l) for l in os.listdir('logs')
      if 'dir.txt' not in l]

  # Rm
  if os.path.exists(model):
    shutil.rmtree(model)
  for f in logs:
    os.remove(f)


def main():
  '''\
  Main function. Called when this file is executed as script
  '''

  # Defaults
  learning_rate = 0.05
  n_steps = 200
  log_every = 20

  ## Parsing arguments
  parser = argparse.ArgumentParser(description='Training and testing with\
      the Maxout network')
  parser.add_argument('op', choices=['train','test','debug'],
      help='What to do with the net')
  parser.add_argument('-d', '--dataset', default='example',
      choices=['example'], help='Which dataset to load')
  parser.add_argument('-r', '--rate', type=float, default=learning_rate,
      help='Learning rate / step size')
  parser.add_argument('-s', '--steps', type=int, default=n_steps,
      help='Number of steps of the optimization')
  parser.add_argument('-l', '--log_every', type=int, default=log_every,
      help='Interval of number of steps between logs/saved models')

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    training(args)
  elif args.op == 'test':
    testing(args)
  elif args.op == 'debug':
    debug()



if __name__ == '__main__':
  main()
