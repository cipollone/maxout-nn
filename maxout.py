#!/usr/bin/env python3

'''\
Main script file.

Depending on the parameters, we can train, test or do predictions with Maxout
networks.
'''


import os
import argparse
import shutil
import time
import tensorflow as tf

from graphs import CGraph
import data

# TODO: move this to main
learning_rate = 0.05
n_steps = 200
log_every = 20

# TODO: add option to continue training
# TODO: subs delays with confirmations


def training(dataset):
  '''\
  Training function. Creates the training tf Graph and starts training. Saves
  checkpoints in models/<dataset>/model

  Args:
    dataset: The name of the dataset to use. Each dataset has a different model
  '''

  # Prints
  print('Training')
  time.sleep(1)

  # Clear old logs and models
  clear_saved(dataset)

  # Instantiate the graph
  graph = CGraph(dataset)

  # Use it
  with graph.graph.as_default():

    # Add the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    minimize = optimizer.minimize(graph.loss)

    # Init
    init = tf.global_variables_initializer()
    
    # Logger
    logs_writer = tf.summary.FileWriter('logs', graph.graph)
    logs_writer.flush()

    # Saver
    saver = tf.train.Saver(max_to_keep=3)

    # Run
    with tf.Session() as sess:

      # Initialize variables
      sess.run(init)

      # Load dataset once
      (features_train, labels_train), _ = data.load(dataset)

      # Train
      for step in range(1, n_steps+1):

        # Minimize
        sess.run( minimize, feed_dict={
              graph.input_ph: features_train,
              graph.labels_ph: labels_train })

        # Every log_every steps
        if step % log_every == 0:

          # Test
          loss = sess.run(graph.loss, feed_dict={
                graph.input_ph: features_train,
                graph.labels_ph: labels_train })

          # Log
          print('Step: ' + str(step) + ', loss: ' + str(loss))

          # Save parameters
          model_name = 'model-step{}'.format(step)
          saver.save(sess, os.path.join('models',dataset,model_name))


def predict(dataset):
  '''\
  Predict from a set of inputs. Creates the tf Graph for predictions, loads the
  weights and makes a prediction for each input. Parameters are loaded from
  last checkpoint: models/<dataset>/model
  
  Args:
    dataset: The name of the dataset to use. Each dataset has a different model
  '''

  # Prints
  print('Predict')
  time.sleep(1)

  # Instantiate the graph
  graph = CGraph(dataset)

  # Use it
  with graph.graph.as_default():
    
    # Create a Saver
    saver = tf.train.Saver()

    # Run
    with tf.Session() as sess:

      # Restore parameters
      checkpoint = tf.train.latest_checkpoint(
          checkpoint_dir=os.path.join('models',dataset))
      saver.restore(sess, checkpoint)

      # Run
      _, (features_test, labels_test) = data.load(dataset)
      output = sess.run( graph.output,
          feed_dict={
            graph.input_ph: features_test[0:8],
            graph.labels_ph: labels_test[0:8] })
      print(output)


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

  ## Parsing arguments

  # Argument parser
  parser = argparse.ArgumentParser(description='Training and predictions with\
      the Maxout network')
  parser.add_argument('op', choices=['train','predict','debug'],
      help='What to do with the net')
  parser.add_argument('-d', '--dataset', default='example',
      choices=['example'], help='Which dataset to load')

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    training(args.dataset)
  elif args.op == 'predict':
    predict(args.dataset)
  elif args.op == 'debug':
    debug()



if __name__ == '__main__':
  main()
