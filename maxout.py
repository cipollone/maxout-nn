#!/usr/bin/env python3

'''\
Main script file.

Depending on the parameters, we can train, test or do predictions with Maxout
networks.
'''


import os
import argparse
import time
import tensorflow as tf

from graphs import MaxoutGraph
import data


def training(dataset, logdir):
  '''\
  Training function. Creates the training tf Graph and starts training. Saves
  checkpoints in models/<dataset>/model

  Args:
    logdir: a new path where tensorboard events are saved
    dataset: The name of the dataset to use. Each dataset has a different model
  '''

  # Prints
  print('Training')
  time.sleep(1)

  # Instantiate the graph
  graphs = MaxoutGraph(dataset)

  # Use it
  with graphs.training_graph.as_default():
    
    # Save the graph
    logs_writer = tf.summary.FileWriter(logdir, graphs.training_graph)
    logs_writer.flush()

    # Create a Saver
    saver = tf.train.Saver(max_to_keep=1)

    # Run
    with tf.Session() as sess:

      # Initialize parameters
      sess.run(tf.global_variables_initializer())

      # Run
      (features_train, labels_train), _ = data.load(dataset)
      ret = sess.run(graphs.output, feed_dict={graphs.input: features_train})
      print(ret)

      # Save parameters
      saver.save(sess, os.path.join('models',dataset,'model'))


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
  graphs = MaxoutGraph(dataset) # TODO: the graphs shouldn't be the same as
                                # in trainig

  # Use it
  with graphs.training_graph.as_default():
    
    # Create a Saver
    saver = tf.train.Saver()

    # Run
    with tf.Session() as sess:

      # Restore parameters
      checkpoint = tf.train.latest_checkpoint(os.path.join('models',dataset))
      saver.restore(sess, checkpoint)

      # Run
      (features_train, labels_train), _ = data.load(dataset)
      ret = sess.run(graphs.output, feed_dict={graphs.input: features_train})
      print(ret)


def debug():
  '''\
  Debugging
  '''

  # Prints
  print('Debug')

  # Testing TF import
  hello = tf.constant('Hello, TensorFlow!')
  with tf.Session() as sess:
    print(sess.run(hello))


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
  parser.add_argument('--load', '-l', action='store_true',
      help='If present, the parameters from last saved checkpoints are loaded')

  args = parser.parse_args()

  # Select log directory. New with increasing numbers
  logsdir = 'logs'
  logs = os.listdir(logsdir)
  logs = [int(d) for d in logs if d.isdigit()]
  if len(logs) == 0: logs.append(0)
  logdir = os.path.join(logsdir,str(max(logs) + 1))

  # Go
  if args.op == 'train':
    training('example', logdir)
  elif args.op == 'predict':
    predict('example')
  elif args.op == 'debug':
    debug()



if __name__ == '__main__':
  main()
