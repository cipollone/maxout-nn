#!/usr/bin/env python3

'''\
Main script file.

Depending on the parameters, we can train, test or do predictions with Maxout
networks.
'''


import argparse
import time
import tensorflow as tf

import nets.example_net
import data


def training():
  '''\
  Training function. Creates the training tf Graph and starts training. Saves
  checkpoints.
  '''

  # Prints
  print('Training')
  time.sleep(1)

  # Training graph
  graph = tf.Graph()
  with graph.as_default():

    # Input
    input_ph = data.placeholder('example')

    # Net
    output = nets.example_net.model(input_ph)

    # Loss
    # TODO:

    # Run
    with tf.Session() as sess:

      # Fetch input
      (features_train, labels_train), _ = data.load('example')
      ret = sess.run(output, feed_dict={input_ph: features_train})
      print(ret)


def predict():
  '''\
  Predict from a set of inputs. Creates the tf Graph for predictions, loads the
  weights and makes a prediction for each input.
  '''

  # Prints
  print('Predict')
  time.sleep(1)

  # Predict graph
  graph = tf.Graph()
  with graph.as_default():

    # Predict ops TODO
    predict_op = tf.constant('Predict ops')

    # Define the net
    output = nets.example_net.model()

    # Run
    with tf.Session() as sess:
      print(sess.run((predict_op, output)))



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

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    training()
  elif args.op == 'predict':
    predict()
  elif args.op == 'debug':
    debug()



if __name__ == '__main__':
  main()
