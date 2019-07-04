
# Maxout-NN

Easily write and train Maxout networks 

This project implements: Ian Goodfellow et al. “Maxout Networks”, [http://proceedings.mlr.press/v28/goodfellow13.html]

## How to run

Make sure you have Tensoflow (version 1.13, at least) and Numpy installed.  Run:

	python3 maxout.py -h

to see all options and commands available.

The input of this program is a dataset in the "datasets/" directory. The datasets now supported are MNIST and CIFAR-10. To use them, download their binary version to "datasets/MNIST" and "datasets/CIFAR-10" directories. You should also create a validation split and set the correct filenames. To use different datasets, write a loader in "data.py" that follows the same Tensorflow `data` API.

The outputs are Tensorboard logs in "logs/" and model parameters in "models/" directories. You can customize the nets or write your own in "nets/\*_net.py" files.
