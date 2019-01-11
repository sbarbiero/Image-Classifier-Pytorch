# Flower-Image-Classifier-Pytorch

This code is my submission for Udacity's AI Programming with Python Nanodegree. In this project, I created a command line application utilizing Pytorch, Neural Networks and Transfer Learning to train a new classifier on top of a pre-existing model trained on ImageNet to identify between 102 different types of flowers.

_This program can be modified to train itself on many different image classification problems if the proper arguments are passed.
Please read the comments to get information on arguments and how to use the program._

## Use the program
Run **Pytorch 0.4.0**

  * Run **train.py** to train the classifier
  * Run **predict.py** to use the checkpoint file created by train.py
  * **processimage.py** contains code to normalize an image you wish to pass to predict.py
  * **network_prep.py** contains all the model loading and configuration modules
  * **cat_to_name.json** is an example of a mapping file to utilize with the program
