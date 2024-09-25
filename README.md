# Fully-Connected-Neural-Network
Implementation for a Neural Network (NN) with 2 hidden layers (40 neurons each) to recognize handwritten digits using the MNIST dataset

## backprop_data.py
This code loads the MNIST dataset, reshapes the data into numpy arrays, and adjusts the size based on the specified train_size and test_size, returning the data and corresponding labels.

## backprop_network.py
This code defines a neural network class that performs forward and backward propagation using ReLU and cross-entropy loss, and updates its parameters via stochastic gradient descent (SGD). It supports training with mini-batches and computes accuracy during training and testing.

## backprop_main.py
This code trains a neural network on the MNIST dataset using different learning rates to evaluate performance. It visualizes training accuracy, training loss, and test accuracy over 30 epochs, then re-trains the network with the full dataset and prints the final test accuracy.
