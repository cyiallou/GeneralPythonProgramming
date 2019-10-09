import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from cnn_utils import *


np.random.seed(1)


def create_placeholders(n_h0: int, n_w0: int, n_c0: int, n_y: int):
    """Creates the placeholders for the tensorflow session.

    Arguments:
        n_h0: int, height of an input image
        n_w0: int, width of an input image
        n_c0: int, number of channels of the input
        n_y: int, number of classes

    Returns:
        x: placeholder for the data input, of shape [None, n_h0, n_w0, n_c0] and dtype "float"
        y: placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    x = tf.placeholder(dtype=tf.float32, shape=[None, n_h0, n_w0, n_c0], name='placeholder_1')
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_y], name='placeholder_2')
    return x, y


def initialize_parameters() -> dict:
    """Initializes weight parameters to build a neural network with tensorflow.

    Shapes:
        W1: [4, 4, 3, 8], W2 : [2, 2, 8, 16]

    Returns:
        parameters: a dictionary of tensors containing w1, w2
    """
    w1 = tf.get_variable('W1', [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    w2 = tf.get_variable('W2', [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {'W1': w1,
                  'W2': w2}
    return parameters


def forward_propagation(x, parameters):
    """Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLY-CONNECTED

    Arguments:
        x: input dataset placeholder of shape (input size, number of examples)
        parameters: dict containing the parameters 'W1', 'W2'
                    the shapes are given in initialize_parameters

    Returns:
        z3: the output of the last LINEAR unit
    """
    w1 = parameters['W1']
    w2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    z1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    a1 = tf.nn.relu(z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    p1 = tf.nn.max_pool(a1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    # CONV2D: filters w2, stride 1, padding 'SAME'
    z2 = tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    a2 = tf.nn.relu(z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    p2 = tf.nn.max_pool(a2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    # FLATTEN
    p2 = tf.contrib.layers.flatten(p2)
    # FULLY-CONNECTED. 6 neurons in output layer.
    z3 = tf.contrib.layers.fully_connected(p2, num_outputs=6, activation_fn=None)
    return z3


def compute_cost(z3, y):
    """Computes the cost.

    Arguments:
        z3: output of forward propagation (output of the last LINEAR unit), of shape (number of examples, 6)
        y: labels vector placeholder, same shape as z3

    Returns:
        cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y))
    return cost


def model(x_train, y_train, x_test, y_test, learning_rate=0.009,
          num_epochs=100, minibatch_size=64, print_cost=True):
    """Implements a three-layer CNN in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLY-CONNECTED

    Arguments:
        x_train: training set, of shape (None, 64, 64, 3)
        y_train: test set, of shape (None, n_y = 6)
        x_test: training set, of shape (None, 64, 64, 3)
        y_test: test set, of shape (None, n_y = 6)
        learning_rate: learning rate of the optimization
        num_epochs: number of epochs of the optimization loop
        minibatch_size: size of a minibatch
        print_cost: True to print the cost every 100 epochs

    Returns:
        train_accuracy: real number, accuracy on the train set (x_train)
        test_accuracy: real number, testing accuracy on the test set (x_test)
        parameters: parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)
    seed = 3  # numpy seed
    (m, n_h0, n_w0, n_c0) = x_train.shape
    # n_y = y_train.shape[1]
    costs = []  # To keep track of the cost

    x, y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    z3 = forward_propagation(x, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(z3, y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)  # Run the initialization

        # training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(x_train, y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_x, minibatch_y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={x: minibatch_x, y: minibatch_y})
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost and epoch % 5 == 0:
                print (f'Cost after epoch {epoch}: {minibatch_cost}')
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title(f'Learning rate = {learning_rate}')
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print(accuracy)
        train_accuracy = accuracy.eval({x: x_train, y: y_train})
        test_accuracy = accuracy.eval({x: x_test, y: y_test})
        print(f'Train Accuracy: {train_accuracy}')
        print(f'Test Accuracy: {test_accuracy}')

        return train_accuracy, test_accuracy, parameters


if __name__ == '__main__':
    pass
