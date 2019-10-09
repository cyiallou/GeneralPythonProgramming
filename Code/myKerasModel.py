"""
Simple NN built with Keras
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')


x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_dataset()

# Normalize image vectors
x_train = x_train_orig/255.
x_test = x_test_orig/255.

y_train = y_train_orig.T
y_test = y_test_orig.T

print (f'number of training examples = {x_train.shape[0]}')
print (f'number of test examples = {x_test.shape[0]}')
print (f'x_train shape: {x_train.shape}')
print (f'y_train shape: {y_train.shape}')
print (f'x_test shape: {x_test.shape}')
print (f'y_test shape: {y_test.shape}')


def keras_model(input_shape):
    """
    Arguments:
        input_shape: shape of the dataset images

    Returns:
        model: a Model() instance in Keras
    """
    x_input = Input(input_shape)

    # Zero-Padding: pads the border of x_input with zeroes
    X = ZeroPadding2D((3, 3))(x_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=x_input, outputs=X, name='model2')
    return model


def main():
    model = keras_model(x_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=20, batch_size=32)
    predictions = model.evaluate(x=x_test, y=y_test)
    print (f'Loss = {predictions[0]}')
    print (f'Test Accuracy = {predictions[1]}')

    model.summary()
    plot_model(model, to_file='keras_model.png')
    SVG(model_to_dot(model).create(prog='dot', format='svg'))


if __name__ == '__main__':
    pass

