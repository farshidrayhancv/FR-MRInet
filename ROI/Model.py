from builtins import str
from netrc import netrc

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_2d, upsample_2d, max_pool_2d, avg_pool_2d, conv_3d, grouped_conv_2d, upsample_2d, \
    max_pool_3d, conv_2d_transpose, atrous_conv_2d, upscore_layer, resnext_block
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.recurrent import lstm
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.merge_ops import merge

from tflearn.objectives import softmax_categorical_crossentropy


class Model:
    # Building convolutional network

    input_shape = None
    output_shape = None
    encoded_network = None

    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def inception(self,network, filter_num,activation='relu', regularizer=None,strides=1,trainable=True):

        network1 = conv_2d(network, int(filter_num/3), 1, activation='relu',strides=1,regularizer=regularizer,trainable=trainable)
        network1 = conv_2d(network1, filter_num, 5, activation='relu',strides=strides,regularizer=regularizer,trainable=trainable)
        network2 = conv_2d(network, int(filter_num/3), 1, activation='relu',strides=1,regularizer=regularizer,trainable=trainable)
        network2 = conv_2d(network2, filter_num, 3, activation='relu',strides=strides,regularizer=regularizer,trainable=trainable)
        network3 = conv_2d(network, int(filter_num/3), 1, activation='relu',strides=1,regularizer=regularizer,trainable=trainable)
        network3 = conv_2d(network3, filter_num, 2, activation='relu',strides=strides,regularizer=regularizer,trainable=trainable)
        # network4 = max_pool_2d(network, kernel_size=2, strides=strides)
        # network5 = max_pool_2d(network, kernel_size=3, strides=strides)
        # network6 = max_pool_2d(network, kernel_size=5, strides=strides)
        network = merge([network1, network2, network3] , mode='concat', axis=3)

        return network

    def load_model(self):

        network = input_data(shape=self.input_shape, name='input')

        network = self.inception(network, 64)
        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = self.inception(network,64,strides=2)

        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu',strides=2)
        network = self.inception(network, 96)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu',strides=2)
        network = self.inception(network, 96)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu', strides=2)
        network = self.inception(network, 128)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = self.inception(network, 128)

        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 6192, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, self.output_shape, activation='relu')

        return network

# tensorboard --logdir=C:/tmp/tflearn_logs/
