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
# from sklearn.tree import DecisionTreeClassifier



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
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 3, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 64, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)

        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 128, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 128, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        #
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 256, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 256, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 256, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        #
        #
        # network = conv_2d(network, 1, 1, activation='relu', padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu',  padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu',  padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)

        # network = conv_2d(network, 1, 1, activation='relu', padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu', padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 1, 1, activation='relu', padding='valid', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 512, 3, activation='relu', padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        #
        # print(network)
        #
        # network = fully_connected(network, 4096, activation='tanh')
        # network = dropout(network, 0.5)
        # network = fully_connected(network, 4096, activation='tanh')
        # network = dropout(network, 0.5)
        # network = fully_connected(network, 17, activation='softmax')
        # network = fully_connected(network, 17, activation='softmax')
        # print(network)
        # network = tf.reshape(network, (-1, 16, 16, 4))
        # print(network)
        # encoder = tflearn.input_data(shape=[None, 784])
        # network = tflearn.fully_connected(network, 256)
        # network = tflearn.fully_connected(network, 64)
        #
        # Building the decoder
        # network = tflearn.fully_connected(network, 256)
        #
        #
        # network = conv_2d(network, 16, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network = conv_2d(network, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        # network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        # network2 = conv_2d(network, 16, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network2 = conv_2d(network2, 32, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network3 = conv_2d(network, 16, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network3 = conv_2d(network3, 32, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)

        # network = conv_2d(network, 32, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        # network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        # network = conv_2d(network, 32, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
        # network = conv_2d(network, 32, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        #
        #
        # network = merge([network1, network2, network3, network4] , mode='concat', axis=3)
        # print(network)
        #
        # network90 = max_pool_2d(network, kernel_size=2, strides=1)
        # #
        # network = max_pool_2d(network, kernel_size=2, strides=2)
        # print(network)
        # network1 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network1 = conv_2d(network1, 128, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network2 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network2 = conv_2d(network2, 128, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network91 = merge([network1, network2] , mode='concat', axis=3)
        #
        # print('91 ',network)
        # #
        # network = max_pool_2d(network, kernel_size=1, strides=1)
        # network3 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network3 = conv_2d(network3, 128, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network4 = max_pool_2d(network, kernel_size=2, strides=1)
        # network5 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network5 = conv_2d(network5, 128, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        #
        # network92 = merge([network3, network4, network5], mode='concat', axis=3)
        #
        # print('92 ',network)
        #
        # network = merge([network91, network92], mode='concat', axis=3)
        # network93 = upsample_2d(network, kernel_size=2)
        # #
        # network = upsample_2d(network,kernel_size=2)
        # print(network)
        # network1 = conv_2d(network, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network1 = conv_2d(network1, 32, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network2 = conv_2d(network, 16, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network2 = conv_2d(network2, 32, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network3 = conv_2d(network, 16, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network3 = conv_2d(network3, 32, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
        # network4 = max_pool_2d(network3, kernel_size=2, strides=1)
        # network = merge([ network3, network4] , mode='concat', axis=3)
        # print(network)
        # network = conv_2d(network, 96, 11, strides=4, activation='relu')
        # network = max_pool_2d(network, 3, strides=2)
        # network = local_response_normalization(network)
        # network = conv_2d(network, 256, 5, activation='relu')
        # network = max_pool_2d(network, 3, strides=2)
        # network = local_response_normalization(network)
        # network = conv_2d(network, 384, 3, activation='relu')
        # network = conv_2d(network, 384, 3, activation='relu')
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = max_pool_2d(network, 3, strides=2)
        # network = local_response_normalization(network)
        # network = fully_connected(network, 4096, activation='tanh')
        # network = dropout(network, 0.5)
        # network = fully_connected(network, 4096, activation='tanh')
        # network = dropout(network, 0.5)

        # network = conv_2d(network, 64, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 64, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = local_response_normalization(network)
        # network = max_pool_2d(network, 2, strides=2)
        # network = conv_2d(network, 64, 3, strides=4, activation='relu')
        # network = max_pool_2d(network, 3, strides=2)
        # network = local_response_normalization(network)
        # tf.image.resize_bilinear()

        # network1 = conv_2d(network, 64, 5, activation='relu')
        # network1 = conv_2d(network1, 64, 5, activation='relu')
        # network1 = conv_2d(network1, 64, 5, activation='relu')
        # network2 = conv_2d(network, 64, 3, activation='relu')
        # network2 = conv_2d(network2, 64, 3, activation='relu')
        # network2 = conv_2d(network2, 64, 3, activation='relu')
        # network = merge([network1, network2] , mode='concat', axis=3)
        # network = max_pool_2d(network, 3, strides=2)

        network = self.inception(network, 64)
        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = self.inception(network,64,strides=2)
        print(network)
        # network = max_pool_2d(network, 2, strides=2)

        # network1 = conv_2d(network, 128, 3, activation='relu')
        # network2 = conv_2d(network, 128, 2, activation='relu')
        # network = merge([network1, network2], mode='concat', axis=3)

        # network = self.inception(network,64)
        # network = self.inception(network,64,strides=2)
        # print(network)

        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu')
        network = conv_2d(network, 128, 3, activation='relu',strides=2)
        network = self.inception(network, 96)
        print(network)


        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu',strides=2)
        network = self.inception(network, 96)
        print(network)

        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu')
        network = conv_2d(network, 256, 3, activation='relu', strides=2)
        network = self.inception(network, 128)

        print(network)

        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = conv_2d(network, 512, 3, activation='relu')
        network = self.inception(network, 128)
        print(network)
        # network = max_pool_2d(network, 2, strides=2)

        # network = max_pool_2d(network, 3, strides=2)
        #
        # network1 = conv_2d(network, 512, 5, activation='relu')
        # network1 = conv_2d(network1, 512, 5, activation='relu')
        # network1 = conv_2d(network1, 512, 5, activation='relu')
        # network2 = conv_2d(network, 512, 5, activation='relu')
        # network2 = conv_2d(network2, 512, 5, activation='relu')
        # network2 = conv_2d(network2, 512, 3, activation='relu')
        # network = merge([network1, network2] , mode='concat', axis=3)
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = max_pool_2d(network, 2, strides=2)
        # network = tf.image.resize_images(network, size=(14,14), method=tf.image.ResizeMethod.BICUBIC)
        # print(network)
        #
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = upsample_2d(network, kernel_size=2)
        # network = tf.image.resize_images(network, size=(28,28), method=tf.image.ResizeMethod.BILINEAR)
        # print(network)

        # network = conv_2d(network, 256, 3, activation='relu')
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = tf.image.resize_images(network, size=(10,10), method=tf.image.ResizeMethod.BICUBIC)
        # network = upsample_2d(network, kernel_size=2)

        # network = conv_2d(network, 256, 3, activation='relu')
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = tf.image.resize_images(network, size=(12,12), method=tf.image.ResizeMethod.BICUBIC)
        # network = upsample_2d(network, kernel_size=2)
        #
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = tf.image.resize_images(network, size=(14,14), method=tf.image.ResizeMethod.BICUBIC)
        # network = upsample_2d(network, kernel_size=2)

        # network = conv_2d(network, 64, 3, activation='relu')
        # network = conv_2d(network, 64, 3, activation='relu')
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = conv_2d(network, 128, 3, activation='relu')
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = conv_2d(network, 256, 3, activation='relu')
        # network = max_pool_2d(network, 2, strides=2)
        #
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = upsample_2d(network, kernel_size=2)

        # network = self.inception(network, filter_num=128)
        # network = self.inception(network, filter_num=128)
        # network = self.inception(network, filter_num=128)
        # network = max_pool_2d(network, 3, strides=2)

        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = upsample_2d(network, kernel_size=2)

        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 512, 3, activation='relu')
        # network = conv_2d(network, 368, 3, activation='relu')
        # network = self.inception(network, filter_num=128)
        # network = self.inception(network, filter_num=128)
        # network = max_pool_2d(network, 3, strides=2)

        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.8)
        network = fully_connected(network, 4096, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, self.output_shape, activation='relu')

        return network

# tensorboard --logdir=C:/tmp/tflearn_logs/