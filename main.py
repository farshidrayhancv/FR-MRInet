from __future__ import division, print_function, absolute_import
from loader import File_loader
import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from tflearn.metrics import Accuracy
from tflearn.objectives import softmax_categorical_crossentropy, categorical_crossentropy, mean_square
from Model import Model
import tflearn

# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
# from tflearn.layers.normalization import local_response_normalization
# from tflearn.layers.estimator import regression
# from tflearn.layers.merge_ops import merge
# from sklearn.tree import DecisionTreeClassifier
#
# # Data loading and preprocessing
# import tflearn.datasets.mnist as mnist
# X, Y, testX, testY = mnist.load_data(one_hot=True)
#
# # Building the encoder
# encoder = tflearn.input_data(shape=[None, 784])
# encoder = tflearn.fully_connected(encoder, 256)
# encoder = tflearn.fully_connected(encoder, 64)
#
# # Building the decoder
# decoder = tflearn.fully_connected(encoder, 256)
# decoder = tflearn.fully_connected(decoder, 784, activation='sigmoid')
#
# # Regression, with mean square error
# net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
#                          loss='mean_square', metric=None)
#
# # Training the auto encoder
# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(X, X, n_epoch=20, validation_set=(testX, testX),
#           run_id="auto_encoder", batch_size=256)
#
# # Encoding X[0] for test
# print("\nTest encoding of X[0]:")
# # New model, re-using the same session, for weights sharing
# encoding_model = tflearn.DNN(encoder, session=model.session)
# print(encoding_model.predict([X[0]]))
#
# # Testing the image reconstruction on new data (test set)
# print("\nVisualizing results after being encoded and decoded:")
# testX = tflearn.data_utils.shuffle(testX)[0]
# # Applying encode and decode over test set
# encode_decode = model.predict(testX)
# # Compare original images with their reconstructions
# f, a = plt.subplots(2, 10, figsize=(10, 2))
# for i in range(10):
#     temp = [[ii, ii, ii] for ii in list(testX[i])]
#     a[0][i].imshow(np.reshape(temp, (28, 28, 3)))
#     temp = [[ii, ii, ii] for ii in list(encode_decode[i])]
#     a[1][i].imshow(np.reshape(temp, (28, 28, 3)))
# f.show()
# plt.draw()
# plt.waitforbuttonpress()

# print(X[0].shape)


path = '/home/farshid/PycharmProjects/MRI/dataset/'
files = File_loader(path=path)
input_image_paths, output_image_paths = files.load_paths()

input_image, output_image = files.load_images(input_path=input_image_paths,output_path=output_image_paths)
# output_image = files.load_images(images_path=output_image_paths)

print(input_image[0].shape)
print(output_image[0].shape)


frame_height = input_image[0].shape[0]
frame_width = input_image[0].shape[1]
frame_channel = input_image[0].shape[2]

input_image = np.reshape(input_image, (-1, frame_height, frame_width, frame_channel))
output_image = np.reshape(output_image, (-1, frame_height * frame_width * frame_channel))

# Building the encoder
# network = tflearn.input_data(shape=[None, 192, 256, 3])
# network = conv_2d(network, 32, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
#
# network = max_pool_2d(network, kernel_size=2, strides=2)
#
# network1 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network1 = conv_2d(network1, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network2 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network2 = conv_2d(network2, 16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network3 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network3 = conv_2d(network3, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network4 = max_pool_2d(network, kernel_size=1, strides=1)
#
# network9 = merge([network1, network2, network3, network4], mode='concat', axis=3)
# network9 = max_pool_2d(network9, kernel_size=2, strides=2)
#
# network99 = max_pool_2d(network9, kernel_size=1, strides=1)
#
# print('99 ', network99)
#
# network5 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network5 = conv_2d(network5, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
#
# network6 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network6 = conv_2d(network6, 16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
#
# network7 = conv_2d(network, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network7 = conv_2d(network7, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 2, 2, 1])
#
# network8 = max_pool_2d(network, kernel_size=1, strides=2)
#
# network10 = merge([network5, network6, network7, network8], mode='concat', axis=3)
#
#
# network10 = merge([network9, network10], mode='concat', axis=3)
#
# network11 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network11 = conv_2d(network11, 16, 3, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network12 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network12 = conv_2d(network12, 16, 5, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network13 = conv_2d(network10, 8, 1, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
# network13 = conv_2d(network13, 16, 2, activation='relu', regularizer="L2", padding='same', strides=[1, 1, 1, 1])
#
# network14 = network = max_pool_2d(network10, kernel_size=1, strides=1)
#
# network9 = merge([network11, network12, network13, network14], mode='concat', axis=3)
#
# network9 = merge([network9, network99], mode='concat', axis=3)
# network = max_pool_2d(network9, kernel_size=2, strides=2)
# network = fully_connected(network, 2048, activation='relu')
#
# network = tflearn.fully_connected(network, 147456, activation='sigmoid')
#
#
# Regression, with mean square error
# with tf.device('/gpu:0'):
# z_mean = tflearn.fully_connected(encoder, latent_dim)
# z_std = tflearn.fully_connected(encoder, latent_dim)
# eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
#                        name='epsilon')
# z = z_mean + tf.exp(z_std / 2) * eps
#
#
#
# def vae_loss(x_reconstructed, x_true):
#     # Reconstruction loss
#     encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
#                          + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
#     encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
#     # KL Divergence loss
#     kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
#     kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
#     return tf.reduce_mean(encode_decode_loss + kl_div_loss)


Model = Model(input_shape=[None, frame_height, frame_width, 3], output_shape=frame_height * frame_width * frame_channel)
network = Model.load_model()
#
net = tflearn.regression(network, optimizer='adam', learning_rate=0.001, shuffle_batches=True,
                         batch_size=50,
                         loss=mean_square)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=3,checkpoint_path='model_finetuning', max_checkpoints=3)
model.fit(input_image, np.reshape(output_image, (-1, frame_height * frame_width * frame_channel)), n_epoch=50, show_metric=True,
          validation_set=(input_image, np.reshape(output_image, (-1, frame_height * frame_width * frame_channel))), batch_size=50,
          run_id="auto_encoder")
