# # import tflearn
# # from tflearn.layers.core import input_data, dropout, fully_connected
# # from tflearn.layers.conv import conv_2d, max_pool_2d
# # from tflearn.layers.normalization import local_response_normalization
# # from tflearn.layers.estimator import regression
# #
# # import tflearn.datasets.oxflower17 as oxflower17
# # X, Y = oxflower17.load_data(one_hot=True, resize_pics=(224, 224))
# # print(X.shape)
# # print(X[0].shape)
# # print(Y.shape)
# # print(Y[0].shape)
# #
# # # Building 'AlexNet'
# # network = input_data(shape=[None, 224, 224, 3])
# #
# # network = conv_2d(network, 64, 11, strides=4, activation='relu')
# # network = max_pool_2d(network, 3, strides=2)
# # network = local_response_normalization(network)
# # network = conv_2d(network, 256, 5, activation='relu')
# # network = max_pool_2d(network, 3, strides=2)
# # network = local_response_normalization(network)
# # network = conv_2d(network, 384, 3, activation='relu')
# # network = conv_2d(network, 384, 3, activation='relu')
# # network = conv_2d(network, 256, 3, activation='relu')
# # network = max_pool_2d(network, 3, strides=2)
# # network = local_response_normalization(network)
# # network = fully_connected(network, 4096, activation='tanh')
# # network = dropout(network, 0.5)
# # network = fully_connected(network, 4096, activation='tanh')
# # network = dropout(network, 0.5)
# # network = fully_connected(network, 17, activation='softmax')
# # network = regression(network, optimizer='momentum',
# #                      loss='categorical_crossentropy',
# #                      learning_rate=0.001)
# #
# # # Training
# # model = tflearn.DNN(network, checkpoint_path='model_alexnet',
# #                     max_checkpoints=1, tensorboard_verbose=2)
# # model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
# #           show_metric=True, batch_size=10, snapshot_step=200,
# #           snapshot_epoch=False, run_id='alexnet_oxflowers17')
# #
# # model.save('asd.m')
#
# import tensorflow as tf
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, upsample_2d, max_pool_2d, avg_pool_2d, conv_3d, grouped_conv_2d, upsample_2d, \
#     max_pool_3d, conv_2d_transpose
# from tflearn.layers.normalization import local_response_normalization
# from tflearn.layers.estimator import regression
# from tflearn.layers.recurrent import lstm
# from tflearn.layers.embedding_ops import embedding
# from tflearn.layers.merge_ops import merge
# from tflearn.objectives import softmax_categorical_crossentropy
# # from sklearn.tree import DecisionTreeClassifier
#
#
# from PIL import Image
#
# from ROI.files_loader import File_Loader
# import cv2
# import numpy as np
# # import matplotlib.pyplot as plt
# import tflearn
# import tensorflow as tf
# from tflearn.metrics import Accuracy
# from tflearn.objectives import softmax_categorical_crossentropy, categorical_crossentropy, mean_square
# from Model import Model
# import tflearn
# import os
# # from sklearn.model_selection import StratifiedKFold
#
#
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
# from tflearn.layers.normalization import local_response_normalization
# from tflearn.layers.estimator import regression
# from tflearn.layers.merge_ops import merge
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
# # # Encoding X[0] for test
# # print("\nTest encoding of X[0]:")
# # # New model, re-using the same session, for weights sharing
# # encoding_model = tflearn.DNN(encoder, session=model.session)
# # print(encoding_model.predict([X[0]]))
# #
# # # Testing the image reconstruction on new data (test set)
# # print("\nVisualizing results after being encoded and decoded:")
# # testX = tflearn.data_utils.shuffle(testX)[0]
# # # Applying encode and decode over test set
# # encode_decode = model.predict(testX)
# # # Compare original images with their reconstructions
# # f, a = plt.subplots(2, 10, figsize=(10, 2))
# # for i in range(10):
# #     temp = [[ii, ii, ii] for ii in list(testX[i])]
# #     a[0][i].imshow(np.reshape(temp, (28, 28, 3)))
# #     temp = [[ii, ii, ii] for ii in list(encode_decode[i])]
# #     a[1][i].imshow(np.reshape(temp, (28, 28, 3)))
# # f.show()
# # plt.draw()
# # plt.waitforbuttonpress()
#
# # print(X[0].shape)
#
# def rmse(y_pred, y_true):
#     return tf.sqrt((tf.square(tf.subtract(y_pred, y_true))))
#
#
# path = os.getcwd() + '/'
# files = File_Loader(path=path)
# input_image_paths, rois_path_list, annotations_path_list, output_images_path_list = files.load_paths()
#
# # print(input_image_paths.sort())
# # print(annotations_path_list)
# input_image = files.load_images(input_path=input_image_paths)
# rois_files = files.load_rois(input_path=rois_path_list)
# output_images = files.load_output_images(input_path=output_images_path_list)
#
# #
# x_list, y_list, w_list, h_list = files.get_location(annotation_paths=annotations_path_list)
# output_list = []
#
# for a, b, c, d in zip(x_list, y_list, w_list, h_list):
#     output_list.append([float(a)/2, float(b)/2, float(c)/2, float(d)/2])
#
# output_list = np.asarray(output_list)
#
# frame_height = input_image[0].shape[0]
# frame_width = input_image[0].shape[1]
# # frame_channel = input_image[0].shape[2]
# print(input_image.shape)
# print(input_image[0].shape)
#
# print(output_images.shape)
# print(output_images[0].shape)
#
# print(output_list.shape)
# print(output_list[0].shape)
#
#
#
#
#
# input_image = np.reshape(input_image, (-1, frame_height, frame_width, 1))
# output_list = np.reshape(output_list,(-1,4))
# output_images = np.reshape(output_images, (-1, frame_height * frame_width * 3))
#
# network = input_data(shape=[None, frame_height, frame_height, 1], name='input')
#
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
# network = conv_2d(network, 312, 3, activation='relu')
# network = conv_2d(network, 312, 3, activation='relu')
# network = conv_2d(network, 312, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
#
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = conv_2d(network, 512, 3, activation='relu')
# network = max_pool_2d(network, 2, strides=2)
#
# network = fully_connected(network, 4096, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 4096, activation='relu')
# network = dropout(network, 0.5)
#
# network = fully_connected(network, 4, activation='relu')
#
# # Model = Model(input_shape=[None, frame_height, frame_width, 1], output_shape=4)
# # network = Model.load_model()
#
# net = tflearn.regression(network, optimizer='adam', learning_rate= 0.00001, shuffle_batches=True,
#                          batch_size=5, loss=rmse)
# #
# # # Training the auto encoder
# model = tflearn.DNN(net, tensorboard_verbose=0)
# model.fit(input_image, output_list, n_epoch=500, show_metric=True,
#           batch_size=5, snapshot_step=200,validation_batch_size=0.2,
#           snapshot_epoch=False,
#           run_id="ROI")
# #
# # model.save('asd.m')


import csv
import numpy as np
from PIL import Image

with open('1.csv', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)
your_list = np.asarray(your_list)
your_list = your_list.flatten()
your_list = list(your_list)

your_list = [int(float(e)) for e in your_list]

your_list = list(map(int,your_list))
your_list = np.asarray(your_list)
print(your_list)
your_list = your_list.reshape((96,96))
# predicted = Image.fromarray(your_list)
# predicted.show()

from scipy.misc import toimage
toimage(your_list).show()