from __future__ import division, print_function, absolute_import
import cv2
import csv
import gc
import time
from PIL import Image
from tensorflow.python.training.training_util import global_step
from tflearn import Momentum, Adam
from tflearn.optimizers import Optimizer, RMSProp
import tensorlayer

from ROI.files_loader import File_Loader
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from tensorflow.contrib.losses import log_loss, absolute_difference, mean_pairwise_squared_error
from tflearn.metrics import accuracy_op
from tflearn.objectives import softmax_categorical_crossentropy, categorical_crossentropy, mean_square, \
    binary_crossentropy, roc_auc_score, hinge_loss
from Model import Model
import tflearn
import os


def rmse(y_pred, y_true):
    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))
    return error


def accuracy_error(y_pred, y_true):
    error = tf.sqrt(tf.subtract(y_pred, y_true))
    return error


path = 'C:\\Users\\Farshid\\Desktop\\MRI\\data\\'
# print(path)
gc.collect()
files = File_Loader(path=path)
input_image_paths, rois_path_list, annotations_path_list, output_images_path_list, rois_array_list = files.load_paths()

# print(input_image_paths.sort())
# print(annotations_path_list)
input_image = files.load_images(input_path=input_image_paths)
# rois_array = files.load_rois_array(input_path=rois_array_list)

# for i in range(len(input_image)):
#     print(i, ' ', input_image[i].shape)
# rois_files = files.load_output_images(input_path=rois_path_list)
output_images = files.load_output_images(input_path=output_images_path_list)
# output_images = files.load_output_images(input_path=output_images_path_list)
# img = Image.fromarray(rois_files[0], 'RGB')
# img.show()

#
# x_list, y_list, w_list, h_list = files.get_location(annotation_paths=annotations_path_list)
# output_list = []

# for a, b, c, d in zip(x_list, y_list, w_list, h_list):
#     output_list.append([float(a)/4, float(b)/4, float(c)/4, float(d)/4])

# output_list = np.asarray(output_list)

frame_height = input_image[0].shape[0]
frame_width = input_image[0].shape[1]
input_image = np.reshape(input_image, (-1, frame_height, frame_width, 1))

# rois_array = np.reshape(rois_array, (-1, len(rois_array[0])))

output_height = output_images[0].shape[0]
output_width = output_images[0].shape[1]

print(output_height, output_width, output_images[0].shape[2])

# input_image = np.reshape(input_image, (-1, frame_height, frame_width, 1))
# rois_files = np.reshape(rois_files, (-1, 64 * 64 * 1))
# output_images = np.reshape(output_images, (-1, output_height * output_width * 1))

# print(len(rois_array), len(rois_array[0]))
# output_list = np.reshape(output_list,(-1,4))
output_images = np.reshape(output_images, (-1, output_height * output_width * 3))
# roi_image = np.reshape(rois_files, (-1, frame_height * frame_width))
gc.collect()
Model = Model(input_shape=[None, frame_height, frame_width, 1], output_shape=output_height * output_width * 3)
network = Model.load_model()

momentum = Momentum(learning_rate=0.00001, lr_decay=0.96, decay_step=900)
adam = Adam(learning_rate= 0.0000001)
rms = RMSProp(learning_rate=0.000003, decay=0.9)

net = tflearn.regression(network, optimizer=adam, shuffle_batches=True, metric='accuracy', batch_size=32,
                         loss=mean_square)
model = tflearn.DNN(net, tensorboard_verbose=0)

model.load('FR-MRInet', weights_only=True)
model.fit(input_image, output_images, n_epoch=100, show_metric=True,
          batch_size = 10, snapshot_step=len(input_image), validation_set=0.1,
          snapshot_epoch=True, run_id="ROI-last2")
# model.save('FR-MRInet')


# start_time = time.time()
# model.evaluate(input_image,output_images,batch_size=10)
# print("--- %s seconds ---" % (time.time() - start_time))
#
#
# predict_these = [0,92,287,27,63,120]
#
# start_time = time.time()
# for i in predict_these:
#     input_images = input_image[i]
#     input_images = np.reshape(input_images, (1, frame_height, frame_width, 1))
#     print(input_images.shape)
#
#     predicted = np.asarray(model.predict(input_images))
#     predicted = np.reshape(predicted, (24, 24, 3))
#
#     cv2.imwrite(str(i)+'.png', predicted)
# #
#
# print("--- %s seconds ---" % (time.time() - start_time))

# input_images = input_image[1]
# input_images = np.reshape(input_images, (1, frame_height, frame_width, 1))
# print(input_images.shape)
#
# predicted = np.asarray(model.predict(input_images))
# predicted = np.reshape(predicted, (24, 24, 3))
#
# cv2.imwrite(str(i) + '.png', predicted)
