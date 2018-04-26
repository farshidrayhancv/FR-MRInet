"""
Created on Wed Apr 25 22:43:42 2018

@author: Farshid
"""
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
import cv2
import numpy as np







Model = Model(input_shape=[None, 64, 64, 1], output_shape=24 * 24 * 3)
network = Model.load_model()

adam = Adam(learning_rate=0.00000002)

net = tflearn.regression(network, optimizer=adam, shuffle_batches=True, metric='accuracy', batch_size=32, loss='mean_square')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('asd', weights_only=True)



cap = cv2.VideoCapture(0)
cap.set(3, 64)
cap.set(4, 64)

while True:
    ret, img = cap.read()

    cv2.imshow('raw',img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (64, 64))


    img =  np.reshape(np.asarray(model.predict(np.reshape(img, (1, 64, 64, 1)))),(24, 24, 3))
    # img = np.reshape(pred, (24, 24, 3))

    # img = np.asarray(img)
    # print(img.shape)

    # cv2.imshow("thresholded", imgray*thresh2)
    img = cv2.resize(img, (160, 160))
    cv2.imshow("input", img)
    key = cv2.waitKey(10)
    if key == 'q':
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()