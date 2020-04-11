#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:09:44 2019

@author: ghosh128
"""

import sys
sys.path.append("../")
import config
import os
import numpy as np
# from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tifffile as tiff
import matplotlib.colors
import matplotlib as mpl



if not os.path.exists(os.path.join(config.RESULT_DIR, "FCN_loss2_20")):
    os.makedirs(os.path.join(config.RESULT_DIR, "FCN_loss2_20"))
#
# #%%
# print("LOAD DATA")
# test_data = np.load(os.path.join(config.NUMPY_DIR, "test_data.npy"))

# testing using train set
test_data = np.load(os.path.join(config.NUMPY_DIR, "test_image.npy"))
test_label = np.load(os.path.join(config.NUMPY_DIR, "test_label.npy"))
# n_data = test_data.shape[0]

n_data = 20
test_data = test_data[:n_data, :,:,:]
test_label = test_label[:n_data, :, :]

#%%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, config.patch_size, config.patch_size, config.channels], name="inputs")
    Y = tf.placeholder(tf.int32, [None, config.patch_size, config.patch_size, config.classes], name="labels")

with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
    conv1_1_W = tf.get_variable("conv1_1_W", [3, 3, config.channels, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_1_b = tf.get_variable("conv1_1_b", [64], initializer=tf.zeros_initializer())
    conv1_2_W = tf.get_variable("conv1_2_W", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_2_b = tf.get_variable("conv1_2_b", [64], initializer=tf.zeros_initializer())
conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, conv1_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv1_1_b))
conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_1, conv1_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv1_2_b))

maxpool1 = tf.nn.max_pool(conv1_2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
    conv2_1_W = tf.get_variable("conv2_1_W", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_1_b = tf.get_variable("conv2_1_b", [128], initializer=tf.zeros_initializer())
    conv2_2_W = tf.get_variable("conv2_2_W", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_2_b = tf.get_variable("conv2_2_b", [128], initializer=tf.zeros_initializer())
conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1, conv2_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv2_1_b))
conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_1, conv2_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv2_2_b))

maxpool2 = tf.nn.max_pool(conv2_2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
    conv3_1_W = tf.get_variable("conv3_1_W", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv3_1_b = tf.get_variable("conv3_1_b", [256], initializer=tf.zeros_initializer())
    conv3_2_W = tf.get_variable("conv3_2_W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv3_2_b = tf.get_variable("conv3_2_b", [256], initializer=tf.zeros_initializer())
    conv3_3_W = tf.get_variable("conv3_3_W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv3_3_b = tf.get_variable("conv3_3_b", [256], initializer=tf.zeros_initializer())
conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool2, conv3_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv3_1_b))
conv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_1, conv3_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv3_2_b))
conv3_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_2, conv3_3_W, strides=[1, 1, 1, 1], padding="SAME"), conv3_3_b))

maxpool3 = tf.nn.max_pool(conv3_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE):
    conv4_1_W = tf.get_variable("conv4_1_W", [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv4_1_b = tf.get_variable("conv4_1_b", [512], initializer=tf.zeros_initializer())
    conv4_2_W = tf.get_variable("conv4_2_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv4_2_b = tf.get_variable("conv4_2_b", [512], initializer=tf.zeros_initializer())
    conv4_3_W = tf.get_variable("conv4_3_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv4_3_b = tf.get_variable("conv4_3_b", [512], initializer=tf.zeros_initializer())
conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool3, conv4_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv4_1_b))
conv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_1, conv4_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv4_2_b))
conv4_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_2, conv4_3_W, strides=[1, 1, 1, 1], padding="SAME"), conv4_3_b))

maxpool4 = tf.nn.max_pool(conv4_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE):
    conv5_1_W = tf.get_variable("conv5_1_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv5_1_b = tf.get_variable("conv5_1_b", [512], initializer=tf.zeros_initializer())
    conv5_2_W = tf.get_variable("conv5_2_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv5_2_b = tf.get_variable("conv5_2_b", [512], initializer=tf.zeros_initializer())
    conv5_3_W = tf.get_variable("conv5_3_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv5_3_b = tf.get_variable("conv5_3_b", [512], initializer=tf.zeros_initializer())
conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool4, conv5_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv5_1_b))
conv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5_1, conv5_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv5_2_b))
conv5_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5_2, conv5_3_W, strides=[1, 1, 1, 1], padding="SAME"), conv5_3_b))

maxpool5 = tf.nn.max_pool(conv5_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("fc6", reuse=tf.AUTO_REUSE):
    fc6_W = tf.get_variable("fc6_W", [7, 7, 512, 4096], initializer=tf.contrib.layers.xavier_initializer())
    fc6_b = tf.get_variable("fc6_b", [4096], initializer=tf.zeros_initializer())
fc6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool5, fc6_W, strides=[1, 1, 1, 1], padding="SAME"), fc6_b))

with tf.variable_scope("fc7", reuse=tf.AUTO_REUSE):
    fc7_W = tf.get_variable("fc7_W", [1, 1, 4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
    fc7_b = tf.get_variable("fc7_b", [4096], initializer=tf.zeros_initializer())
fc7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(fc6, fc7_W, strides=[1, 1, 1, 1], padding="SAME"), fc7_b))

with tf.variable_scope("fc8", reuse=tf.AUTO_REUSE):
    fc8_W = tf.get_variable("fc8_W", [1, 1, 4096, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    fc8_b = tf.get_variable("fc8_b", [config.classes], initializer=tf.zeros_initializer())
fc8 = tf.nn.bias_add(tf.nn.conv2d(fc7, fc8_W, strides=[1, 1, 1, 1], padding="SAME"), fc8_b)

with tf.variable_scope("Ufc7", reuse=tf.AUTO_REUSE):
    Ufc7_W = tf.get_variable("Ufc7_W", [4, 4, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Ufc7 = tf.nn.conv2d_transpose(fc8, Ufc7_W, tf.stack([tf.shape(maxpool4)[0], tf.shape(maxpool4)[1], tf.shape(maxpool4)[2], config.classes]), strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("Spool4", reuse=tf.AUTO_REUSE):
    Spool4_W = tf.get_variable("Spool4_W", [1, 1, 512, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Spool4_b = tf.get_variable("Spool4_b", [config.classes], initializer=tf.zeros_initializer())
Spool4 = tf.nn.bias_add(tf.nn.conv2d(maxpool4, Spool4_W, strides=[1, 1, 1, 1], padding="SAME"), Spool4_b)

fuse1 = tf.add(Ufc7, Spool4)

with tf.variable_scope("Ufuse1", reuse=tf.AUTO_REUSE):
    Ufuse1_W = tf.get_variable("Ufuse1_W", [4, 4, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Ufuse1 = tf.nn.conv2d_transpose(fuse1, Ufuse1_W, tf.stack([tf.shape(maxpool3)[0], tf.shape(maxpool3)[1], tf.shape(maxpool3)[2], config.classes]), strides=[1, 2, 2, 1], padding="SAME")

with tf.variable_scope("Spool3", reuse=tf.AUTO_REUSE):
    Spool3_W = tf.get_variable("Spool3_W", [1, 1, 256, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Spool3_b = tf.get_variable("Spool3_b", [config.classes], initializer=tf.zeros_initializer())
Spool3 = tf.nn.bias_add(tf.nn.conv2d(maxpool3, Spool3_W, strides=[1, 1, 1, 1], padding="SAME"), Spool3_b)

fuse2 = tf.add(Ufuse1, Spool3)

with tf.variable_scope("Output", reuse=tf.AUTO_REUSE):
    Output_W = tf.get_variable("Output_W", [16, 16, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Z = tf.nn.conv2d_transpose(fuse2, Output_W, tf.stack([tf.shape(maxpool4)[0], config.patch_size, config.patch_size, config.classes]), strides=[1, 8, 8, 1], padding="SAME")
Z = tf.argmax(Z, dimension=3)
#%%
print("TEST MODEL")
saver = tf.train.Saver()
preds = np.zeros((test_data.shape[0], config.patch_size, config.patch_size))
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "FCN_loss2", "model20.ckpt"))
    n_batches = test_data.shape[0]//config.FCN_batch_size
    for batch in range(n_batches):
        print(batch)
        data_batch = test_data[batch*config.FCN_batch_size:(batch+1)*config.FCN_batch_size, :, :, :]
        feed_dict = {X: data_batch}
        preds[batch*config.FCN_batch_size:(batch+1)*config.FCN_batch_size, :, :] = sess.run(Z, feed_dict=feed_dict)
#%%

N= 6
# define the colormap
cmap = plt.cm.jet
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# create the new map
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

# define the bins and normalize
bounds = np.linspace(0,N,N+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

for i in range(n_data):
    image = test_data[i,:,:,:]

    label = test_label[i,:,:]
    print("For ith test file", label.shape)

    plt.figure(figsize = (30,10))
    plt.subplot(1,3,1)
    plt.imshow(image.astype('uint8'))
    plt.subplot(1,3,2 )
    plt.imshow(label, cmap= cmap, norm = norm)
    plt.subplot(1,3,3)
    plt.imshow(preds[i,:,:] , cmap= cmap, norm = norm)
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(os.path.join(config.RESULT_DIR, "FCN_loss2_20", str(i)+".png"), format='png')
    print(os.path.join(config.RESULT_DIR, "FCN_loss2_20", str(i)+".png"))
    plt.close()
