#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tifffile as tiff
import matplotlib.colors
import matplotlib as mpl

N_DATA = 20
N= 6
MODEL = 'UNET'
LOSS_FUNCTION = "SMILE"
EPOCH = 50

result_folder = MODEL + '_' + LOSS_FUNCTION + '_' + str(EPOCH)
model_folder = MODEL + '_' + LOSS_FUNCTION
checkpoint_name = 'model_'+str(EPOCH)+ '.ckpt'

print("LOAD DATA")
test_data = np.load(os.path.join(config.NUMPY_DIR, "test_image.npy"))
test_label = np.load(os.path.join(config.NUMPY_DIR, "test_label.npy"))
# n_data = test_data.shape[0]

test_data = test_data[:N_DATA, :,:,:]
test_label = test_label[:N_DATA, :, :]


def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)

print("BUILD MODEL")
tf.reset_default_graph()
parameters = []
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, config.patch_size, config.patch_size, config.channels], name="inputs")
    Y = tf.placeholder(tf.int32, [None, config.patch_size, config.patch_size], name="labels")

with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
    conv1_1_W = tf.get_variable("conv1_1_W", [3, 3, config.channels, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_1_b = tf.get_variable("conv1_1_b", [64], initializer=tf.zeros_initializer())
    conv1_2_W = tf.get_variable("conv1_2_W", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_2_b = tf.get_variable("conv1_2_b", [64], initializer=tf.zeros_initializer())
conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, conv1_1_W, strides=[1,1,1,1], padding="SAME"), conv1_1_b))
conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_1, conv1_2_W, strides=[1,1,1,1], padding="SAME"), conv1_2_b))
parameters += [conv1_1_W, conv1_1_b, conv1_2_W, conv1_2_b]

maxpool1, maxpool1_ind = tf.nn.max_pool_with_argmax(conv1_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("maxpool1:", maxpool1.shape)

with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
    conv2_1_W = tf.get_variable("conv2_1_W", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_1_b = tf.get_variable("conv2_1_b", [128], initializer=tf.zeros_initializer())
    conv2_2_W = tf.get_variable("conv2_2_W", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_2_b = tf.get_variable("conv2_2_b", [128], initializer=tf.zeros_initializer())
conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1, conv2_1_W, strides=[1,1,1,1], padding="SAME"), conv2_1_b))
conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_1, conv2_2_W, strides=[1,1,1,1], padding="SAME"), conv2_2_b))
parameters += [conv2_1_W, conv2_1_b, conv2_2_W, conv2_2_b]

maxpool2, maxpool2_ind = tf.nn.max_pool_with_argmax(conv2_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("maxpool2:", maxpool2.shape)

with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
    conv3_1_W = tf.get_variable("conv3_1_W", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv3_1_b = tf.get_variable("conv3_1_b", [256], initializer=tf.zeros_initializer())
    conv3_2_W = tf.get_variable("conv3_2_W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    conv3_2_b = tf.get_variable("conv3_2_b", [256], initializer=tf.zeros_initializer())
conv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool2, conv3_1_W, strides=[1,1,1,1], padding="SAME"), conv3_1_b))
conv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_1, conv3_2_W, strides=[1,1,1,1], padding="SAME"), conv3_2_b))
parameters += [conv3_1_W, conv3_1_b, conv3_2_W, conv3_2_b]

maxpool3, maxpool3_ind = tf.nn.max_pool_with_argmax(conv3_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("maxpool3:", maxpool3.shape)

with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE):
    conv4_1_W = tf.get_variable("conv4_1_W", [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv4_1_b = tf.get_variable("conv4_1_b", [512], initializer=tf.zeros_initializer())
    conv4_2_W = tf.get_variable("conv4_2_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    conv4_2_b = tf.get_variable("conv4_2_b", [512], initializer=tf.zeros_initializer())
conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool3, conv4_1_W, strides=[1,1,1,1], padding="SAME"), conv4_1_b))
conv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_1, conv4_2_W, strides=[1,1,1,1], padding="SAME"), conv4_2_b))
parameters += [conv4_1_W, conv4_1_b, conv4_2_W, conv4_2_b]

maxpool4, maxpool4_ind = tf.nn.max_pool_with_argmax(conv4_2, ksize = [1,2,2,1], strides=[1,2,2,1], padding="SAME")
print("maxpool4:", maxpool4.shape)

with tf.variable_scope("conv5", reuse=tf.AUTO_REUSE):
    conv5_1_W = tf.get_variable("conv5_1_W", [3, 3, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
    conv5_1_b = tf.get_variable("conv5_1_b", [1024], initializer=tf.zeros_initializer())
    conv5_2_W = tf.get_variable("conv5_2_W", [3, 3, 1024, 1024], initializer=tf.contrib.layers.xavier_initializer())
    conv5_2_b = tf.get_variable("conv5_2_b", [1024], initializer=tf.zeros_initializer())
conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool4, conv5_1_W, strides=[1,1,1,1], padding="SAME"), conv5_1_b))
conv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5_1, conv5_2_W, strides=[1,1,1,1], padding="SAME"), conv5_2_b))
parameters += [conv5_1_W, conv5_1_b, conv5_2_W, conv5_2_b]
print("conv5:", conv5_2.shape)

with tf.variable_scope("upconv5", reuse=tf.AUTO_REUSE):
    upconv5_1_W = tf.get_variable("upconv5_1_W", [2, 2, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
    upconv5_1_b = tf.get_variable("upconv5_1_b", [512], initializer=tf.zeros_initializer())
    upconv5_2_W = tf.get_variable("upconv5_2_W", [3, 3, 1024, 512], initializer=tf.contrib.layers.xavier_initializer())
    upconv5_2_b = tf.get_variable("upconv5_2_b", [512], initializer=tf.zeros_initializer())
    upconv5_3_W = tf.get_variable("upconv5_3_W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer())
    upconv5_3_b = tf.get_variable("upconv5_3_b", [512], initializer=tf.zeros_initializer())
upconv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv5_2, upconv5_1_W, tf.stack([tf.shape(conv5_2)[0], 2*tf.shape(conv5_2)[1], 2*tf.shape(conv5_2)[2], 512]), strides=[1,2,2,1], padding="SAME"), upconv5_1_b))
upconv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(crop_and_concat(conv4_2, upconv5_1), upconv5_2_W, strides=[1,1,1,1], padding="SAME"), upconv5_2_b))
upconv5_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(upconv5_2, upconv5_3_W, strides=[1,1,1,1], padding="SAME"), upconv5_3_b))
parameters += [upconv5_1_W, upconv5_1_b, upconv5_2_W, upconv5_2_b, upconv5_3_W, upconv5_3_b]
print("upconv5:", upconv5_3.shape)

with tf.variable_scope("upconv4", reuse=tf.AUTO_REUSE):
    upconv4_1_W = tf.get_variable("upconv4_1_W", [2, 2, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
    upconv4_1_b = tf.get_variable("upconv4_1_b", [256], initializer=tf.zeros_initializer())
    upconv4_2_W = tf.get_variable("upconv4_2_W", [3, 3, 512, 256], initializer=tf.contrib.layers.xavier_initializer())
    upconv4_2_b = tf.get_variable("upconv4_2_b", [256], initializer=tf.zeros_initializer())
    upconv4_3_W = tf.get_variable("upconv4_3_W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
    upconv4_3_b = tf.get_variable("upconv4_3_b", [256], initializer=tf.zeros_initializer())
upconv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv5_3, upconv4_1_W, tf.stack([tf.shape(upconv5_3)[0], 2*tf.shape(upconv5_3)[1], 2*tf.shape(upconv5_3)[2], 256]), strides=[1,2,2,1], padding="SAME"), upconv4_1_b))
upconv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(crop_and_concat(conv3_2, upconv4_1), upconv4_2_W, strides=[1,1,1,1], padding="SAME"), upconv4_2_b))
upconv4_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(upconv4_2, upconv4_3_W, strides=[1,1,1,1], padding="SAME"), upconv4_3_b))
parameters += [upconv4_1_W, upconv4_1_b, upconv4_2_W, upconv4_2_b, upconv4_3_W, upconv4_3_b]
print("upconv4:", upconv4_3.shape)

with tf.variable_scope("upconv3", reuse=tf.AUTO_REUSE):
    upconv3_1_W = tf.get_variable("upconv3_1_W", [2, 2, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    upconv3_1_b = tf.get_variable("upconv3_1_b", [128], initializer=tf.zeros_initializer())
    upconv3_2_W = tf.get_variable("upconv3_2_W", [3, 3, 256, 128], initializer=tf.contrib.layers.xavier_initializer())
    upconv3_2_b = tf.get_variable("upconv3_2_b", [128], initializer=tf.zeros_initializer())
    upconv3_3_W = tf.get_variable("upconv3_3_W", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    upconv3_3_b = tf.get_variable("upconv3_3_b", [128], initializer=tf.zeros_initializer())
upconv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv4_3, upconv3_1_W, tf.stack([tf.shape(upconv4_3)[0], 2*tf.shape(upconv4_3)[1], 2*tf.shape(upconv4_3)[2], 128]), strides=[1,2,2,1], padding="SAME"), upconv3_1_b))
upconv3_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(crop_and_concat(conv2_2, upconv3_1), upconv3_2_W, strides=[1,1,1,1], padding="SAME"), upconv3_2_b))
upconv3_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(upconv3_2, upconv3_3_W, strides=[1,1,1,1], padding="SAME"), upconv3_3_b))
parameters += [upconv3_1_W, upconv3_1_b, upconv3_2_W, upconv3_2_b, upconv3_3_W, upconv3_3_b]
print("upconv3:", upconv3_3.shape)

with tf.variable_scope("upconv2", reuse=tf.AUTO_REUSE):
    upconv2_1_W = tf.get_variable("upconv2_1_W", [2, 2, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    upconv2_1_b = tf.get_variable("upconv2_1_b", [64], initializer=tf.zeros_initializer())
    upconv2_2_W = tf.get_variable("upconv2_2_W", [3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
    upconv2_2_b = tf.get_variable("upconv2_2_b", [64], initializer=tf.zeros_initializer())
    upconv2_3_W = tf.get_variable("upconv2_3_W", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    upconv2_3_b = tf.get_variable("upconv2_3_b", [64], initializer=tf.zeros_initializer())
upconv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv3_3, upconv2_1_W, tf.stack([tf.shape(upconv3_3)[0], 2*tf.shape(upconv3_3)[1], 2*tf.shape(upconv3_3)[2], 64]), strides=[1,2,2,1], padding="SAME"), upconv2_1_b))
upconv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(crop_and_concat(conv1_2, upconv2_1), upconv2_2_W, strides=[1,1,1,1], padding="SAME"), upconv2_2_b))
upconv2_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(upconv2_2, upconv2_3_W, strides=[1,1,1,1], padding="SAME"), upconv2_3_b))

with tf.variable_scope("Output", reuse=tf.AUTO_REUSE):
    Output_W = tf.get_variable("Output_W", [1, 1, 64, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Output_b = tf.get_variable("Output_b", [config.classes], initializer=tf.zeros_initializer())
Z = tf.nn.bias_add(tf.nn.conv2d(upconv2_3, Output_W, strides=[1, 1, 1, 1], padding="SAME"), Output_b)
parameters += [Output_W, Output_b]
Z = tf.argmax(Z, dimension=3)

print("TEST MODEL")
saver = tf.train.Saver()
preds = np.zeros((test_data.shape[0], config.patch_size, config.patch_size))
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, model_folder, checkpoint_name))
    n_batches = test_data.shape[0]//config.UNET_batch_size
    for batch in range(n_batches):
        print(batch)
        data_batch = test_data[batch*config.UNET_batch_size:(batch+1)*config.UNET_batch_size, :, :, :]
        feed_dict = {X: data_batch}
        preds[batch*config.UNET_batch_size:(batch+1)*config.UNET_batch_size, :, :] = sess.run(Z, feed_dict=feed_dict)

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

for i in range(N_DATA):
    image = test_data[i,:,:,:]

    label = test_label[i,:,:]

    plt.figure(figsize = (30,10))
    plt.subplot(1,3,1)
    plt.imshow(image.astype('uint8'))
    plt.subplot(1,3,2 )
    plt.imshow(label, cmap= cmap, norm = norm)
    plt.subplot(1,3,3)
    plt.imshow(preds[i,:,:] , cmap= cmap, norm = norm)
    plt.tight_layout()
    plt.colorbar()
    plt.savefig(os.path.join(config.RESULT_DIR, result_folder, str(i)+".png"), format='png')
    print(os.path.join(config.RESULT_DIR, result_folder, str(i)+".png"))
    plt.close()
