#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config
import os
import numpy as np
import tensorflow as tf
import tifffile as tiff

MODEL = "FCN"
LOSS_FUNCTION = "NORMAL"

model_folder = MODEL + '_' + LOSS_FUNCTION

print("LOAD DATA")

train_data = np.load(os.path.join(config.NUMPY_DIR, "train_image.npy"))
train_label = np.load(os.path.join(config.NUMPY_DIR, "train_label.npy"))
epoch_loss = []

print("BUILD MODEL")
tf.reset_default_graph()
parameters = []
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, config.patch_size, config.patch_size, config.channels], name="inputs")
    # Y = tf.placeholder(tf.int32, [None, config.patch_size, config.patch_size, config.classes], name="labels")
    Y = tf.placeholder(tf.int32, [None, config.patch_size, config.patch_size], name="labels")


with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
    conv1_1_W = tf.get_variable("conv1_1_W", [3, 3, config.channels, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_1_b = tf.get_variable("conv1_1_b", [64], initializer=tf.zeros_initializer())
    conv1_2_W = tf.get_variable("conv1_2_W", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer())
    conv1_2_b = tf.get_variable("conv1_2_b", [64], initializer=tf.zeros_initializer())
conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, conv1_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv1_1_b))
conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_1, conv1_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv1_2_b))
parameters += [conv1_1_W, conv1_1_b, conv1_2_W, conv1_2_b]

maxpool1 = tf.nn.max_pool(conv1_2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")
print("maxpool1:", maxpool1.shape)

with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
    conv2_1_W = tf.get_variable("conv2_1_W", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_1_b = tf.get_variable("conv2_1_b", [128], initializer=tf.zeros_initializer())
    conv2_2_W = tf.get_variable("conv2_2_W", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
    conv2_2_b = tf.get_variable("conv2_2_b", [128], initializer=tf.zeros_initializer())
conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1, conv2_1_W, strides=[1, 1, 1, 1], padding="SAME"), conv2_1_b))
conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_1, conv2_2_W, strides=[1, 1, 1, 1], padding="SAME"), conv2_2_b))
parameters += [conv2_1_W, conv2_1_b, conv2_2_W, conv2_2_b]

maxpool2 = tf.nn.max_pool(conv2_2, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")
print("maxpool2:", maxpool2.shape)

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
parameters += [conv3_1_W, conv3_1_b, conv3_2_W, conv3_2_b, conv3_3_W, conv3_3_b]

maxpool3 = tf.nn.max_pool(conv3_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")
print("maxpool3:", maxpool3.shape)

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
parameters += [conv4_1_W, conv4_1_b, conv4_2_W, conv4_2_b, conv4_3_W, conv4_3_b]

maxpool4 = tf.nn.max_pool(conv4_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")
print("maxpool4:", maxpool4.shape)

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
parameters += [conv5_1_W, conv5_1_b, conv5_2_W, conv5_2_b, conv5_3_W, conv5_3_b]

maxpool5 = tf.nn.max_pool(conv5_3, ksize = [1,2,2,1], strides=[1, 2, 2, 1], padding="SAME")
print("maxpool5:", maxpool5.shape)

with tf.variable_scope("fc6", reuse=tf.AUTO_REUSE):
    fc6_W = tf.get_variable("fc6_W", [7, 7, 512, 4096], initializer=tf.contrib.layers.xavier_initializer())
    fc6_b = tf.get_variable("fc6_b", [4096], initializer=tf.zeros_initializer())
fc6 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool5, fc6_W, strides=[1, 1, 1, 1], padding="SAME"), fc6_b))
print("fc6:", fc6.shape)
parameters += [fc6_W, fc6_b]

with tf.variable_scope("fc7", reuse=tf.AUTO_REUSE):
    fc7_W = tf.get_variable("fc7_W", [1, 1, 4096, 4096], initializer=tf.contrib.layers.xavier_initializer())
    fc7_b = tf.get_variable("fc7_b", [4096], initializer=tf.zeros_initializer())
fc7 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(fc6, fc7_W, strides=[1, 1, 1, 1], padding="SAME"), fc7_b))
print("fc7:", fc7.shape)
parameters += [fc7_W, fc7_b]

with tf.variable_scope("fc8", reuse=tf.AUTO_REUSE):
    fc8_W = tf.get_variable("fc8_W", [1, 1, 4096, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    fc8_b = tf.get_variable("fc8_b", [config.classes], initializer=tf.zeros_initializer())
fc8 = tf.nn.bias_add(tf.nn.conv2d(fc7, fc8_W, strides=[1, 1, 1, 1], padding="SAME"), fc8_b)
parameters += [fc8_W, fc8_b]

with tf.variable_scope("Ufc7", reuse=tf.AUTO_REUSE):
    Ufc7_W = tf.get_variable("Ufc7_W", [4, 4, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Ufc7 = tf.nn.conv2d_transpose(fc8, Ufc7_W, tf.stack([tf.shape(maxpool4)[0], tf.shape(maxpool4)[1], tf.shape(maxpool4)[2], config.classes]), strides=[1, 2, 2, 1], padding="SAME")
parameters += [Ufc7_W]

with tf.variable_scope("Spool4", reuse=tf.AUTO_REUSE):
    Spool4_W = tf.get_variable("Spool4_W", [1, 1, 512, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Spool4_b = tf.get_variable("Spool4_b", [config.classes], initializer=tf.zeros_initializer())
Spool4 = tf.nn.bias_add(tf.nn.conv2d(maxpool4, Spool4_W, strides=[1, 1, 1, 1], padding="SAME"), Spool4_b)
parameters += [Spool4_W, Spool4_b]

fuse1 = tf.add(Ufc7, Spool4)
print("fuse1:", fuse1.shape)

with tf.variable_scope("Ufuse1", reuse=tf.AUTO_REUSE):
    Ufuse1_W = tf.get_variable("Ufuse1_W", [4, 4, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Ufuse1 = tf.nn.conv2d_transpose(fuse1, Ufuse1_W, tf.stack([tf.shape(maxpool3)[0], tf.shape(maxpool3)[1], tf.shape(maxpool3)[2], config.classes]), strides=[1, 2, 2, 1], padding="SAME")
parameters += [Ufuse1_W]

with tf.variable_scope("Spool3", reuse=tf.AUTO_REUSE):
    Spool3_W = tf.get_variable("Spool3_W", [1, 1, 256, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Spool3_b = tf.get_variable("Spool3_b", [config.classes], initializer=tf.zeros_initializer())
Spool3 = tf.nn.bias_add(tf.nn.conv2d(maxpool3, Spool3_W, strides=[1, 1, 1, 1], padding="SAME"), Spool3_b)
parameters += [Spool3_W, Spool3_b]

fuse2 = tf.add(Ufuse1, Spool3)
print("fuse2:", fuse2.shape)

with tf.variable_scope("Output", reuse=tf.AUTO_REUSE):
    Output_W = tf.get_variable("Output_W", [16, 16, config.classes, config.classes], initializer=tf.contrib.layers.xavier_initializer())
Z = tf.nn.conv2d_transpose(fuse2, Output_W, tf.stack([tf.shape(maxpool4)[0], config.patch_size, config.patch_size, config.classes]), strides=[1, 8, 8, 1], padding="SAME")
print("Z:", Z.shape)
parameters += [Output_W]

print("Y:", Y.shape)
with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z))
#    print('Inside loss', tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z).shape)
#    print('After reduce mean',loss.shape)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.FCN_learning_rate).minimize(loss, global_step)

print("TRAIN MODEL")
pre_saver = tf.train.Saver()
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "FCN"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # try:
    #     pre_saver.restore(sess, os.path.join(config.MODEL_DIR, "FCN", "model.ckpt"))
    # except:
    #     continue
    n_batches = train_data.shape[0]//config.FCN_batch_size
    for i in range(config.FCN_n_epochs):
        total_loss = 0
        if i % 10 == 1:
            try:
                checkpoint_filename = "model_" + str(i) ".ckpt"
                pre_saver.restore(sess, os.path.join(config.MODEL_DIR, model_folder, checkpoint_filename))
            except:
                continue

        for batch in range(n_batches):
            data_batch = train_data[batch*config.FCN_batch_size:(batch+1)*config.FCN_batch_size, :, :, :]
            label_batch = train_label[batch*config.FCN_batch_size:(batch+1)*config.FCN_batch_size, :, :]
            feed_dict = {X: data_batch, Y: label_batch}
            summary_str, _, loss_batch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step.eval())
            total_loss += loss_batch
            if not batch%10:
                print('Batch {0}: Loss: {1}'.format(batch, loss_batch))
        print('\n Epoch {0}: Loss: {1}\n'.format(i, total_loss/n_batches))
        epoch_loss.append(total_loss/n_batches)
        if i % 10 == 0:
            try:
                checkpoint_filename = "model_" + str(i+1) ".ckpt"
                save_path = saver.save(sess, os.path.join(config.MODEL_DIR, model_folder, checkpoint_filename))
                epoch_loss_npy = np.array(epoch_loss)
                epoch_loss_npy_filename = 'loss_'+str(i+1)
                np.save(os.path.join(config.MODEL_DIR, model_folder, epoch_loss_npy_filename), epoch_loss_npy)
            except:
                continue

    summary_writer.close()
    epoch_loss_npy_filename = 'loss_'+str(config.FCN_n_epochs) 
    checkpoint_filename = "model_" + str(config.FCN_n_epochs) ".ckpt"
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, model_folder, checkpoint_filename))
    np.save(os.path.join(config.MODEL_DIR, model_folder, epoch_loss_npy_filename), epoch_loss)
