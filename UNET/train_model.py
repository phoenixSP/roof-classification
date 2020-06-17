#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.append("../")
import config
import os
import numpy as np
import tensorflow as tf

MODEL = "UNET"
LOSS_FUNCTION = "NORMAL"

model_folder = MODEL + '_' + LOSS_FUNCTION

print("LOAD DATA")
train_data = np.load(os.path.join(config.NUMPY_DIR, "train_image.npy"))
train_label = np.load(os.path.join(config.NUMPY_DIR, "train_label.npy"))
epoch_loss = []

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
upconv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv5_2, upconv5_1_W, tf.stack([tf.shape(conv5_2)[0], 2*conv5_2.shape[1].value, 2*conv5_2.shape[2].value, 512]), strides=[1,2,2,1], padding="SAME"), upconv5_1_b))
print('conv4_2', conv4_2.shape)
print('upconv5_1', upconv5_1.shape)
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
upconv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv5_3, upconv4_1_W, tf.stack([tf.shape(upconv5_3)[0], 2*upconv5_3.shape[1].value, 2*upconv5_3.shape[2].value, 256]), strides=[1,2,2,1], padding="SAME"), upconv4_1_b))
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
upconv3_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv4_3, upconv3_1_W, tf.stack([tf.shape(upconv4_3)[0], 2*upconv4_3.shape[1].value, 2*upconv4_3.shape[2].value, 128]), strides=[1,2,2,1], padding="SAME"), upconv3_1_b))
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
upconv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(upconv3_3, upconv2_1_W, tf.stack([tf.shape(upconv3_3)[0], 2*upconv3_3.shape[1].value, 2*upconv3_3.shape[2].value, 64]), strides=[1,2,2,1], padding="SAME"), upconv2_1_b))
upconv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(crop_and_concat(conv1_2, upconv2_1), upconv2_2_W, strides=[1,1,1,1], padding="SAME"), upconv2_2_b))
upconv2_3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(upconv2_2, upconv2_3_W, strides=[1,1,1,1], padding="SAME"), upconv2_3_b))
parameters += [upconv2_1_W, upconv2_1_b, upconv2_2_W, upconv2_2_b, upconv2_3_W, upconv2_3_b]
print("upconv2:", upconv2_3.shape)

with tf.variable_scope("Output", reuse=tf.AUTO_REUSE):
    Output_W = tf.get_variable("Output_W", [1, 1, 64, config.classes], initializer=tf.contrib.layers.xavier_initializer())
    Output_b = tf.get_variable("Output_b", [config.classes], initializer=tf.zeros_initializer())
Z = tf.nn.bias_add(tf.nn.conv2d(upconv2_3, Output_W, strides=[1, 1, 1, 1], padding="SAME"), Output_b)
parameters += [Output_W, Output_b]
print("Z:", Z.shape)

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=Z))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.UNET_learning_rate).minimize(loss, global_step)

print("TRAIN MODEL")
pre_saver = tf.train.Saver()
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, model_folder), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    n_batches = train_data.shape[0]//config.UNET_batch_size
    for i in range(config.UNET_n_epochs):
        total_loss = 0
        if i == 0:
            try:
                pre_saver.restore(sess, os.path.join(config.MODEL_DIR, model_folder, "model.ckpt"))
            except:
                continue

        for batch in range(n_batches):
            data_batch = train_data[batch*config.UNET_batch_size:(batch+1)*config.UNET_batch_size, :, :, :]
            label_batch = train_label[batch*config.UNET_batch_size:(batch+1)*config.UNET_batch_size, :, :]
            feed_dict = {X: data_batch, Y: label_batch}
            summary_str, _, loss_batch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step.eval())
            total_loss += loss_batch
            if not batch%10:
                print('Batch {0}: Loss: {1}'.format(batch, loss_batch))
        print('\n Epoch {0}: Loss: {1}\n'.format(i, total_loss/n_batches))
        epoch_loss.append(total_loss/n_batches)

        if i % 5 == 0:
            iter = i+1
            save_path = saver.save(sess, os.path.join(config.MODEL_DIR, model_folder, "model_"+str(iter),".ckpt"))
            epoch_loss_npy = np.array(epoch_loss)
            np.save(os.path.join(config.MODEL_DIR, model_folder, "loss_"+str(iter)), epoch_loss_npy)

    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, model_folder, "model_"+ str(config.UNET_n_epochs)+ ".ckpt"))
    np.save(os.path.join(config.MODEL_DIR, model_folder, 'loss_'+str(config.UNET_n_epochs)), epoch_loss)
