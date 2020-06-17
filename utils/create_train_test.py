# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 16:14:50 2019
@author: shrey
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
import json
import random
import shutil

data = "./image"
region = 'borde_soacha'
categories = ['concrete_cement','healthy_metal','incomplete','irregular_metal','other']
label_list = []


classes = 6
patch_size = 512

save_data = "./TEST"
test_data = os.listdir(os.path.join(save_data, "IMAGE"))
n_test_data = len(test_data)


test_data_npy = np.zeros((n_test_data, patch_size, patch_size, 3))
test_label_onehot_npy = np.zeros((n_test_data, patch_size, patch_size, classes))
test_label_npy = np.zeros((n_test_data, patch_size, patch_size))

print("Creating test directory")

for i, file in enumerate(test_data):
    if i%10 == 0:
        print(i)

    test_data = tiff.imread(os.path.join(save_data,"IMAGE", file))
    test_data = np.array(test_data)
    test_data_npy[i, :, :, :] = test_data

    test_label = tiff.imread(os.path.join(save_data,"LABEL", file))
    test_label = np.array(test_label)
    test_label_npy[i,:,:] = test_label

    for x in range(test_label.shape[0]):
        for y in range(test_label.shape[1]):
            cat = int(test_label[x,y])
            test_label_onehot_npy[i,x,y,cat] = 1

test_data_npy = test_data_npy.astype('int16')
test_label_onehot_npy = test_label_onehot_npy.astype('int16')
test_label_npy = test_label_npy.astype('int16')

np.save(os.path.join(save_data,"test_data"), test_data_npy)
np.save(os.path.join(save_data,"test_label_onehot"), test_label_onehot_npy)
np.save(os.path.join(save_data,"test_label"), test_label_npy)


# for i, file in enumerate(rem_data):
#     if i%10 == 0:
#         print(i)
#
#     try:
#         image = tiff.imread(os.path.join(save_data,"image", file))
#         image = np.array(image)
#         dst = shutil.move(os.path.join(save_data,"image", file), os.path.join(save_data,"TEST","IMAGE"))
#         label = tiff.imread(os.path.join(save_data,"label", file))
#         label = np.array(label)
#         dst = shutil.move(os.path.join(save_data,"label", file), os.path.join(save_data,"TEST","LABEL"))
#
#     except:
#         print('Either image or label missing for ', file)


# all_files = os.listdir(os.path.join(save_data,"image"))
# n_data = len(all_files)
# random.seed(19)
# random.shuffle(all_files)
# n_train_data = int(n_data*0.7)
# n_test_data = n_data - n_train_data
#
# train_files = sorted(all_files[:n_train_data])
# test_files = sorted(all_files[n_train_data:])
#
#
# train_data_npy = np.zeros((n_train_data, patch_size, patch_size, 3))
# train_label_onehot_npy = np.zeros((n_train_data, patch_size, patch_size, classes))
# train_label_npy = np.zeros((n_train_data, patch_size, patch_size))
#
# print("Creating train directory")
# for i, file in enumerate(train_files):
#     if i%10 == 0:
#         print(i)
#
#     train_data = tiff.imread(os.path.join(save_data,"image", file))
#     train_data = np.array(train_data)
#     train_data_npy[i, :, :, :] = train_data
#
#     dst = shutil.move(os.path.join(save_data,"image", file), os.path.join(save_data,"TRAIN","IMAGE"))
#     #im = tiff.imsave(os.path.join(save_data,"TRAIN","IMAGE", file), train_data)
#
#     label_data = tiff.imread(os.path.join(save_data,"label", file))
#     label_data = np.array(label_data)
#
#     train_label_npy[i,:,:] = label_data
#
#     for x in range(label_data.shape[0]):
#         for y in range(label_data.shape[1]):
#             cat = int(label_data[x,y])
#             train_label_onehot_npy[i,x,y,cat] = 1
#
#     dst = shutil.move(os.path.join(save_data,"label", file), os.path.join(save_data,"TRAIN","LABEL"))
#     #patch_label = tiff.imsave(os.path.join(save_data,"TRAIN","LABEL", file), train_label_npy[i,:,:])
#
# np.save(os.path.join(save_data,"TRAIN","train_image"), train_data_npy)
# np.save(os.path.join(save_data,"TRAIN","train_label_onehot"), train_label_onehot_npy)
# np.save(os.path.join(save_data,"TRAIN","train_label"), train_label_npy)

