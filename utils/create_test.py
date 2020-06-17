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

save_data = "./TRAIN"
data_dir = os.listdir(os.path.join(save_data, "IMAGE"))
n_data = len(data_dir)

image_data_npy = np.zeros((n_data, patch_size, patch_size, 3))
label_onehot_npy = np.zeros((n_data, patch_size, patch_size, classes))
label_npy = np.zeros((n_data, patch_size, patch_size))

print("Creating directory")

for i, file in enumerate(data_dir):
    if i%10 == 0:
        print(i)

    image_data = tiff.imread(os.path.join(save_data,"IMAGE", file))
    image_data = np.array(image_data)
    image_data_npy[i, :, :, :] = image_data

    label = tiff.imread(os.path.join(save_data,"LABEL", file))
    label = np.array(label)
    label_npy[i,:,:] = label


    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            cat = int(label[x,y])
            label_onehot_npy[i,x,y,cat] = 1

image_data_npy = image_data_npy.astype('int16')
label_onehot_npy = label_onehot_npy.astype('int16')
label_npy = label_npy.astype('int16')

np.save(os.path.join(save_data,"train_data"), image_data_npy)
np.save(os.path.join(save_data,"train_label_onehot"), label_onehot_npy)
np.save(os.path.join(save_data,"train_label"), label_npy)
