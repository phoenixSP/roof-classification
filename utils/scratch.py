#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
import config
import os
import numpy as np
import tensorflow as tf
import tifffile as tiff
import matplotlib.pyplot as plt

#
# loss_info_1 = np.load(os.path.join(config.MODEL_DIR, "FCN_EXT_40.npy"))
# print(loss_info_1[0], loss_info_1[-1], len(loss_info_1))
#
# loss_info_2 = np.load(os.path.join(config.MODEL_DIR, "FCN_EXT_21.npy"))
# print(loss_info_2[0], loss_info_2[-1], len(loss_info_2))

loss_info = np.zeros(40)
loss_info_1 = np.load(os.path.join(config.MODEL_DIR, "FCN_EXT_40.npy"))
loss_info_2 = np.load(os.path.join(config.MODEL_DIR, "FCN_EXT_21.npy"))

print(len(loss_info_1))
print(loss_info_1[0], loss_info_1[-1])

loss_info[:21] = loss_info_2
loss_info[21:] = loss_info_1

np.save(os.path.join(config.MODEL_DIR, "FCN_EXT_40"), loss_info)
