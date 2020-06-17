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

MODEL_NAME = 'FCN_EXT'
ITER = 40

model_info_file = MODEL_NAME + '_' + str(ITER) + '.npy' 
loss_filename =  MODEL_NAME + '_' + str(ITER) + '.png' 

if not os.path.exists(os.path.join(config.RESULT_DIR, MODEL_NAME, "PLOTS")):
    os.makedirs(os.path.join(config.RESULT_DIR,  MODEL_NAME, "PLOTS"))
#
loss_info = np.load(os.path.join(config.MODEL_DIR, model_info_file))

iter = range(40)
fig = plt.figure()
plt.plot(iter, loss_info)
plt.title('Train loss for {} for {} epochs'.format(MODEL_NAME, ITER))
plt.xlabel('Iteration')
plt.ylabel('Loss value')
plt.savefig(os.path.join(config.RESULT_DIR, MODEL_NAME, 'PLOTS', loss_filename), format='png')
