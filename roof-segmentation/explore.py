# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:57:38 2019

@author: shrey
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
#%%
data = "C:\\Users\\shrey\\Downloads\\roof_classification\\stac\\colombia\\borde_rural"

image = tiff.imread(os.path.join(data,"borde_rural_ortho-cog.tif"))
image = np.array(image)
height, width, channels = image.shape
label = tiff.imread(os.path.join(data,"labels.tif"))
#%%
patch_size = 500
i=int(height/(patch_size*2))
j=int(width/(patch_size*2))
image_patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
label_patch = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

plt.figure(figsize = (30,10))
plt.subplot(1,3,1)
plt.imshow(image_patch[:,:,:3])
plt.subplot(1,3,2)
plt.imshow(label_patch)
plt.subplot(1,3,3)
plt.imshow(image_patch[:,:,:3])
plt.imshow(label_patch, alpha = 0.5)
plt.show()