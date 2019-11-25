# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 22:57:38 2019

@author: shrey
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
#%%
print("start")
data = "C:\\Users\\shrey\\Downloads\\roof-classification\\stac\\colombia"
region_name = ['borde_rural']
categories = ['concrete_cement','healthy_metal','incomplete','irregular_metal','other']
label_list = []
patch_size = 500

save_data = "C:\\Users\\shrey\\Downloads\\roof-classification-patch"

for region in region_name:
    print("reading data", region)

    image = tiff.imread(os.path.join(data,region, region+"_ortho-cog.tif"))
    image = np.array(image)
    bulding_footprint = tiff.imread(os.path.join(data,region,"train-"+region+'.tif'))
    bulding_footprint = np.array(bulding_footprint)
    
    height, width, channels = image.shape
    
    combined_label = np.zeros((image.shape[0], image.shape[1]))
    
    print("starting combination")
    for i, cat in enumerate(categories):
        print("combining"+cat)
        label = tiff.imread(os.path.join(data,region,"train-"+region+"-"+cat+'.tif'))
        label = np.array(label)
        label = label*(i+1)
        combined_label += label
    
    np.save("train-"+region+"-combined",combined_label)
    
    number_patch_i = height//patch_size
    number_patch_j = width//patch_size
    
    for i in range(number_patch_i):
        for j in range(number_patch_j):
            
            bulding_footprint_patch = bulding_footprint[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    
            if 0.1 < np.sum(bulding_footprint_patch)/(patch_size*patch_size):
                print(i, j)
                image_patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
                label_patch = combined_label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
    
                im = tiff.imsave(os.path.join(save_data, 'image', str(i)+"_"+str(j)+".tif"), image_patch)
                im = tiff.imsave(os.path.join(save_data, 'label', str(i)+"_"+str(j)+".tif"), label_patch)

#%%
#
#patch_size = 500
#i=int(height/(patch_size*2))
#j=int(width/(patch_size*2))
#image_patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
#label_patch = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
#
#plt.figure(figsize = (30,10))
#plt.subplot(1,3,1)
#plt.imshow(image_patch[:,:,:3])
#plt.subplot(1,3,2)
#plt.imshow(label_patch)
#plt.subplot(1,3,3)
#plt.imshow(image_patch[:,:,:3])
#plt.imshow(label_patch, alpha = 0.5)
#plt.show()
#
#im = tiff.imsave('C://Users//shrey//Downloads//roof-classification-patch//test.tif', image_patch)

#%%


#%%
#columbia_dir = 'C://Users//shrey//Downloads//roof-classification-patch//columbia'
#count = 0
#for i in range(number_patch_i):
#    for j in range(number_patch_j):
#        image_patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
#        label_patch = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
#                
#        if 0.01 < np.sum(label_patch)/(patch_size*patch_size):
#            count += 1
##            plt.figure(figsize = (30,10))
##            plt.subplot(1,3,1)
##            plt.imshow(image_patch[:,:,:3])
##            plt.subplot(1,3,2)
##            plt.imshow(label_patch)
##            plt.subplot(1,3,3)
##            plt.imshow(image_patch[:,:,:3])
##            plt.imshow(label_patch, alpha = 0.5)
##            plt.show()
##            
##            print(np.sum(label_patch)/(patch_size*patch_size) )


#%%
#columbia_dir = 'C://Users//shrey//Downloads//roof-classification-patch//columbia'
#for i in range(number_patch_i):
#    for j in range(number_patch_j):
#        image_patch = image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
#        label_patch = label[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
#        
#        if 0.01 < np.sum(label_patch)/(patch_size*patch_size):
#
#            plt.figure(figsize = (30,10))
#            plt.subplot(1,3,1)
#            plt.imshow(image_patch[:,:,:3])
#            plt.subplot(1,3,2)
#            plt.imshow(label_patch)
#            plt.subplot(1,3,3)
#            plt.imshow(image_patch[:,:,:3])
#            plt.imshow(label_patch, alpha = 0.5)
#            plt.show()
#            
#            print("Enter A Key: (a):Accept, (r):Reject, (q):Quit")
#            a = input("Key: ")
#            if a=="a":
#                im = tiff.imsave(os.path.join(columbia_dir, "full", "image", str(i)+"_"+str(j)+".tif"), image_patch)
#                im = tiff.imsave(os.path.join(columbia_dir, "full", "label", str(i)+"_"+str(j)+".tif"), label_patch)
#            elif a=="r":
#                im = tiff.imsave(os.path.join(columbia_dir, "partial", "image", str(i)+"_"+str(j)+".tif"), image_patch)
#                im = tiff.imsave(os.path.join(columbia_dir, "partial", "label", str(i)+"_"+str(j)+".tif"), label_patch)
#            elif a == 'q':
#                sys.exit(0)
#            print("##################################################################")
        

