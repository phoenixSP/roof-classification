# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:14:12 2019

@author: shrey
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import sys
import json 



data = "C:\\Users\\shrey\\Downloads\\roof-classification\\stac\\colombia"
region_name = ['borde_soacha']

for region in region_name:
    image = tiff.imread(os.path.join(data,region, region+"_ortho-cog.tif"))
    
    with open(os.path.join(data,region, "train-"+region+".geojson")) as f_geojson:
        geo_json = json.load(f_geojson)
    
    concrete_cement = []
    healthy_metal = []
    incomplete = []
    irregular_metal = []
    other = []
    
    for item in geo_json['features']:
        if item['properties']['roof_material'] == 'concrete_cement':
            concrete_cement.append(item)
        elif item['properties']['roof_material'] == 'healthy_metal':
            healthy_metal.append(item)
        elif item['properties']['roof_material'] == 'incomplete':
            incomplete.append(item)
        elif item['properties']['roof_material'] == 'irregular_metal':
            irregular_metal.append(item)
        elif item['properties']['roof_material'] == 'other':
            other.append(item)
        

    categories = ['concrete_cement','healthy_metal','incomplete','irregular_metal','other']
    categories_list = [concrete_cement,healthy_metal,incomplete,irregular_metal,other]
    
    for i, cat in enumerate(categories):
        
        material_geojson = {'type': 'FeatureCollection', 'features' : categories_list[i]}
        
        with open(os.path.join(data,region,"train-"+region+"-"+cat+'.geojson'), 'w') as f:
            json.dump(material_geojson, f)
