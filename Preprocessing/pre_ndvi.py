
#*************************************************************************************************
# Author: Luis Ernesto Colchado 
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Crop NTL data based on interest region
# Based on https://hdfeos.org/zoo/LAADS_MOD_py.php
#*************************************************************************************************
import pyproj
import re
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

import json
from sklearn.preprocessing import MinMaxScaler
from PIL import Image   
import os

MOD13A2_DIR = '../Data/Beijing/Satellite_data/MOD13A2'

if not (os.path.isdir(MOD13A2_DIR)):
    os.mkdir(MOD13A2_DIR)
else:
    print('MOD13A2 directory exists.')

PNG_SAVE_DIR = MOD13A2_DIR+'/PNG'

HDF_SAVE_DIR = MOD13A2_DIR+'/HDF'

if not (os.path.isdir(PNG_SAVE_DIR)):
    os.mkdir(PNG_SAVE_DIR)
else:
    print('PNG directory exists.')

ARRAY_SAVE_DIR = MOD13A2_DIR+'/DATA_CROPPED'

if not (os.path.isdir(ARRAY_SAVE_DIR)):
    os.mkdir(ARRAY_SAVE_DIR)
else:
    print('Data cropped directory exists.')

product_type = 'MOD13A2'

# Selected the cover area of AQM stations
grid_bb = pd.read_csv('../Data/Beijing/2km_beijing_qgis.csv')

proj_qgis = pyproj.Proj(3857)
west, north = proj_qgis(np.min(grid_bb['left'].values), np.max(grid_bb['top'].values), inverse=True)
east, south = proj_qgis(np.max(grid_bb['right'].values), np.min(grid_bb['bottom'].values), inverse=True)

# Time interval
#!start_date = '2015-01-01 00:00'
#!end_date = '2016-12-31 23:59'

# Name datafield for get NDVI index (1 km resolution pixel)
datafield_name = '1 km 16 days NDVI'

# NVDI data to (-1, 1) range
scale = MinMaxScaler(-1, 1)

for hdf_file in os.listdir(HDF_SAVE_DIR):
    if hdf_file.endswith('.hdf'):
        print(hdf_file)
        try:

            hdf = SD(HDF_SAVE_DIR+'/'+hdf_file, SDC.READ)
            data = hdf.select(datafield_name)[:, :].astype(np.float)

            # Normalize NDVI data on range -1 to 1
            data = (((data - np.min(data)) / (np.max(data) - np.min(data))) * 2) - 1

            # Get Latitude and Longitude
            fattrs = hdf.attributes(full=1)
            ga = fattrs["StructMetadata.0"]
            gridmeta = ga[0]
            ul_regex = re.compile(r'''UpperLeftPointMtrs=\(
                                         (?P<upper_left_x>[+-]?\d+\.\d+)
                                         ,
                                         (?P<upper_left_y>[+-]?\d+\.\d+)
                                         \)''', re.VERBOSE)

            match = ul_regex.search(gridmeta)
            x0 = np.float(match.group('upper_left_x'))
            y0 = np.float(match.group('upper_left_y'))

            lr_regex = re.compile(r'''LowerRightMtrs=\(
                                         (?P<lower_right_x>[+-]?\d+\.\d+)
                                         ,
                                         (?P<lower_right_y>[+-]?\d+\.\d+)
                                         \)''', re.VERBOSE)
            match = lr_regex.search(gridmeta)
            x1 = np.float(match.group('lower_right_x'))
            y1 = np.float(match.group('lower_right_y'))

            nx, ny = data.shape
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
            xv, yv = np.meshgrid(x, y)

            sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
            #wgs84 = pyproj.Proj("+init=EPSG:4326")
            lon, lat = sinu(xv, yv, inverse=True)

            # Crop images
            row_index, col_index = np.where((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))
            latitude = lat[np.min(row_index):np.max(row_index), np.min(col_index):np.max(col_index)]
            longitude = lon[np.min(row_index):np.max(row_index), np.min(col_index):np.max(col_index)]
            data = data[np.min(row_index):np.max(row_index), np.min(col_index):np.max(col_index)]

            name = hdf_file.split('.')[0] + '_' + hdf_file.split('.')[1] + '_' +  hdf_file.split('.')[2] + '_' + hdf_file.split('.')[4]

            count = 0
            image_array = np.zeros((latitude.shape[0]*latitude.shape[1], 3))
            for i in range(latitude.shape[0]):
                for j in range(latitude.shape[1]):
                    image_array[count, 0] = latitude[i, j]
                    image_array[count, 1] = longitude[i, j]
                    image_array[count, 2] = data[i, j]
                    count += 1

            image_array.tofile(ARRAY_SAVE_DIR+'/ndvi_cropped_'+name+'.dat')
            with open(ARRAY_SAVE_DIR+'/info.txt', 'w') as f:
                f.write('{}x{}'.format(latitude.shape[0], latitude.shape[1]))
            
            scale_image = MinMaxScaler((0, 255))
            image_data = np.asarray(data).copy()
            rescaled = scale_image.fit_transform(image_data).astype(np.uint8)
            image = Image.fromarray(rescaled)
            image.save(PNG_SAVE_DIR + '/' + name + '.png')

        except Exception as e:
            print(e)

##