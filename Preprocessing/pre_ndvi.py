
#**************************************************
# Based on https://hdfeos.org/zoo/LAADS_MOD_py.php
#**************************************************
import re
import pyproj

import numpy as np

from pyhdf.SD import SD, SDC

import json
from sklearn.preprocessing import MinMaxScaler

from PIL import Image   
import os

#PROJECT_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
MOD13A2_DIR = '../Data/Satellite_data/MOD13A2'

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

# Selected the cover area of 35 air quality monitoring stations on Beijing
north, east = 40.3415493967356, 116.67319258392523
south, west = 39.86056186496943, 116.1701360248183 #39.49534528, 115.58538899

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

            name = hdf_file.split('.')[0] + '_' + hdf_file.split('.')[1] + '_' + hdf_file.split('.')[4]

            with open(ARRAY_SAVE_DIR+'/ndvi_cropped_data_' + name + '.txt', 'w') as f:
                for i in range(latitude.shape[0]):
                    for j in range(longitude.shape[1]):
                        format_ = '%f/%f/%f|' if j < latitude.shape[1] - 1 else '%f/%f/%f'
                        f.write(format_ % (latitude[i, j], longitude[i, j], data[i, j]))
                    f.write('\n')
            
            scale_image = MinMaxScaler((0, 255))
            image_data = np.asarray(data).copy()
            rescaled = scale_image.fit_transform(image_data).astype(np.uint8)
            image = Image.fromarray(rescaled)
            image.save(PNG_SAVE_DIR + '/' + name + '.png')

        except Exception as e:
            print(e)

##