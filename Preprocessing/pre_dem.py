#*****************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Crop DEM data based on interest region
#******************************************************************************************
import numpy as np
from osgeo import gdal
import os
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import pandas as pd
import pyproj

name_images = ['srtm_beijing_1.tif', 'srtm_beijing_2.tif']
DIR_DEM = '../Data/Satellite_data/DEM'

PNG_SAVE_DIR = f'{DIR_DEM}/PNG'
if not (os.path.isdir(PNG_SAVE_DIR)):
    os.mkdir(PNG_SAVE_DIR)
else:
    print('PNG directory exists.')

ARRAY_SAVE_DIR = f'{DIR_DEM}/DATA_CROPPED'
if not (os.path.isdir(ARRAY_SAVE_DIR)):
    os.mkdir(ARRAY_SAVE_DIR)
else:
    print('Data cropped directory exists.')

grid_bb = pd.read_csv('../Data/Ground_data/2km_beijing_qgis.csv')

proj_qgis = pyproj.Proj(3857)

# Selected the cover area of 35 air quality monitoring stations on Beijing
west, north = proj_qgis(np.min(grid_bb['left'].values), np.max(grid_bb['top'].values), inverse=True)
east, south = proj_qgis(np.max(grid_bb['right'].values), np.min(grid_bb['bottom'].values), inverse=True)

for name in name_images:
    current_name = f'{DIR_DEM}/{name}'
    image = gdal.Open(current_name)
    print(current_name)
    width = image.RasterXSize
    height = image.RasterYSize
    gt = image.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    band = image.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    x = np.linspace(minx, maxx, width)
    y = np.linspace(miny, maxy, height)
    lon, lat = np.meshgrid(x, y)
    row_index, col_index = np.where((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))
    latitude = lat[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
    longitude = lon[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]

    array = image.ReadAsArray().astype(np.float)
    if np.any(array == nodata):
        array[array == nodata] = np.nan
    array = array[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]

    count = 0
    image_array = np.zeros((latitude.shape[0]*latitude.shape[1], 3))
    for i in range(latitude.shape[0]):
        for j in range(latitude.shape[1]):
            image_array[count, 0] = latitude[i, j]
            image_array[count, 1] = longitude[i, j]
            image_array[count, 2] = array[i, j]
            count += 1

    image_array.tofile(f'{ARRAY_SAVE_DIR}/{name[:-4]}.dat')
    with open(f'{ARRAY_SAVE_DIR}/info_{name[:-4]}.txt', 'w') as f:
        f.write('{}x{}'.format(latitude.shape[0], latitude.shape[1]))
        
    scale_image = MinMaxScaler((0, 255))
    image_data = np.asarray(array).copy()
    rescaled = scale_image.fit_transform(image_data).astype(np.uint8)
    image = Image.fromarray(rescaled)
    image.save(f'{PNG_SAVE_DIR}/{name[:-4]}.png')
##