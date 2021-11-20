#*************************************************************************************************
# Author: Luis Ernesto Colchado 
# Email: luis.colchado@ucsp.edu.pe
# Description: Crop NTL data based on interest region
# Based on https://hdfeos.org/zoo/MORE/LPDAAC/MCD/MCD19A2.A2010010.h25v06.006.2018047103710.hdf.py
#*************************************************************************************************

import numpy as np
from h5py import File
import pyproj
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import os
import pandas as pd
##

VNP46A1_DIR = '../Data/Satellite_data/VNP46A1'

if not (os.path.isdir(VNP46A1_DIR)):
    os.mkdir(VNP46A1_DIR)
else:
    print('VNP46A1 directory exists.')

SP_PNG_SAVE_DIR = VNP46A1_DIR+'/PNG'

SP_HDF_SAVE_DIR = VNP46A1_DIR+'/HDF'

if not (os.path.isdir(SP_PNG_SAVE_DIR)):
    os.mkdir(SP_PNG_SAVE_DIR)
else:
    print('PNG directory exists.')

ARRAY_SAVE_DIR = VNP46A1_DIR+'/DATA_CROPPED'

if not (os.path.isdir(ARRAY_SAVE_DIR)):
    os.mkdir(ARRAY_SAVE_DIR)
else:
    print('Data cropped directory exists.')

product_type = 'VNP46A1'

grid_bb = pd.read_csv('../Data/Ground_data/2km_beijing_qgis.csv')

proj_qgis = pyproj.Proj(3857)

# Selected the cover area of AQM stations
west, north = proj_qgis(np.min(grid_bb['left'].values), np.max(grid_bb['top'].values), inverse=True)
east, south = proj_qgis(np.max(grid_bb['right'].values), np.min(grid_bb['bottom'].values), inverse=True)

# Name datafield for get Lightime
datafield_name = 'DNB_At_Sensor_Radiance_500m'

for hdf_file in os.listdir(SP_HDF_SAVE_DIR):
    if hdf_file.endswith('.h5'):
        print(hdf_file)
        try:
            file = File(SP_HDF_SAVE_DIR+'/'+hdf_file, 'r')
            grids = file['HDFEOS']['GRIDS']
            vnp = grids['VNP_Grid_DNB']
            data_fields = vnp['Data Fields']
            data_dbn = data_fields[datafield_name]
            data = data_dbn[:, :].astype(float)

            attrs = data_dbn.attrs
            lna = attrs["long_name"]
            long_name = lna[0]
            vmax = attrs["valid_max"]
            valid_max = vmax[0]
            vmin = attrs["valid_min"]
            valid_min = vmin[0]
            fva = attrs["_FillValue"]   
            _FillValue = fva[0]
            sfa = attrs["scale_factor"]
            scale_factor = sfa[0]
            ua = attrs["units"]
            units = ua[0]
            aoa = attrs["add_offset"]
            add_offset = aoa[0]

            # Apply the attributes to the data.
            invalid = np.logical_or(data > valid_max, data < valid_min)
            invalid = np.logical_or(invalid, data == _FillValue)
            data[invalid] = np.nan
            data = (data - add_offset) * scale_factor
            data = np.ma.masked_array(data, np.isnan(data))

            fileMetadata = file['HDFEOS INFORMATION']['StructMetadata.0'][()].split()
            fileMetadata = [m.decode('utf-8') for m in fileMetadata]

            ulc = [i for i in fileMetadata if 'UpperLeftPointMtrs' in i][0]
            ulc_lat = float(ulc.split('=(')[-1].replace(')', '').split(',')[0]) / 1000000
            ulc_lon = float(ulc.split('=(')[-1].replace(')', '').split(',')[1]) / 1000000

            lrc = [i for i in fileMetadata if 'LowerRightMtrs' in i][0]
            lrc_lat = float(lrc.split('=(')[-1].replace(')', '').split(',')[0]) / 1000000
            lrc_lon = float(lrc.split('=(')[-1].replace(')', '').split(',')[1]) / 1000000

            # wgs84 = pyproj.Proj("+init=EPSG:4326")
            sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
            x0, y0 = sinu(ulc_lat, ulc_lon)
            x1, y1 = sinu(lrc_lat, lrc_lon)

            x_metersxpix = np.abs((x1 - x0) / 2400)
            y_metersxpix = np.abs((y1 - y0) / 2400)

            nx, ny = data.shape
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
            xv, yv = np.meshgrid(x, y)

            lon, lat = sinu(xv, yv, inverse=True)

            row_index, col_index = np.where((lat >= south) & (lat <= north) & (lon >= west) & (lon <= east))
            latitude = lat[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
            longitude = lon[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]
            data = data[np.min(row_index):np.max(row_index)+1, np.min(col_index):np.max(col_index)+1]

            name = hdf_file.split('.')[0] + '_' + hdf_file.split('.')[1] + '_' + hdf_file.split('.')[2] + '_' + hdf_file.split('.')[4]
            
            count = 0
            image_array = np.zeros((latitude.shape[0]*latitude.shape[1], 3))
            for i in range(latitude.shape[0]):
                for j in range(latitude.shape[1]):
                    image_array[count, 0] = latitude[i, j]
                    image_array[count, 1] = longitude[i, j]
                    image_array[count, 2] = data[i, j]
                    count += 1

            image_array.tofile(ARRAY_SAVE_DIR+'/vnp_cropped_'+name+'.dat')
            with open(ARRAY_SAVE_DIR+'/info.txt', 'w') as f:
                f.write('{}x{}'.format(latitude.shape[0], latitude.shape[1]))

            scale = MinMaxScaler((0, 255))
            image_data = np.asarray(data).copy()
            rescaled = scale.fit_transform(image_data).astype(np.uint8)
            image = Image.fromarray(rescaled)
            image.save(SP_PNG_SAVE_DIR+'/'+name+".png")

        except Exception as e:
            print(e)
##