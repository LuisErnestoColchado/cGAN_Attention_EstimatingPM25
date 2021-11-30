# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Match meteorological condition, satellite products and pollution data
# ******************************************************************************************

from typing import Coroutine
from PIL.Image import PERSPECTIVE
from h5py._hl.base import Empty
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import timedelta
from point import Point
from coord import Coord
from bounding_box import Bounding_box
import json 

##


class setting:
    NDVI_DIR = '../Data/Satellite_data/MOD13A2/DATA_CROPPED'
    VNP_DIR = '../Data/Satellite_data/VNP46A1/DATA_CROPPED'
    DEM_DIR = '../Data/Satellite_data/DEM/DATA_CROPPED'
    filename_data = '../Data/Ground_data/preprocessed_data.csv'
    filename_points = '../Data/Ground_data/points.csv'
    #number_img_ndvi = 2
    #number_img_ntl = 2
    number_img_dem = 2
    knn = 7
    normalize_meo = ['temp', 'pres', 'dewp', 'wd', 'ws']
    ndvi_interval = 16


def get_data(DATA_DIR, start_date, end_date, interval=1):
    days = (end_date - start_date).days
    #print(days)
    #dates = [str(start_date + timedelta(days=x))[:10] for x in range(0, days+1, interval)]
    #print(dates)
    #data = {k: [] for k in dates}
    #return data
    data = {}
    print(f'Loading data from {DATA_DIR} ...')
    #with tqdm(total=days) as bar:
    for file in os.listdir(DATA_DIR):
        if file.endswith('.dat'):
            year_days = file.split('_')[3][1:]
            year = int(year_days[:4])
            days = int(year_days[4:])
            current_date = pd.to_datetime(f'{year - 1}-12-31')
            current_date += timedelta(days=days)
            
            if current_date >= start_date and current_date <= end_date:
                current_date = str(current_date)[:-9]
                #print(str(current_date))
                array = np.fromfile(f'{DATA_DIR}/{file}', dtype=float)
                array = array.reshape(int(len(array) / 3), 3)
                #prev_array = data[current_date]
                #print(len(prev_array))
                if current_date not in data:
                    data.update({current_date: array})
                else:
                    prev_array = data[current_date]
                    concat = np.concatenate([prev_array, array], axis=0)
                    data.update({current_date: concat})
                #print(days, current_date)
                #data.update({current_date: concat})
                #data.update({str(current_date)[:-9]+'_'+file.split('_')[4]: array})
                   # bar.update(1)

    #!with open(DATA_DIR + '/info.txt', 'r') as f:
    #!    nrows, ncolumns = f.readline().split('x')
    '''
    with tqdm(total=len(os.listdir(DATA_DIR))) as bar:
        for file in os.listdir(DATA_DIR):
            if file.endswith('.dat'):
                year_days = file.split('_')[3][1:]
                year = int(year_days[:4])
                days = int(year_days[4:])
                current_date = pd.to_datetime(str(year - 1) + '-12-31')
                current_date += timedelta(days=days)
                #print(str(current_date))
                array = np.fromfile(DATA_DIR + '/' + file, dtype=float)
                array = array.reshape(int(len(array) / 3), 3)
                data.update({str(current_date)[:-9]+'_'+file.split('_')[4]: array})
                bar.update(1)
    '''
    print('Loaded!')
    return data#!, int(nrows), int(ncolumns)


# Read Satellite data preprocessed 

data = pd.read_csv(setting.filename_data)
data.datetime = pd.to_datetime(data.datetime)
start_date = min(data['datetime'])
end_date = max(data['datetime'])

print(f'Loading data from {setting.DEM_DIR} ...')
dem_data = np.zeros(shape=(0, 3))
for file in os.listdir(setting.DEM_DIR):
    if file.endswith('.dat'):
        dem_array = np.fromfile(f'{setting.DEM_DIR}/{file}', dtype=float)
        
        dem_array = dem_array.reshape(int(len(dem_array) / 3), 3)
        #print(len(dem_array))
        dem_data = np.concatenate([dem_data, dem_array], axis=0)
print('Loaded!')
#for j in range(setting.number_img_dem):
    

    #dem_array1 = np.fromfile(setting.DEM_DIR + '/srtm_beijing_1.dat', dtype=float)
    
    #dem_array2 = np.fromfile(setting.DEM_DIR + '/srtm_beijing_2.dat', dtype=float)
    #dem_array2 = dem_array2.reshape(int(len(dem_array2) / 3), 3)

data_ndvi = get_data(setting.NDVI_DIR, start_date, end_date, setting.ndvi_interval)
#print(data_ndvi.keys())

data_ntl = get_data(setting.VNP_DIR, start_date, end_date)



points = pd.read_csv(setting.filename_points)
#!aqm_point = points[~points['station'].isna()]

data_knn = pd.DataFrame()
data_attention = pd.DataFrame()



# TODO recorrido por las fechas
current_date = start_date
count_error = 0
while current_date <= end_date: #-timedelta(days=700)
    current_data = data[data['datetime']==current_date]
    ndvi_date = str(current_date)[:10]
    
    if ndvi_date in data_ndvi:
        current_ndvi = data_ndvi[ndvi_date]
        prev_ndvi = current_ndvi
        #print(current_date)
    else:
        #print('No')
        current_ndvi = prev_ndvi
    
    ntl_date = str(current_date)[:10]
    if ntl_date in data_ntl:
        current_ntl = data_ntl[ntl_date]
        prev_ntl = current_ntl
    else:
        count_error += 1
        current_ntl = prev_ntl

    count = 0
    with tqdm(total=len(points)) as bar:
        bar.set_description(str(current_date)[:10])
        for i in points.index:
            current_id = points.loc[i, 'id']
            lat = points.loc[i, 'lat']
            lon = points.loc[i, 'long']
            #print(type(points.loc[i, 'left_top']))
            lat_left_top, lon_left_top = Point(points.loc[i, 'left'], 
                                            points.loc[i, 'top']).convertToCoord()
            lat_right_bottom, lon_right_bottom = Point(points.loc[i, 'right'], 
                                            points.loc[i, 'bottom']).convertToCoord()

            ndvi = current_ndvi[(current_ndvi[:, 0] <= lat_left_top) & (current_ndvi[:, 1] >= lon_left_top) &
                                (current_ndvi[:, 0] >= lat_right_bottom) & (current_ndvi[:, 1] <= lon_right_bottom)][:, 2]
            ntl = current_ntl[(current_ntl[:, 0] <= lat_left_top) & (current_ntl[:, 1] >= lon_left_top) &
                                (current_ntl[:, 0] >= lat_right_bottom) & (current_ntl[:, 1] <= lon_right_bottom)][:, 2]
            dem = dem_data[(dem_data[:, 0] <= lat_left_top) & (dem_data[:, 1] >= lon_left_top) &
                            (dem_data[:, 0] >= lat_right_bottom) & (dem_data[:, 1] <= lon_right_bottom)][:, 2]
            
            
            if len(ndvi) > 0:
                mean_ndvi = ndvi.mean()
                mean_ntl = ntl.mean()
            else:
                mean_ndvi = np.nan
                mean_ntl = np.nan
            mean_dem = dem.mean()
            #if mean_dem < 100:
            #    count += 1
            point_knn = json.loads(points.loc[i, 'knn'])
            stations = [k for k in point_knn.keys()][:setting.knn]
            dist = [k for k in point_knn.values()][:setting.knn]
            
            meo_neighbor = current_data[current_data['station']==stations[0]][setting.normalize_meo].reset_index(drop=True)
            #print(stations)

            row = {'id': current_id, 'current_date': current_date, 'lat': lat, 'lon': lon}

            for meo_feature in setting.normalize_meo:
                row.update({meo_feature: meo_neighbor.loc[0, meo_feature]})
            
            row.update({'ndvi': mean_ndvi, 'ntl': mean_ntl, 'dem': mean_dem})

            row_knn = row
            for k, station in enumerate(stations):
                key_pm25 = f'pm25_{k}'
                key_dist = f'dist_{k}'
                current_neighbors = current_data[current_data['station']==station].reset_index(drop=True)
                value_pm25 = current_neighbors.loc[0, 'PM25']
                value_dist = dist[k]
                row_knn.update({key_pm25: value_pm25})
                row_knn.update({key_dist: value_dist})
            data_knn = data_knn.append(row_knn, ignore_index=True)
            bar.update(1)
        count += 1
    current_date += timedelta(days=1)

data_knn.to_csv('data_knn.csv')

#print(count)
    #lat, lon = points.loc[i, 'lat'], points.loc[i, 'long']
    #current_point = Coord(lat, lon)
    
    #print(lat, lon)