# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Match meteorological condition, satellite products and pollution data
# ******************************************************************************************

from numpy.lib.function_base import kaiser
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import timedelta
from coord import Coord
from pyproj import Proj
import json 
import math
import pickle

##


class setting:
    NDVI_DIR = '../Data/Satellite_data/MOD13A2/DATA_CROPPED'
    VNP_DIR = '../Data/Satellite_data/VNP46A1/DATA_CROPPED'
    DEM_DIR = '../Data/Satellite_data/DEM/DATA_CROPPED'
    filename_data = '../Data/Ground_data/preprocessed_data.csv'
    filename_points = '../Data/Ground_data/points.csv'
    knn = 7

##


def get_data(DATA_DIR):
    data = {}
    #!with open(DATA_DIR + '/info.txt', 'r') as f:
    #!    nrows, ncolumns = f.readline().split('x')
    print('getting data from {}'.format(DATA_DIR))
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
    return data#!, int(nrows), int(ncolumns)


# Read Satellite data preprocessed 



data_ndvi = get_data(setting.NDVI_DIR)
data_ntl = get_data(setting.VNP_DIR)

data = pd.read_csv(setting.filename_data)
start_date = min(data['datetime'])
end_date = max(data['datetime'])

points = pd.read_csv(setting.filename_points)
#!aqm_point = points[~points['station'].isna()]

data_train = pd.DataFrame()

# TODO recorrido por las fechas
current_date = start_date
current_data = data[data['datetime']==current_date]
for i in points.index:
    current_id = points.loc[i, 'id']
    lat = points.loc[i, 'lat']
    lon = points.loc[i, 'long']

    point_knn = json.loads(points.loc[0, 'knn'])
    stations = [k for k in point_knn.keys()][:setting.knn]
    dist = [k for k in point_knn.values()][:setting.knn]

    meo_neighbor = current_data[current_data['station']==stations[0]] 
    pm25_neighbors = current_data[current_data['station'].isin(stations)].loc[:, 'PM25']

    row = {'id': current_id, 'lat': lat, 'lon': lon}
    for k in range(setting.knn):
        key_pm25 = f'pm25_{k}'
        key_dist = f'dist_{k}'
        value_pm25 = pm25_neighbors.iloc[k]
        value_dist = dist[k]
        row.update({key_pm25: value_pm25})
        row.update({key_dist: value_dist})
    
    data_train = data_train.append(row, ignore_index=True)


    #lat, lon = points.loc[i, 'lat'], points.loc[i, 'long']
    #current_point = Coord(lat, lon)
    
    #print(lat, lon)


