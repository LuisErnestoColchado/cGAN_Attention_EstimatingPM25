# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Match meteorological condition, satellite products and pollution data
# ******************************************************************************************

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import timedelta
from pyproj import Proj
import math
import pickle

from Preprocessing.pre_ntl import VNP46A1_DIR
##


settin
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


# Read Data
NDVI_DATA_DIR = '../Data/Satellite_data/MOD13A2/DATA_CROPPED'
VNP_DATA_DIR = '../Data/Satellite_data/VNP46A1/DATA_CROPPED'
DEM_DATA_DIR = '../Data/Satellite_data/DEM/DATA_CROPPED'


data_ndvi = get_data(NDVI_DATA_DIR)
#data_ndvi.keys()
dict_ = data_ndvi['2015-01-01_h26v04'] 
print(max(dict_[:, 0]), max(dict_[:, 1]))
print(min(dict_[:, 0]), min(dict_[:,1]))

dict_ = data_ndvi['2015-01-01_h26v05'] 
print(max(dict_[:, 0]), max(dict_[:, 1]))
print(min(dict_[:, 0]), min(dict_[:,1]))

data_ntl = get_data(VNP46A1_DIR)



