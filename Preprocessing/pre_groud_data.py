import os
from posixpath import split
import pandas as pd 
import numpy as np 
import glob 
from tqdm import tqdm
from point import Point
from cord import Coord

# Setting 
class setting:
    directory_data = '../Data/Ground_data/pollution_meteorological'
    num_files = len([name for name in os.listdir(directory_data) if os.path.isfile(name)])
    grid1km_qgis = '../Data/Ground_data/grids_1km_qgis.csv'

def get_bb_region():
    data = pd.read_csv(setting.grid1km_qgis)
    print(data.columns)
    left, top = data.loc[0, 'left'], data.loc[0, 'top']
    right, bottom = data.iloc[-1, data.columns.get_loc('right')], data.iloc[-1, data.columns.get_loc('bottom')]
    lt_point = Point(left, top)
    lon, lat = lt_point.convertToCoord()
    rb_point = Point(right, bottom)
    lonrb, latrb = rb_point.convertToCoord()
    return Coord(long=lon, lat=lat), Coord(long=lonrb, lat=latrb)
    

# Function read the data for a directory (format: csv)
def read_data():
    dict_data = {}
    print(setting.num_files)
    with tqdm(total=len(glob.glob(os.path.join(setting.directory_data, '*.csv')))) as bar:
        for filename in glob.glob(os.path.join(setting.directory_data, '*.csv')):
            split_file = filename.split('_')
            station = split_file[4]
            data = pd.read_csv(filename)
            dict_data.update({station: data})
            bar.set_description(f'Reading data from {station} station')
            bar.update(1)
    return dict_data


def get_dates(dict_data):
    pass
    #for k, df in dict_data.items():
        
if __name__ == '__main__':
    #dict_data = read_data()['Dongsi']
    min, max = get_bb_region()
    print(f'{min.lat}, {min.long} , {max.lat}, {max.long}')
    #print([k for k, v in dict_data.items()])

