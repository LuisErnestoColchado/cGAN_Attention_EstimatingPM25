from datetime import datetime, timedelta
import os
from posixpath import split
from typing import Coroutine
import pandas as pd 
import numpy as np 
import glob
from pandas.core.frame import DataFrame 
from tqdm import tqdm
from point import Point
from cord import Coord
from bounding_box import Bounding_box


# Setting 
class setting:
    directory_data = '../Data/Ground_data/pollution_meteorological'
    num_files = len([name for name in os.listdir(directory_data) if os.path.isfile(name)])
    grid1km_qgis = '../Data/Ground_data/grids_1km_qgis.csv'
    aq_stations = '../Data/Ground_data/stations_graph.csv'
    dict_cardinal = {"N": 360, "NbE": 11.25, "NNE": 22.5, "NEbN": 33.75, "NE": 45, "NEbE": 56.25,
                    "ENE": 67.5, "EbN": 78.75, "E": 90, "EbS": 101.25, "ESE": 112.5, "SEbE": 123.75,
                    "SE": 135, "SEbS": 146.25, "SSE": 157.5, "SbE": 168.75, "S": 180, "SbW": 191.25,
                    "SSW": 202.5, "SWbS": 213.75, "SW": 225, "SWbW": 236.25, "WSW": 247.5, "WbS": 258.75,
                    "W": 270, "WbN": 281.25, "WNW": 292.5, "NWbW": 303.75, "NW": 315, "NWbN": 326.25,
                    "NNW": 337.5, "NbW": 348.75, 'nan': np.nan}
    selected_columns = ['station', 'datetime', 'TEMP', 'PRES', 'DEWP', 'wd', 'WSPM', 'PM25']
    limit_nan = 0.05 # 5%
    start_date = pd.to_datetime('2015-01-01')
    end_date = pd.to_datetime('2016-12-31')
    knn = 7


# Get bounding box region
def get_bb_region() -> Coord:
    data = pd.read_csv(setting.grid1km_qgis)
    print(data.columns)
    left, top = data.loc[0, 'left'], data.loc[0, 'top']
    right, bottom = data.iloc[-1, data.columns.get_loc('right')], data.iloc[-1, data.columns.get_loc('bottom')]
    lt_point = Point(left, top)
    lon, lat = lt_point.convertToCoord()
    rb_point = Point(right, bottom)
    lonrb, latrb = rb_point.convertToCoord()
    return Coord(long=lon, lat=lat), Coord(long=lonrb, lat=latrb)


# Get 
def grid_stations() -> DataFrame:
    dict_neighbors = {}
    grid_data = pd.read_csv(setting.grid1km_qgis)
    coord_aq_stations = pd.read_csv(setting.aq_stations)
    grid_station = pd.DataFrame(columns=['id', 'lat', 'long', 'is_aqm', 'station'])

    # get center of points without station
    with tqdm(total=len(grid_data)) as pbar:
        for i in grid_data.index:
            grid_lt = Point(grid_data.loc[i, 'left'], grid_data.loc[i, 'top'])
            grid_rb = Point(grid_data.loc[i, 'right'], grid_data.loc[i, 'bottom'])
            bb_grid = Bounding_box(i,1,grid_lt, grid_rb)
            x_center, y_center = bb_grid.getCenter()
            long_center, lat_center = Point(x_center, y_center).convertToCoord()
            current_coord = Coord(long_center, lat_center)
            flag = False 
            station_ = None
            for station, long, lat in zip(coord_aq_stations['Label'], coord_aq_stations['Longitude'], coord_aq_stations['Latitude']):
                aqm_coord = Coord(long, lat)
                dist = current_coord.haversineDistance(aqm_coord)
                dict_neighbors.update({station: dist})
                x, y = aqm_coord.convertToPoint()
                point_station = Point(x, y)
                if point_station.isWithinBB(bb_grid):
                    flag = True
                    station_ = station
            dict_sorted = dict(sorted(dict_neighbors.items(), key=lambda item: item[1]))
            if station_ is not None: 
                print(station_)
                del dict_sorted[station_]
            row = {'id': 'point_'+str(i), 'lat': lat_center, 'long': long_center, 'is_aqm': flag, 
                    'station': station_, 'knn': dict_sorted}
            grid_station = grid_station.append(row, ignore_index=True)
            pbar.update(1)
    return grid_station


# Function read the data for a directory (format: csv)
def read_data() -> dict:
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


def get_date(data_frame: DataFrame) -> DataFrame: 
    columns_ = ['year', 'month', 'day', 'hour']
    data_frame['datetime'] = pd.to_datetime(data_frame[columns_])
    return data_frame


def check_complete_series(data_frame: DataFrame) -> bool:
    print(len(data_frame))
    count = 0
    start_d = setting.start_date 
    flag = True
    while start_d <= setting.end_date:
        #print(start_d, data_frame.loc[count, 'datetime'])
        if count < len(data_frame):
            if start_d != data_frame.loc[count, 'datetime']:
            #print("Missing date: ", start_d)
                flag = False
        else:
            flag = False
            
        start_d += timedelta(hours=1)
        count += 1
    return flag


def preprocessing(dict_data: dict) -> DataFrame:
    
    with tqdm(total=len(dict_data)) as bar:
        #!for k, v in dict_data.items():
            #print(k, v.columns)
        k, v = dict_data.popitem()

        v['PM25'] = v['PM2.5']
        v['station'] = k

        df = get_date(v)
        df = df[setting.selected_columns]
        df = df[(df['datetime'] >= setting.start_date) & (df['datetime'] <= setting.end_date)].reset_index(drop=True)
#list_ang = [setting.dict_cardinal[str(x)] for x in df['wd']]
#df['wd'] = list_ang
        df = df.replace({'wd': setting.dict_cardinal})
#print(df['wd'])
#print(k)
        
        print(k)
        for column in df.columns:
            if column not in ['station', 'datetime']:
                nan_per = len(df[df[column].isna()])/len(df)
                if nan_per <= setting.limit_nan:
                    df[column] = df[column].interpolate(method='linear')
                #print(df)
            #print(len(df[df[column].isna()])/len(df))
            #print(column, nan_per)

            #if column == 'PM25':
            #    mean_nan_pm25 += nan_per
        print(check_complete_series(df))
        dict_data.update({k: df})
    #print(dict_data['Aotizhongxin']['wd'])
    #!k, v = dict_data.popitem()
    #!print(v['wd'])
    #print(mean_nan_pm25/len(dict_data))
    return dict_data

if __name__ == '__main__':
    dict_data = read_data()
    min, max = get_bb_region()
    print(f'{min.lat}, {min.long} , {max.lat}, {max.long}')
    grid_aqm = grid_stations()
    for i in grid_aqm.index:
        if len(grid_aqm.loc[i, 'knn']) < 12:
            print(i)
    
    #!dict_ = preprocessing(dict_data)['Wanshouxigong']
    #data = dict_data.get('Huairou')
    #!print(dict_.columns)
    #print([k for k, v in dict_data.items()])

