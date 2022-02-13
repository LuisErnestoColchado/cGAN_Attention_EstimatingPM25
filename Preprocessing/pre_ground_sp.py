# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Preprocess meteorological condition and pollution data from Sao Paulo CETESB
# ******************************************************************************************
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


from coord import Coord
from point import Point
from bounding_box import Bounding_box
import pandas as pd
from datetime import timedelta
import numpy as np
import json 
from pandas.core.frame import DataFrame
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class setting:
    DIR_CETESB = '../Data/CETESB'
    grid2km_qgis = '../Data/CETESB/grid2km_qgis.csv'
    stations = '../Data/CETESB/stations.csv'
    start_date = pd.to_datetime('2017-01-01 00:00:00')
    end_date = pd.to_datetime('2019-12-31 23:00:00')
    selected_columns  = ['stationname', 'date', 'conc']
    limit_nan = 0.05 


def read_data():
    wind_direction = pd.read_csv(setting.DIR_CETESB+'/DV.csv')
    wind_speed = pd.read_csv(setting.DIR_CETESB+'/VV.csv')
    pressure = pd.read_csv(setting.DIR_CETESB+'/PRESS.csv')
    temperature = pd.read_csv(setting.DIR_CETESB+'/TEMP.csv')
    humidity = pd.read_csv(setting.DIR_CETESB+'/UR.csv')
    pm25 = pd.read_csv(setting.DIR_CETESB+'/MP25.csv')
    stations = pd.read_csv(setting.DIR_CETESB+'/stations.csv')
    return wind_direction, wind_speed, pressure, temperature, humidity, pm25, stations


def create_datetime(df: DataFrame):
    df['datetime'] = pd.to_datetime(df.date) + (df.hour.astype('timedelta64[h]') - timedelta(hours=1))
    df['datetime'] = pd.to_datetime(df.date) + (df.hour.astype('timedelta64[h]') - timedelta(hours=1))    
    df = df[(df['datetime'] >= setting.start_date) & (df['datetime'] <= setting.end_date)].reset_index(drop=True)
    return df
    

def clean_data(df: DataFrame):
    q_low = df.conc.quantile(0.25)
    q_high = df.conc.quantile(0.75)
    IQR = q_high - q_low
    low = q_low - (1.5 * IQR)
    high = q_high + (1.5 * IQR)
    mask = ((df['conc'] < low) | (df['conc'] > high))

    df.loc[mask, 'conc'] = np.nan 

    df = df.assign(conc=df['conc'].interpolate(method='linear'))
    sum_null = df[['conc']].isna().sum()
    percentage_null = (sum_null['conc'] * 100) / len(df)
    print('Data ' + str(df['parameter'][0]) + ':\n Nun Values ' + \
           str(sum_null[0]) + '\n Percentage nun values ' + str(percentage_null) + '%')
    print('Fill nan values ...')
    return df


# Get bounding box region
def get_bb_region() -> Coord:
    data = pd.read_csv(setting.grid2km_qgis)
    left, top = data.loc[0, 'left'], data.loc[0, 'top']
    right, bottom = data.iloc[-1, data.columns.get_loc('right')], data.iloc[-1, data.columns.get_loc('bottom')]
    lt_point = Point(left, top)
    lon, lat = lt_point.convertToCoord()
    rb_point = Point(right, bottom)
    lonrb, latrb = rb_point.convertToCoord()
    return Coord(lat=lat, long=lon), Coord(lat=latrb, long=lonrb)


# Get neighbors for points with and without aqm stations 
def grid_stations(labeled_stations) -> DataFrame:
    dict_neighbors = {}
    grid_data = pd.read_csv(setting.grid2km_qgis)
    coord_aq_stations = pd.read_csv(setting.stations)
    grid_station = pd.DataFrame(columns=['id', 'lat', 'long', 'is_aqm', 'station'])
    # get center of points without station      
    with tqdm(total=len(grid_data)) as pbar:
        for i in grid_data.index:
            grid_lt = Point(grid_data.loc[i, 'left'], grid_data.loc[i, 'top'])
            grid_rb = Point(grid_data.loc[i, 'right'], grid_data.loc[i, 'bottom'])
            bb_grid = Bounding_box(i,1,grid_lt, grid_rb)
            x_center, y_center = bb_grid.getCenter()
            lat_center, long_center = Point(x_center, y_center).convertToCoord()
            current_coord = Coord(lat_center, long_center)
            lt_lat, lt_lon = grid_lt.convertToCoord()
            rb_lat, rb_lon = grid_rb.convertToCoord()
            lt_lat, lt_lon = round(lt_lat, 4), round(lt_lon, 4)
            rb_lat, rb_lon = round(rb_lat, 4), round(rb_lon, 4)
            
            flag = False 
            station_ = None
            for station, long, lat in zip(coord_aq_stations['Label'], coord_aq_stations['Longitude'], coord_aq_stations['Latitude']):
                aqm_coord = Coord(lat, long)
                dist = current_coord.haversineDistance(aqm_coord)
                dict_neighbors.update({station: dist})
                #!x, y = aqm_coord.convertToPoint()
                #point_station = Point(x, y)
                #!if point_station.isWithinBB(bb_grid):
                if (lt_lat >= aqm_coord.lat >= rb_lat) and (lt_lon <= aqm_coord.long <= rb_lon):
                    flag = True
                    station_ = station
            dict_sorted = dict(sorted(dict_neighbors.items(), key=lambda item: item[1]))
            if station_ is not None and station_ in labeled_stations: 
                #print(station_)
                del dict_sorted[station_]
            else: 
                station_ = None
            row = {'id': 'point_'+str(i), 'lat': lat_center, 'long': long_center, 'left': grid_lt.x, 'top': grid_lt.y,
                   'right': grid_rb.x, 'bottom': grid_rb.y, 'is_aqm': flag, 'station': station_, 'knn': json.dumps(dict_sorted)}
            grid_station = grid_station.append(row, ignore_index=True)
            pbar.set_description('point_'+str(i))
            pbar.update(1)
    return grid_station


if __name__ == '__main__':
    wind_direction, wind_speed, pressure, temperature, humidity, pm25, stations = read_data()
    wind_direction = create_datetime(wind_direction)
    pressure = create_datetime(pressure)
    temperature = create_datetime(temperature)
    humidity = create_datetime(humidity)
    pm25 = create_datetime(pm25)

    wind_direction = clean_data(wind_direction)
    pressure = clean_data(pressure)
    temperature = clean_data(temperature)
    humidity = clean_data(humidity)
    pm25 = clean_data(pm25)

    pm25 = pm25[pm25['stationname'] != 'Mooca'].reset_index(drop=True)
    
    pm25_daily = pm25.groupby(['stationname', 'date'], as_index=False).mean().drop('hour', axis=1)


    temperature_daily = temperature.groupby(['stationname', 'date'], as_index=False).mean().drop('hour', axis=1)
    humidity_daily = humidity.groupby(['stationname', 'date'], as_index=False).mean().drop('hour', axis=1)
    pressure_daily = pressure.groupby(['stationname', 'date'], as_index=False).mean().drop('hour', axis=1)
    pressure_daily = pressure_daily[pressure_daily['stationname'] != 'Interlagos'].reset_index(drop=True)
    wd_daily = wind_direction.groupby(['stationname', 'date'], as_index=False).mean().drop('hour', axis=1)

    pm25 = pm25_daily[setting.selected_columns]
    temperature = temperature_daily[setting.selected_columns]
    wind_direction = wd_daily[setting.selected_columns]
    humidity = humidity_daily[setting.selected_columns] 
    pressure = pressure_daily[setting.selected_columns]

    min, max = get_bb_region()
    print(f'{min.lat}, {min.long} , {max.lat}, {max.long}')

    labeled_stations = pm25_daily['stationname'].unique().tolist()
    label_stations = stations[stations['Label'].isin(labeled_stations)]
    label_stations.to_csv('AQM_stations.csv', index=False)
    
    points = grid_stations(labeled_stations)
    
    pm25.to_csv(setting.DIR_CETESB+'/data_pm25.csv', index=False)
    temperature.to_csv(setting.DIR_CETESB+'/data_temperature.csv', index=False)
    humidity.to_csv(setting.DIR_CETESB+'/data_humidity.csv', index=False)
    pressure.to_csv(setting.DIR_CETESB+'/data_pressure.csv', index=False)
    wind_direction.to_csv(setting.DIR_CETESB+'/data_wind_direction.csv', index=False)
    points.to_csv(setting.DIR_CETESB+'/points.csv')
    
##

