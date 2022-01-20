import os
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm import tqdm 
import json 
from point import Point


class setting:
    DIR_CETESB = '../Data/CETESB'
    DIR_POINTS = f'{DIR_CETESB}/points.csv'
    start_date = pd.to_datetime('01-01-2017') 
    end_date = pd.to_datetime('12-31-2019')
    NDVI_DIR = f'{DIR_CETESB}/Satellite_data/MOD13A2/DATA_CROPPED'
    VNP_DIR = f'{DIR_CETESB}/Satellite_data//VNP46A1/DATA_CROPPED'
    DEM_DIR = f'{DIR_CETESB}/Satellite_data//DEM/DATA_CROPPED'
    DIR_RESULTS = DIR_CETESB + '../Preprocessing_results'
    selected_columns = ['lat', 'lon', 'temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem', 'pm25']


def get_data(DATA_DIR):
    data = {}
    with open(DATA_DIR + '/info.txt', 'r') as f:
        nrows, ncolumns = f.readline().split('x')
    print('getting data from {}'.format(DATA_DIR))
    with tqdm(total=len(os.listdir(DATA_DIR))) as bar:
        for file in os.listdir(DATA_DIR):
            if file.endswith('.dat'):
                year_days = file.split('_')[3][1:-4]
                year = int(year_days[:4])
                days = int(year_days[4:])
                current_date = pd.to_datetime(str(year - 1) + '-12-31')
                current_date += timedelta(days=days)
                array = np.fromfile(DATA_DIR + '/' + file, dtype=float)
                array = array.reshape(int(len(array) / 3), 3)
                #print(str(current_date)[:-9])
                data.update({str(current_date)[:-9]: array})
                bar.update(1)
    return data


def match_data(points, df_pm25, df_temp, df_hum, df_press, df_wd, d_ndvi, d_ntl, array_dem):
    current_date = setting.start_date
    data = pd.DataFrame()
    count_error = 0
    while current_date <= setting.end_date:
        string_date = str(current_date)[:10]
        current_temp = df_temp[df_temp['date']==string_date]
        current_hum = df_hum[df_hum['date']==string_date]
        current_press = df_press[df_press['date']==string_date]
        current_wd = df_wd[df_wd['date']==string_date]
        
        current_pm25 = df_pm25[df_pm25['date']==string_date]
        if string_date in data_ndvi:
            current_ndvi = data_ndvi[string_date]
            prev_ndvi = current_ndvi
        else:
            current_ndvi = prev_ndvi
            
        if string_date in data_ntl:
            current_ntl = data_ntl[string_date]
            prev_ntl = current_ntl
        else:
            count_error += 1
            current_ntl = prev_ntl

        with tqdm(total=len(points)) as bar:
            bar.set_description(string_date)
            for p in points.index:
                current_station = points.loc[p, 'station']
                lat, lon = points.loc[p, 'lat'], points.loc[p, 'long']
                lat_left_top, lon_left_top = Point(points.loc[p, 'left'], 
                                                    points.loc[p, 'top']).convertToCoord()
                lat_right_bottom, lon_right_bottom = Point(points.loc[p, 'right'], 
                                                    points.loc[p, 'bottom']).convertToCoord()
                
                point_knn = json.loads(points.loc[p, 'knn'])
                stations = [k for k in point_knn.keys()]
                dist = [k for k in point_knn.values()]
                
                s = 0
                temp = hum = press = wd = None
                while s < len(stations) and (temp is None or hum is None or press is None or wd is None):
                    station = stations[s]
                    station_temp = current_temp[current_temp['stationname']==station].reset_index(drop=True)
                    station_hum = current_hum[current_hum['stationname']==station].reset_index(drop=True)
                    station_press = current_press[current_press['stationname']==station].reset_index(drop=True)
                    station_wd = current_wd[current_wd['stationname']==station].reset_index(drop=True)

                    if not station_temp.empty and temp is None:
                        temp = station_temp.loc[0, 'conc']
                    if not station_hum.empty and hum is None:
                        hum = station_hum.loc[0, 'conc']
                    if not station_press.empty and press is None:
                        #print(station_press)
                        press = station_press.loc[0, 'conc']
                    if not station_wd.empty and wd is None:
                        wd = station_wd.loc[0, 'conc']
                    s += 1

                d_ndvi = current_ndvi[(current_ndvi[:, 0] <= lat_left_top) & (current_ndvi[:, 1] >= lon_left_top) &
                                    (current_ndvi[:, 0] >= lat_right_bottom) & (current_ndvi[:, 1] <= lon_right_bottom)][:, 2]
                d_ntl = current_ntl[(current_ntl[:, 0] <= lat_left_top) & (current_ntl[:, 1] >= lon_left_top) &
                                    (current_ntl[:, 0] >= lat_right_bottom) & (current_ntl[:, 1] <= lon_right_bottom)][:, 2]
                
                d_dem = array_dem[(array_dem[:, 0] <= lat_left_top) & (array_dem[:, 1] >= lon_left_top) &
                                (array_dem[:, 0] >= lat_right_bottom) & (array_dem[:, 1] <= lon_right_bottom)][:, 2]


                if len(d_ndvi) > 0:
                    mean_ndvi = d_ndvi.mean()
                    mean_ntl = d_ntl.mean()
                else:
                    mean_ndvi = np.nan
                    mean_ntl = np.nan
                mean_dem = d_dem.mean()

                row = {'lat': lat, 'lon': lon, 'temp': temp, 'hum': hum, 'press': press, 'wd': wd, 'ndvi': mean_ndvi, 
                        'ntl': mean_ntl, 'dem': mean_dem}

                k = 0
                for station in stations:
                    current_neighbors = current_pm25[current_pm25['stationname']==station].reset_index(drop=True)
                    if not current_neighbors.empty:
                        value_pm25 = current_neighbors.loc[0, 'conc']
                        key_pm25 = f'pm25_{k}'
                        key_dist = f'dist_{k}'

                        value_dist = dist[k]
                        row.update({key_pm25: value_pm25})
                        row.update({key_dist: value_dist})
                        k += 1

                pm25_ = None
                if str(current_station) != 'nan':
                    curr_data = current_pm25[current_pm25['stationname']==current_station].reset_index(drop=True)
                    pm25_ = curr_data.loc[0, 'conc']
                row.update({'pm25': pm25_})
                data = data.append(row, ignore_index=True)
                bar.update(1)

            # get temperature and humidity  
            
        current_date += timedelta(days=1)
    return data


if __name__ == '__main__':
    pm25 = pd.read_csv(setting.DIR_CETESB+'/data_pm25.csv')
    temperature = pd.read_csv(setting.DIR_CETESB+'/data_temperature.csv')
    humidity = pd.read_csv(setting.DIR_CETESB+'/data_humidity.csv')
    pressure = pd.read_csv(setting.DIR_CETESB+'/data_pressure.csv')
    wind_direction = pd.read_csv(setting.DIR_CETESB+'/data_wind_direction.csv')
    points = pd.read_csv(setting.DIR_POINTS)

    print(f'Loading data from {setting.DEM_DIR} ...')
    dem_data = np.zeros(shape=(0, 3))
    for file in os.listdir(setting.DEM_DIR):
        if file.endswith('.dat'):
            dem_array = np.fromfile(f'{setting.DEM_DIR}/{file}', dtype=float)
            dem_array = dem_array.reshape(int(len(dem_array) / 3), 3)
            dem_data = np.concatenate([dem_data, dem_array], axis=0)
    print('Loaded!')
    
    data_ndvi = get_data(setting.NDVI_DIR)

    data_ntl = get_data(setting.VNP_DIR)

    data_train = match_data(points, pm25, temperature, humidity, pressure, wind_direction, data_ndvi, data_ntl, dem_array)
    