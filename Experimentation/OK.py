
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np  
import pandas as pd
from tqdm import tqdm
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Models.commom import *
import pickle

variograms = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']

try:
    DATASOURCE = sys.argv[1]
    #knn = int(sys.argv[2])
    variogram_model = sys.argv[2]
except Exception as e:
    raise 'Input SP: Sao paulo Data or BE: Beijing Data and Variogram (linear, power, gaussian, spherical, exponential, hole-effect) as arguments'


class setting:
    if DATASOURCE == 'BE':
        DIR_DATA = '../Preprocessing/Results/data_train.csv'
        station_test = ['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
                   'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan'] 
    
    elif DATASOURCE == 'SP':
        DIR_DATA = '../Preprocessing/Results/data_train_sp.csv'
        station_test = ['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
        'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
        'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
    else:
        raise 'Invalid value. Input SP: Sao paulo Data or BE: Beijing Data as argument'

    if variogram_model in variograms:
        variogram = variogram_model
    else:
        raise 'Invalid variogram'
    selected_features_ok = ['station', 'current_date', 'lat', 'lon', 'PM25']


if __name__ == '__main__':
    data_ok = pd.read_csv(setting.DIR_DATA)
    data_labeled = data_ok[~data_ok['PM25'].isna()].reset_index()
    data_labeled = data_labeled[setting.selected_features_ok]
    dict_results = {}
    with tqdm(total=len(setting.station_test)) as bar:
        for t_station in setting.station_test:
            train, test = split_data(data_labeled, station=t_station)

            data_loader = []
            for date in train.current_date.unique().tolist():
                current_data = train[train['current_date']==date]
                current_data = current_data.values
                data_loader.append(current_data)
            
            test = test.values

            y = test[:, -1].reshape(len(test), 1)
            Z = np.zeros((len(y), 1))
            
            for i, data_train in enumerate(data_loader):
                #x_train = data_train[i*10:(i+1)*10, :]
                #current_data = train_data[train_data["utc_time"] == test_data[i,0]].values
                #if len(current_data) >= 0:
                # Create the ordinary kriging object. Required inputs are the X-coordinates of
                # the data points, the Y-coordinates of the data points, and the Z-values of the
                # data points. If no variogram model is specified, defaults to a linear variogram
                # model. If no variogram model parameters are specified, then the code automatically
                # calculates the parameters by fitting the variogram model to the binned
                # experimental semivariogram. The verbose kwarg controls code talk-back, and
                # the enable_plotting kwarg controls the display of the semivariogram.
                OK = OrdinaryKriging(data_train[:, 2], data_train[:, 3], data_train[:, 4], variogram_model=setting.variogram,
                                    verbose=False, enable_plotting=False)

                # Creates the kriged grid and the variance grid. Allows for kriging on a rectangular
                # grid of points, on a masked rectangular grid of points, or with arbitrary points.
                # (See OrdinaryKriging.__doc__ for more information.)
                z,_ = OK.execute('grid', test[i, 2], test[i, 3])

                Z[i] = z
            
            # Writes the kriged grid to an ASCII grid file.
            #kt.write_asc_grid(test_data[:, 0], test_data[:, 1], z, filename="output.asc")
            
            rmse_error = round(np.sqrt(mean_squared_error(y, Z)), 4)
            mae = round(mean_absolute_error(y, Z), 4)
            r2 = round(r2_score(y, Z), 4)
            dict_result = {'RMSE': rmse_error, 'MAE': mae, 'R2': r2}
            dict_results.update({f'{t_station}': dict_result})
            
            with open(f'Results/{DATASOURCE}/dict_ok_{DATASOURCE}_{setting.variogram}.p', 'wb') as fd:
                pickle.dump(dict_results, fd, protocol=pickle.HIGHEST_PROTOCOL)
            msg = f'{t_station}: RMSE: {rmse_error}, MAE: {mae}, R2: {r2}'
            bar.set_description(msg)
            bar.update(1)
