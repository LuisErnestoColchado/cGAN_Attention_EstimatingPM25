import os, sys
from shutil import ExecError

#from soupsieve import match
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pickle
import pandas as pd
import Models.IDW as IDW
from Models.commom import *
from tqdm import tqdm 


try:
    DATASOURCE = sys.argv[1]
    knn = int(sys.argv[2])
except Exception as e:
    raise 'Input SP: Sao paulo Data or BE: Beijing Data and knn (3, 5, 7) as arguments'


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
    idw_cols = ['station', 'current_date', 'lat', 'lon', 'PM25']
    p = 1


if __name__ == '__main__':
    data_idw = pd.read_csv(setting.DIR_DATA)
    data_idw = data_idw[~data_idw.PM25.isna()].reset_index(drop=True)[setting.idw_cols]
    print(f'Testing IDW for DATASOURCE {DATASOURCE}')
    dict_results = {}
    with tqdm(total=len(setting.station_test)) as bar:
        for station_ in setting.station_test:
            dict_result = {}
            train, test = split_data(data_idw, station=station_)

            data_loader = []
            for date in train.current_date.unique().tolist():
                current_data = train[train['current_date']==date]
                current_data = current_data.values
                data_loader.append(current_data)
            
            test = test.values

            y = test[:, -1].reshape(len(test), 1)
            Z = np.zeros((len(test), 1))
            for i, data_train in enumerate(data_loader):
                idw_tree = IDW.tree(data_train[:, 2:4], data_train[:, -1])
                x_test = test[i, 2:4].reshape(1, 2)
                z = idw_tree(x_test, k=knn, p=setting.p)
                Z[i] = z
            
            rmse = np.sqrt(mean_squared_error(y, Z))
            mae = mean_absolute_error(y, Z)
            r2 = r2_score(y, Z)
            dict_result = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'output': y}
            dict_results.update({f'{station_}': dict_result})
            
            with open(f'Results/{DATASOURCE}/dict_idw_{DATASOURCE}_{knn}.p', 'wb') as fd:
                pickle.dump(dict_results, fd, protocol=pickle.HIGHEST_PROTOCOL)

            msg = f'{station_}: RMSE: {rmse}, MAE: {mae}, R2: {r2}'
            bar.set_description(msg)
            bar.update(1)

