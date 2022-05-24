import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import math 
import pickle 
import pandas as pd
import matplotlib.pyplot as plt

try: 
    DATASOURCE = sys.argv[1]
    if not DATASOURCE in ['BE', 'SP']:
        raise 'Invalid parameters: 1: BE or SP'
except Exception as e:
    raise 'Not found parameters: input DATASOURCE: SP or BE'

class setting:
    DIR_RESULST = f'../Experimentation/Results'
    knn = [3, 5, 7]
    variograms = ['linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect']

if __name__ == '__main__':
    
    results_data = pd.DataFrame(columns=['variogram', 'station_test', 'RMSE', 'MAE', 'R2'])
    #!for k in setting.knn:
    for variogram in setting.variograms:
        current_dir = F'{setting.DIR_RESULST}/{DATASOURCE}/dict_ok_{DATASOURCE}_{variogram}.p'
        try:
            with open(current_dir, 'rb') as fp:
                dict_results = pickle.load(fp)
        except Exception as e:
            raise f'Not found results for knn={k}'
        row = {}        

        for key_, v in dict_results.items():
            current_station = dict_results[key_]
            #for k_, v_ in current_station.items():
            row = {'variogram': variogram, 'station_test': key_, 'RMSE': current_station.get('RMSE'), 'MAE': current_station.get('MAE'), 
            'R2': current_station.get('R2')}
            results_data = results_data.append(row, ignore_index=True)
    #print(results_data)
    plt.rcParams.update({'font.size': 14})
    #variogram = 'linear'
    r = 0
    j = 0
    fig, ax = plt.subplots(math.ceil(len(dict_results)/3), 3, figsize=(80, 60))
    for k, v in dict_results.items():
        current_data = results_data[(results_data['station_test']==k)][['variogram', 'RMSE', 'MAE']]
        t_current_data = current_data.transpose()
        t_current_data.columns = t_current_data.iloc[0, :]
        t_current_data = t_current_data.drop('variogram')    
        print(t_current_data)
        t_current_data.plot(kind='barh', figsize=(25, 15), xlabel='Metrics', 
            ax=ax[r, j], fontsize=16).legend(loc='center left', title='Variogram')
        ax[r, j].text(0.38, 0.43, f'{k}', horizontalalignment='left', verticalalignment='bottom', transform   =ax[r, j].transAxes, fontsize=18)
        #ax[r, j].set_xlabel('Error $(\mu g m^{-3})$', fontsize=16)
        ax[r, j].set_ylabel('Metrics', fontsize=14)

        if r == ax.shape[0] - 1:
            ax[r, j].set_xlabel('Error $(\mu g m^{-3})$', fontsize=16)
            j += 1
            r = 0 
        else:
            r += 1
    
    if not os.path.isdir('Graphics'):
        os.mkdir('Graphics')
    
    plt.savefig(f'Graphics/{DATASOURCE}_RMSE_MAE_OK.png')    
    
    means = results_data.groupby('variogram').mean()
    print(means)