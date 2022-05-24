import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import math 
import pickle 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

try: 
    DATASOURCE = sys.argv[1]
    if not DATASOURCE in ['BE', 'SP']:
        raise 'Invalid parameters: 1: BE or SP'
except Exception as e:
    raise 'Not found parameters: input DATASOURCE: SP or BE'

class setting:
    DIR_DATA = ""
    DIR_RESULST = f"../Experimentation/Results"
    knn = [3, 5, 7]#, 10]

if __name__ == '__main__':
    
    results_data = pd.DataFrame(columns=['knn', 'station_test', 'RMSE', 'MAE', 'R2'])
    for k in setting.knn:
        current_dir = F'{setting.DIR_RESULST}/{DATASOURCE}/dict_idw_{DATASOURCE}_{k}.p'
        try:
            with open(current_dir, 'rb') as fp:
                dict_results = pickle.load(fp)
        except Exception as e:
            raise f'Not found results for knn={k}'
        row = {}        

        for key_, v in dict_results.items():    
            current_station = dict_results[key_]
            #for k_, v_ in current_station.items():
            row = {'knn': k, 'station_test': key_, 'RMSE': current_station.get('RMSE'), 'MAE': current_station.get('MAE'), 
            'R2': current_station.get('R2'), 'output': current_station.get('output')}
            results_data = results_data.append(row, ignore_index=True)
    plt.rcParams.update({'font.size': 16})
    r = 0
    j = 0

    print(type(results_data.loc[0, 'output']))
    fig, ax = plt.subplots(math.ceil(len(dict_results)/2), 2, figsize=(90, 45))

    for k, v in dict_results.items():
        current_data = results_data[results_data['station_test']==k][['knn', 'MAE', 'RMSE']]#, 'MAE']]
        t_current_data = current_data.transpose()
        #!current_data.columns = current_data.iloc[0, :]
        
        
        #!current_data = current_data.transpose()
        #!t_current_data = current_data.transpose().reset_index(drop=True)
        
        #!st_current_data.index = t_current_data.knn.values
        #!print('1', t_current_data)
        
        t_current_data.columns = t_current_data.iloc[0, :]
        
        t_current_data = t_current_data.drop('knn')
        print(t_current_data)
        t_current_data.plot(kind='barh', figsize=(25, 15), xlabel='Error $(\mu g m^{-3})$ \n',
                            ax=ax[r, j], fontsize=16).legend(loc='center left', title='KNN')
        #!ax[r, j].legend(loc='center left')
        ax[r, j].text(0.38, 0.41, f'{k}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=18)
        ax[r, j].set_ylabel('Metrics', fontsize=14)
        #!ax[r, j].set_xlabel('Error $(\mu g m^{-3})$', fontsize=16)
        #!ax[r, j].set_xticklabels(['3', '5', '7'])
        #!ax[r, j].legend(loc='center left')

        if r == ax.shape[0] - 1:
            ax[r, j].set_xlabel('Error $(\mu g m^{-3})$', fontsize=16)
            #!ax[r, j].set_xticks(['MAE', 'RMSE'])   
            j += 1
            r = 0 
        else:
            #!ax[r, j].set_xticks([])
            r += 1

    if not os.path.isdir('Graphics'):
        os.mkdir('Graphics')

    plt.savefig(f'Graphics/{DATASOURCE}_RMSE_MAE_IDW.png')    
    """
    fig, ax = plt.subplots(math.ceil(len(dict_results)/2), 2, figsize=(90, 45))

    for k, v in dict_results.items():
        current_ouput = results_data[results_data['station_test']==k][['knn', 'output']]
        t_current_data = current_ouput.transpose()
        line_1 = np.random.randint(low = 0, high = 50, size = 50)
        line_2 = np.random.randint(low = -15, high = 100, size = 50)
        t_current_data.columns = t_current_data.iloc[0, :]
        t_current_data = t_current_data.drop('knn')
        ax[r, j].plot(t_current_data[3].iloc[0][:5], label='3', color='b')
        ax[r, j].plot(t_current_data[5].iloc[0][:5], label='5', color='g')
        ax[r, j].plot(t_current_data[7].iloc[0][:5], label='7', color='r')
        ax[r, j].legend(['3', '5', '7'])
            #!].plot(kind='barh', figsize=(25, 15), xlabel='Metric', 
            #!    ax=ax[r, j], fontsize=16)
        
        r += 1
        if r == ax.shape[0]:
            j += 1
            r = 0 
        #t_current_data.plot()
    """
    #!plt.show()

    means = results_data.groupby('knn').mean()
    print(means)