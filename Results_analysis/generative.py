import os 
import math
import sys
from anyio import current_default_worker_thread_limiter
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

try: 
    DATASOURCE = sys.argv[1]
    if not DATASOURCE in ['BE', 'SP']:
        raise 'Invalid parameters: 1: BE or SP'
except Exception as e:
    raise 'Not found parameters: input DATASOURCE: SP or BE'

class setting:
    DIR_RESULST = f'../Experimentation/Results'
    if DATASOURCE == 'BE':
        stations = ['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
                   'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan'] 
        knn = [3, 5, 7]
        better_k = 3
    else:
        stations = ['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios', 'Cid.Universitária-USP-Ipen',
                 'Pinheiros', 'Parelheiros', 'Ibirapuera', 'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
        knn = [3, 5, 7, 10]
        better_k = 10

    comb_parameters = ((1, 0), (0, 1), (0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (1, 2), (2, 1)) # (1, 0) is only cGAN , (0.6, 0.4) 
    colors = ['r', 'b', 'g', 'black', 'c', 'm', 'y', 'peru']
    comb_parameters_cgansl = ((0, 1), (0.1, 0.9), (0.2, 0.8), (0.4, 0.6), (1, 2), (2, 1))

if __name__ == '__main__':
    """
    r = 0
    j = 0
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(math.ceil(len(setting.stations)/2), 2, figsize=(60, 30)) #/4
    
    results_data = pd.DataFrame(columns=['knn', 'station_test', 'RMSE', 'MAE', 'R2'])
    mean_r2 = 0


    x = np.arange(1, 800+1, 1)
    for station in setting.stations[6:]: #[6:]
        idx_max = 0
        ax[r, j].text(0.8, 0.8, f'{station}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)
        for c, k in enumerate(setting.knn):
            current_dir = f"{setting.DIR_RESULST}/{DATASOURCE}/cgansl_advloss{setting.comb_parameters[4][0]}_spl{setting.comb_parameters[4][1]}_knn{k}/station_{station}"
            
            r2 = np.load(f"{current_dir}/r2_test.npy")
            mae = np.load(f"{current_dir}/mae_test.npy")
            rmse = np.load(f"{current_dir}/rmse_test.npy")

            r2_max = np.max(r2)
            index_max = np.argmax(r2)
            mae_max = mae[index_max]
            rmse_max = rmse[index_max]
            #!ax[r, j].text(0.60, 0, f'{station}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)
            ax[r, j].plot(x, rmse, setting.colors[c], label=f"RMSE with kNN={k}", linewidth= 6)
            #ax[r, j].set_ylim(-1, 1)
            row = {'knn': k, 'station_test': station, 'RMSE': rmse_max, 'MAE': mae_max, 'R2': r2_max}
            results_data = results_data.append(row, ignore_index=True)
            mean_r2 += r2_max
            if idx_max < index_max:
                idx_max = index_max
        
        results_station = results_data[results_data['station_test']==station]
    
        #!for k_ in setting.knn:
        #!    results_k = results_data[results_data['knn'] == k_]
            
        ax[r, j].legend(loc='lower left', prop={'size': 40})
        ax[r, j].set_ylabel('RMSE error ($(\mu g m^{-3}$)', fontsize= 40)

        if r == ax.shape[0] - 1:
            ax[r, j].set_xlabel('Learning epoch', fontsize=40)
            j += 1
            r = 0 
        else:
            r += 1
    
    if not os.path.isdir('Graphics'):
        os.mkdir('Graphics')
    
    plt.savefig(f'Graphics/{DATASOURCE}_RMSE_cgan_k_2.png')  
    print(results_data.groupby('knn').mean())
    """

    data_params = pd.DataFrame(columns=['params', 'RMSE'])
    for comb in setting.comb_parameters:   
        print(comb)
        sum_r2_max = 0
        sum_mae_max = 0
        sum_rmse_max = 0
        for station in setting.stations:
            current_dir = f"{setting.DIR_RESULST}/{DATASOURCE}/cgansl_advloss{comb[0]}_spl{comb[1]}_knn{setting.better_k}/station_{station}"
            
            r2 = np.load(f"{current_dir}/r2_test.npy")
            mae = np.load(f"{current_dir}/mae_test.npy")
            rmse = np.load(f"{current_dir}/rmse_test.npy")
            
            #!print(station, np.max(r2))
            r2_max = np.max(r2)
            index_max = np.argmax(r2)
            mae_max = mae[index_max]
            rmse_max = rmse[index_max]
            sum_r2_max += r2_max
            sum_mae_max += mae_max
            sum_rmse_max += rmse_max
            row = {'params': f"{comb[0]}, {comb[1]}", 'station': station, 'RMSE': rmse_max, 'MAE': mae_max, 'r2': r2_max}
            data_params = data_params.append(row, ignore_index=True)
        print(comb, round(sum_rmse_max/len(setting.stations), 4), round(sum_mae_max/len(setting.stations), 4), round(sum_r2_max/len(setting.stations), 4))

    r = 0
    j = 0
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(math.ceil(len(setting.stations)/4), 2, figsize=(60, 30)) #/4
    x = np.arange(1, 101, 1)
    for station in setting.stations[6:]: #[6:]
        idx_max = 0
        ax[r, j].text(0.8, 0.8, f'{station}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=38)
        for c, comb in enumerate(setting.comb_parameters_cgansl):
            
            current_dir = f"{setting.DIR_RESULST}/{DATASOURCE}/cgansl_advloss{comb[0]}_spl{comb[1]}_knn3/station_{station}"
            
            r2 = np.load(f"{current_dir}/r2_test.npy")
            mae = np.load(f"{current_dir}/mae_test.npy")
            rmse = np.load(f"{current_dir}/rmse_test.npy")

            r2_max = np.max(r2)
            index_max = np.argmax(r2)
            mae_max = mae[index_max]
            rmse_max = rmse[index_max]
            #!ax[r, j].text(0.60, 0, f'{station}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)
        
            ax[r, j].plot(x[:72], rmse[:72], setting.colors[c], label=f"RMSE with Comb: λ={comb[0]}, θ={comb[1]}", linewidth= 2)

        ax[r, j].legend(loc='lower left', prop={'size': 38})
        ax[r, j].set_ylabel('RMSE error ($(\mu g m^{-3}$)', fontsize= 38)

        if r == ax.shape[0] - 1:
            ax[r, j].set_xlabel('Learning epoch', fontsize=38)
            j += 1
            r = 0 
        else:
            r += 1     
    
    plt.savefig(f'Graphics/{DATASOURCE}_RMSE_cgansl_2.png')  

