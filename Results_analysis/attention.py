
import os, sys
import pandas as pd 
import math
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
    else:
        stations = ['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
                  'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
             'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
    kernel = ['affine','cosine', 'gaussian', 'inner-product']#, 'gaussian']#'cosine', 'gaussian', 'inner-product']
    #['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
    #    'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
    #    'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']

    #['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
    #               'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan'] 

        
if __name__ == '__main__':
    
    results_data = pd.DataFrame(columns=['kernel', 'station_test', 'RMSE', 'MAE', 'R2'])
    avg_r2 = 0
    plt.rcParams.update({'font.size': 40})
    fig, ax = plt.subplots(math.ceil(len(setting.stations)/2), 2, figsize=(60, 30))

    #!for ker in setting.kernel:
    #!    r2 = 0
    r = 0
    j = 0
    r2_avg_affine = 0
    mae_avg_affine = 0
    rmse_avg_affine = 0
    r2_avg_cosine = 0
    mae_avg_cosine = 0
    rmse_avg_cosine = 0
    r2_avg_gaussian = 0
    mae_avg_gaussian = 0
    rmse_avg_gaussian = 0
    r2_avg_inner_product = 0
    mae_avg_inner_product = 0
    rmse_avg_inner_product = 0
    for station in setting.stations:    
        current_affine = f'{setting.DIR_RESULST}/{DATASOURCE}/attention_kernel_affine/station_{station}'
        current_cosine = f'{setting.DIR_RESULST}/{DATASOURCE}/attention_kernel_cosine/station_{station}'
        current_gaussian = f'{setting.DIR_RESULST}/{DATASOURCE}/attention_kernel_gaussian/station_{station}'
        current_inner_product = f'{setting.DIR_RESULST}/{DATASOURCE}/attention_kernel_inner-product/station_{station}'
        
        #!try:
            #!with open(f'{current_dir}/result.txt', 'rb') as fp:
            #!    dict_results = fp.readline()
            #!print()
            #rmse = np.avg(np.load(f'{current_dir}/rmse_test.npy'))
            #!rmse = np.max(np.load(f'{current_dir}/rmse_test.npy'))#float(str(dict_results).split('RMSE')[1].split(',')[0])
            #!mae = np.max(np.load(f'{current_dir}/mae_test.npy'))#float(str(dict_results).split('MAE')[1].split(',')[0])
            #!r2 += np.max(np.load(f'{current_dir}/r2_test.npy'))#float(str(dict_results).split('R2')[-1][1:-1])
            #!row = {'kernel': ker, 'station_test': station, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
            #!results_data = results_data.append(row, ignore_index=True)
        r2_affine = np.load(f"{current_affine}/r2_test.npy")     
        r2_cosine = np.load(f"{current_cosine}/r2_test.npy")
        r2_gaussian = np.load(f"{current_gaussian}/r2_test.npy")
        r2_inner_product = np.load(f"{current_inner_product}/r2_test.npy")
        
        mae_affine = np.load(f"{current_affine}/mae_test.npy")     
        mae_cosine = np.load(f"{current_cosine}/mae_test.npy")
        mae_gaussian = np.load(f"{current_gaussian}/mae_test.npy")
        mae_inner_product = np.load(f"{current_inner_product}/mae_test.npy")

        rmse_affine = np.load(f"{current_affine}/rmse_test.npy")     
        rmse_cosine = np.load(f"{current_cosine}/rmse_test.npy")
        rmse_gaussian = np.load(f"{current_gaussian}/rmse_test.npy")
        rmse_inner_product = np.load(f"{current_inner_product}/rmse_test.npy")

        dict_max = {'affine': np.argmax(r2_affine), 'cosine': np.argmax(r2_cosine), 'gaussian': np.argmax(r2_gaussian), 
                    'inner-product': np.argmax(r2_inner_product)}
        #print(station, ' - ', np.argmax(r2_affine))
        r2_avg_affine += np.max(r2_affine)
        r2_avg_cosine += np.max(r2_cosine)
        r2_avg_gaussian += np.max(r2_gaussian)
        r2_avg_inner_product += np.max(r2_inner_product)

        mae_avg_affine += mae_affine[dict_max['affine']]
        mae_avg_cosine += mae_cosine[dict_max['cosine']]
        mae_avg_gaussian += mae_gaussian[dict_max['gaussian']]
        mae_avg_inner_product += mae_inner_product[dict_max['inner-product']]

        rmse_avg_affine += rmse_affine[dict_max['affine']]
        rmse_avg_cosine += rmse_cosine[dict_max['cosine']]
        rmse_avg_gaussian += rmse_gaussian[dict_max['gaussian']]
        rmse_avg_inner_product += rmse_inner_product[dict_max['inner-product']]

        #!print(np.argmax(r2_affine), np.argmax(r2_cosine), np.argmax(r2_gaussian), np.argmax(r2_inner_product))
        max_kernel = max(dict_max, key=dict_max.get)
        ax[r, j].text(0.60, 0, f'{station}', horizontalalignment='left', verticalalignment='bottom', transform=ax[r, j].transAxes, fontsize=40)
        dir_max = f'{setting.DIR_RESULST}/{DATASOURCE}/attention_kernel_{max_kernel}/station_{station}'
        index_r2_max = np.argmax(np.load(f"{dir_max}/r2_test.npy"))

        x = np.arange(1, index_r2_max+1, 1)
        ax[r, j].plot(x, r2_affine[:index_r2_max], 'r', label='R2 with kernel Affine', linewidth= 8)
        #!ax[r, j].plot(x, r2_gaussian[:index_r2_max], 'b', label='R2 with kernel Gaussian', linewidth= 8)
        #!ax[r, j].plot(x, r2_inner_product[:index_r2_max], 'g', label='R2 with kernel Inner Product', linewidth= 8)
        #!ax[r, j].plot(x, r2_cosine[:index_r2_max], 'black', label='R2 with kernel Cosine', linewidth= 8)
        #!ax[r, j].legend(loc='lower left', prop={'size': 45})
        #plt.xlabel('Learning epoch')
        ax[r, j].set_ylabel('R2 Score')
        #plt.legend()
        #!print(current_data)
        #!avg_r2 += r2 
        #!print(np.argmax(np.load(f'{current_dir}/r2_test.npy')))
        #!print(station, np.max(np.load(f'{current_dir}/r2_test.npy')))
        #print(avg_r2 / len(setting.stations))
        if r == ax.shape[0] - 1:
            ax[r, j].set_xlabel('Learning epoch', fontsize=40)
            j += 1
            r = 0 
        else:
            r += 1
        
        print(station, np.max(r2_affine), np.max(r2_cosine))
        #!except Exception as e:
        #!    print(e)
        #!    raise f'Not found results for {station}'
        #!r2 /= len(setting.stations)
        #!print(f'{ker}: {r2}')
    #!print(results_data)
    r2_avg_affine /= len(setting.stations)
    r2_avg_cosine /= len(setting.stations)
    r2_avg_gaussian /= len(setting.stations)
    r2_avg_inner_product /= len(setting.stations)

    mae_avg_affine /= len(setting.stations)
    mae_avg_cosine /= len(setting.stations)
    mae_avg_gaussian /= len(setting.stations)
    mae_avg_inner_product /= len(setting.stations)

    rmse_avg_affine /= len(setting.stations)
    rmse_avg_cosine /= len(setting.stations)
    rmse_avg_gaussian /= len(setting.stations)
    rmse_avg_inner_product /= len(setting.stations)
    
    print(f"Affine {round(rmse_avg_affine, 4)}, {round(mae_avg_affine, 4)}, {round(r2_avg_affine, 4)}")
    print(f"Cosine {round(rmse_avg_cosine, 4)}, {round(mae_avg_cosine, 4)}, {round(r2_avg_cosine, 4)}")
    print(f"Gaussian {round(rmse_avg_gaussian, 4)}, {round(mae_avg_gaussian, 4)}, {round(r2_avg_gaussian, 4)}")
    print(f"Inner product {round(rmse_avg_inner_product, 4)}, {round(mae_avg_inner_product, 4)}, {round(r2_avg_inner_product, 4)}")

    if not os.path.isdir('Graphics'):
        os.mkdir('Graphics')
    
    plt.savefig(f'Graphics/{DATASOURCE}_R2_Attention_2.png')   
        #row = {}        
        #print(dict_results)
        #for key_, v in dict_results.items():
        #    current_station = dict_results[key_]
            #for k_, v_ in current_station.items():
        #    row = {'knn': k, 'station_test': key_, 'RMSE': current_station.get('RMSE'), 'MAE': current_station.get('MAE'), 
        #    'R2': current_station.get('R2')}
        #    results_data = results_data.append(row, ignore_index=True)
