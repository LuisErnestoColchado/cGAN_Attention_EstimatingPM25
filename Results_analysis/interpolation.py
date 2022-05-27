#*****************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Generate Interpolation maps
#******************************************************************************************
import os, sys

from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
from datetime import timedelta
import shapefile as shp
from Models.Attention.ANN import ANN
from Models.Attention.attetion_layer import GraphAttentionLayer
from Models.commom import *
from Models.cGANSL.generator import Generator
import Models.IDW as IDW
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')


try: 
    DATASOURCE = sys.argv[1]
    if not DATASOURCE in ['BE', 'SP']:
        raise 'Invalid parameters: 1: BE or SP'
except Exception as e:
    raise 'Not found parameters: input DATASOURCE: SP or BE'


class setting:
    z_dim = 100
    if DATASOURCE == 'BE':
        dir_results = '../Experimentation/Results/BE'    
        stations = ['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
                   'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan']
        knn = 3 
        dir_cgansl = f"../Models/cGANSL/Backup/cgansl_advloss1_spl2_knn{knn}"
        sel_test_station = "Shunyi"
        selected_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem']
        dir_cgansl_r2 = f"../Experimentation/Results/{DATASOURCE}/cgansl_advloss1_spl2_knn{knn}" 
        dir_data = f"../Preprocessing/Results/data_train.csv"
        start_date = pd.to_datetime("2015-02-01") 
        end_date = pd.to_datetime("2015-02-01")
        condition_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem']
        dir_shp = "shp/beijing.shp"
        name_city = 'Beijing'
    else:
        dir_results = '../Experimentation/Results/SP'   
        stations = ['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
                  'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
             'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
        knn = 10
        dir_cgansl = f"../Models/cGANSL/Backup/cgansl_advloss0.2_spl0.8_knn{knn}"
        sel_test_station = "Itaim Paulista"
        selected_features = ['temp', 'press', 'hum', 'wd', 'ndvi', 'ntl', 'dem']
        dir_cgansl_r2 = f"../Experimentation/Results/{DATASOURCE}/cgansl_advloss0.2_spl0.8_knn{knn}" 
        dir_data = f"../Preprocessing/Results/data_train_sp_.csv"
        start_date = pd.to_datetime("2018-07-01")
        end_date = pd.to_datetime("2018-07-01")#pd.to_datetime("2018-08-31")
        condition_features = ['temp', 'press', 'hum', 'wd', 'ndvi', 'ntl', 'dem']
        dir_shp = "shp/sao_paulo.shp"
        name_city = 'Sao Paulo'
    z_dim = 100
    dir_cgansl_r2 = f"{dir_cgansl_r2}/station_{sel_test_station}/r2_test.npy"
    dir_attention_r2 = f"../Experimentation/Results/{DATASOURCE}/attention_kernel_affine/station_{sel_test_station}/r2_test.npy"
    dir_attention = f"../Models/Attention/Backup/{DATASOURCE}_affine"
    for_normalization = selected_features+['PM25']
    input_g = len(selected_features)+(knn*2)+z_dim
    hidden_g = 100
    output_g = 1
    f_sig = torch.nn.Sigmoid()  
    f_relu = torch.nn.LeakyReLU() 
    idw_cols_train = ['lat', 'lon', 'PM25']
    idw_cols_test = ['lat', 'lon']
    p = 1


def load_map_shp(shp_path):
    sf = shp.Reader(shp_path, encoding='latin-1')
    return sf


def load_attention(epoch):
    number_features = len(setting.selected_features)
    dir_attention = f"{setting.dir_attention}/attention_layer_{epoch}.pt"
    attention_layer = GraphAttentionLayer(number_features, number_features, k=10,
                                    no_feature_transformation=True, graph=None,
                                    kernel="affine", nonlinearity_1=None, nonlinearity_2=nn.LeakyReLU())
    attention_layer.load_state_dict(torch.load(dir_attention))
    attention_layer.eval()

    dir_ann = f"{setting.dir_attention}/ann_{epoch}.pt"   
    ann = ANN(number_features*2+1, 1)
    ann.load_state_dict(torch.load(dir_ann))
    ann.eval() 
    return attention_layer, ann 


def load_cgansl_g(epoch):
    dir_g = f"{setting.dir_cgansl}/g_model_{DATASOURCE}_{epoch}_{setting.sel_test_station}.pt"
    g = Generator(setting.input_g, setting.hidden_g, setting.output_g, f=setting.f_sig, f_relu=setting.f_relu)
    g.load_state_dict(torch.load(dir_g))
    g.eval()
    return g 

    
def preparing_data(labeled_data, data_, name_model='attention'):
    if name_model == 'attention':
        data_ = data_[~data_['ntl'].isna()]
        scaled_data = data_.copy()
        scalers = {}
        for column in setting.for_normalization:
            scaled, scaler = normalization_interpolation(labeled_data, data_, column)
            scaled_data[column] = scaled.reshape(len(data_), 1)
            scalers.update({column: scaler})
        return scaled_data, scalers


def interpolation(x_tensor, name_model, model, ann=None):
    if name_model == 'attention':
        _, output_atten, _ = model(x_tensor)
        pm25_inter = ann(output_atten, DATASOURCE=DATASOURCE)
    elif name_model == 'cgansl':
        pm25_inter = model(x_tensor)
    return pm25_inter.detach().numpy()[-1]
    

if __name__ == '__main__':
    
    values_knn = []
    distance_features = []
    for k in range(setting.knn):
        values_knn.append(f'pm25_{k}')
        distance_features.append(f'dist_{k}')
    features_training = setting.condition_features+values_knn+distance_features

    data_frame = pd.read_csv(setting.dir_data)
    r2_attention = np.load(setting.dir_attention_r2)
    r2_cgansl = np.load(setting.dir_cgansl_r2)

    sf = load_map_shp(setting.dir_shp)
    
    max_atten_index = 500 if DATASOURCE == 'BE' else 328 #SP328#np.argmax(r2_attention)
    max_cgan_index = np.argmax(r2_cgansl)
    
    print(max_cgan_index)
    for j, v in enumerate(r2_cgansl):
        print(j, v)

    atten_layer, ann_model = load_attention(max_atten_index)
    cgansl_model = load_cgansl_g(max_cgan_index)
    
    labeled_data = data_frame[~data_frame['PM25'].isna()]
    lat_min, lat_max = min(data_frame['lat']), max(data_frame['lat'])
    lon_min, lon_max = min(data_frame['lon']), max(data_frame['lon'])

    scaled_data, scalers = preparing_data(labeled_data, data_frame, 'attention')
    scaled_data_cgansl, scalers_cgansl = preparing_data(data_frame, data_frame, 'attention')
    
    d_ = setting.start_date
    with tqdm(total=(setting.end_date - setting.start_date).days) as bar:
        while d_ <= setting.end_date:
            
            curr_data = scaled_data[scaled_data['current_date']==str(d_.date())]
            lab_data = curr_data[~curr_data['PM25'].isna()]
            unlab_data = curr_data[curr_data['PM25'].isna()]
            lab_data = lab_data[~lab_data['ntl'].isna()]
            x_lab = lab_data[setting.for_normalization].values

            curr_data_cgansl = scaled_data_cgansl[scaled_data_cgansl['current_date']==str(d_.date())]
            lab_data_cgansl = curr_data_cgansl[~curr_data_cgansl['PM25'].isna()]
            unlab_data_cgansl = curr_data_cgansl[curr_data_cgansl['PM25'].isna()]

            idw_train = lab_data[setting.idw_cols_train]
            idw_tree = IDW.tree(idw_train.values[:, :2], idw_train.values[:, -1])
            for i in unlab_data.index:
                x_unlab = unlab_data[setting.for_normalization].loc[i].values
                concat_data =  np.concatenate([x_lab, x_unlab.reshape(1, len(x_unlab))])
                x_tensor = to_tensor(concat_data, is_cuda=False)
                pm25 = interpolation(x_tensor=x_tensor, name_model='attention',
                                    model=atten_layer, ann=ann_model)
                pm25 = scalers['PM25'].inverse_transform(pm25.reshape(-1, 1))
                unlab_data.loc[i, "PM25_atten"] = pm25[-1]

                x_unlab_cgansl = unlab_data_cgansl[features_training].loc[i].values
                z = np.random.rand(1, setting.z_dim)
                x_unlab_cgansl = x_unlab_cgansl.reshape(1, len(x_unlab_cgansl))
                concat_cgansl = np.concatenate([z, x_unlab_cgansl], axis=1)
                x_tensor = to_tensor(concat_cgansl, is_cuda=False)
                pm25 = interpolation(x_tensor=x_tensor, name_model='cgansl', 
                                    model=cgansl_model)
                pm25 = scalers_cgansl['PM25'].inverse_transform(pm25.reshape(-1, 1))
                unlab_data.loc[i, "PM25_cgansl"] = pm25[-1]

                x_test = unlab_data[setting.idw_cols_test].loc[i].values.reshape(1, 2)
                z = idw_tree(x_test, k=setting.knn, p=setting.p)
                pm25_idw = scalers['PM25'].inverse_transform(z.reshape(-1, 1))
                unlab_data.loc[i, "PM25_IDW"] = pm25_idw[-1]

            fig, axs = plt.subplots(figsize=(25, 15), nrows=2, ncols=3)
            fig.suptitle(f'{setting.name_city} ({str(d_.date())})', fontsize=22)
            x = []
            y = []
            for shape in sf.shapeRecords():
                for i in shape.shape.points[:]:
                    if i[0] <= lon_max and i[0] >= lon_min and i[1] <= lat_max and i[1] >= lat_min:
                        y.append(i[0])
                        x.append(i[1])
            axs[0, 0].plot(x,y, c='black', linewidth=0.02)
            axs[0, 1].plot(x, y, c='black', linewidth=0.02)
            axs[0, 2].plot(x, y, c='black', linewidth=0.02) 
            axs[1, 0].plot(x, y, c='black', linewidth=0.02)  
            axs[1, 1].plot(x, y, c='black', linewidth=0.02) 

            lab_data['PM25_inv'] = scalers['PM25'].inverse_transform(lab_data['PM25'].values.reshape(-1, 1)) 

            im = axs[0, 0].scatter(x=lab_data['lat'], y=lab_data['lon'], c=lab_data['PM25_inv'], marker='s', s=150, cmap='YlOrBr')
            ax = axs[0, 0]
            ax.set_title('PM2.5 by AQM stations', fontsize=18)
            fig.colorbar(im, ax=ax) 


            im = axs[0, 1].scatter(x=unlab_data['lat'], y=unlab_data['lon'], c=unlab_data['PM25_IDW'], marker='s', s=150, cmap='YlOrBr')
            ax = axs[0, 1]
            ax.set_title('PM2.5 by Inverse Distance Weighting', fontsize=18)
            fig.colorbar(im, ax=ax)
        
            im = axs[0, 2].scatter(x=unlab_data['lat'], y=unlab_data['lon'], c=unlab_data['PM25_atten'], marker='s', s=150, cmap='YlOrBr')
            ax = axs[0, 2]
            ax.set_title('PM2.5 by Neural Network with Attention Layer', fontsize=18)
            fig.colorbar(im, ax=ax)
        

            im = axs[1, 0].scatter(x=unlab_data['lat'], y=unlab_data['lon'], c=unlab_data['PM25_cgansl'], marker='s', s=150, cmap='YlOrBr')
            ax = axs[1, 0]
            ax.set_title('PM2.5 by Generative Model (cGAN & Spatial Learning)', fontsize=18)
            fig.colorbar(im, ax=ax)

            im = axs[1, 1].scatter(x=curr_data['lat'], y=curr_data['lon'], c=curr_data['ndvi'], marker='s', s=150, cmap='YlGn')
            ax = axs[1, 1]
            ax.set_title('NDVI (Vegetation Index)', fontsize=18)
            fig.colorbar(im, ax=ax)


            im = axs[1, 2].scatter(x=curr_data['lat'], y=curr_data['lon'], c=curr_data['ntl'], marker='s', s=150, cmap='gray')
            ax = axs[1, 2]
            ax.set_title('NTL (~Population Level)', fontsize=18)
            fig.colorbar(im, ax=ax)
        
            plt.savefig(f"maps/{DATASOURCE}/inter_{DATASOURCE}_{d_}", bbox_inches = 'tight')
            bar.update(1)
            d_ += timedelta(days=1)
