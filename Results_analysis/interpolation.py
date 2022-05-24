#*****************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Generate Interpolation maps
#******************************************************************************************
import os, sys
from traceback import print_tb

from numpy import interp

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
from datetime import timedelta
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from Models.Attention.ANN import ANN
from Models.Attention.attetion_layer import GraphAttentionLayer
from Models.commom import *
from Models.cGANSL.generator import Generator

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
        sel_test_station = "Tiantan"
        selected_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem']
        dir_cgansl_r2 = f"../Experimentation/Results/{DATASOURCE}/cgansl_advloss1_spl2_knn{knn}" 
        dir_data = f"../Preprocessing/Results/data_train.csv"
        start_date = pd.to_datetime("2015-04-01") 
        end_date = pd.to_datetime("2015-04-01")
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
        start_date = pd.to_datetime("2018-05-01")
        end_date = pd.to_datetime("2018-05-01")
    dir_cgansl_r2 = f"{dir_cgansl_r2}/station_{sel_test_station}/r2_test.npy"
    dir_attention_r2 = f"../Experimentation/Results/{DATASOURCE}/attention_kernel_affine/station_{sel_test_station}/r2_test.npy"
    dir_attention = f"../Models/Attention/Backup/{DATASOURCE}_affine"
    for_normalization = selected_features+['PM25']
    input_g = len(selected_features)+(knn*2)+z_dim
    hidden_g = 100
    output_g = 1
    f_sig = torch.nn.Sigmoid()  
    f_relu = torch.nn.LeakyReLU() 
    
    


def load_map_shp(shp_file):
    reader_map = shpreader.Reader(shp_file)
    map = list(reader_map.geometries())
    MAP = cfeature.ShapelyFeature(map, ccrs.PlateCarree())
    return MAP


def load_attention(epoch):
    number_features = len(setting.selected_features)
    dir_attention = f"{setting.dir_attention}/attention_layer_{epoch}.pt"
    attention_layer = GraphAttentionLayer(number_features, number_features, k=None,
                                    no_feature_transformation=True, graph=None,
                                    kernel="affine", nonlinearity_1=None, nonlinearity_2=nn.LeakyReLU())
    attention_layer = torch.load(dir_attention)
    attention_layer.eval()

    dir_ann = f"{setting.dir_attention}/ann_{epoch}.pt"   
    ann = ANN(number_features*2+1, 1)
    ann = torch.load(dir_ann)
    ann.eval() 
    return attention_layer, ann 


def load_cgansl_g(epoch):
    dir_g = f"{setting.dir_cgansl}/g_model_{DATASOURCE}_{epoch}_{setting.sel_test_station}.pt"
    g = Generator(setting.input_g, setting.hidden_g, setting.output_g, f=setting.f_sig, f_relu=setting.f_relu)
    g.load_state_dict(torch.load(dir_g))
    g.eval()
    return g 

    
def preparing_data(data_, name_model='attention'):
    if name_model == 'attention':
        data_ = data_[~data_['ntl'].isna()]
        scaled_data = data_.copy()
        scalers = {}
        for column in setting.for_normalization:
            scaled, scaler = normalization(data_, column, 1)
            scaled_data[column] = scaled.reshape(len(data_), 1)
            scalers.update({column: scaler})
        return scaled_data, scalers
    #elif name_model == 'cgansl':



def interpolation(data_unl, name_model, model, ann=None):
    if name_model == 'attention':
        pm25_inter = atten_layer(data_unl)
        return pm25_inter
    #TODO 
    #!elif name_model == 'cgansl':

    #!elif name_model == 'idw':
    
    #!elif name_model == 'ok':
    

if __name__ == '__main__':
    data_frame = pd.read_csv(setting.dir_data)
    r2_attention = np.load(setting.dir_attention_r2)
    r2_cgansl = np.load(setting.dir_cgansl_r2)

    max_atten_index = np.argmax(r2_attention)
    max_cgan_index = np.argmax(r2_cgansl)

    atten_layer, ann_model = load_attention(max_atten_index)
    cgansl_model = load_cgansl_g(max_cgan_index)

    #!results_atten = interpolation(data_=data_frame, name_model="attention", 
     #                   model=atten_layer, ann=ann_model)

    d_ = setting.start_date
    while d_ <= setting.end_date:
        curr_data = data_frame[data_frame['current_date']==str(d_.date())]
        #unl_data = curr_data[curr_data['PM25'].isna()]
        scaled_data, scalers = preparing_data(curr_data, 'attention')
         
        d_ += timedelta(days=1)




    





    #!for n, ax in enumerate(axes.flat):
    #!    mapp = load_map_shp('beijing_map.shp')

