# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: experimentation of attention model 
# ******************************************************************************************
import os, sys
#from torch.utils import data
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd 
from Models.commom import *
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from Models.Attention.ANN import ANN
from Models.Attention.attetion_layer import GraphAttentionLayer
from Models.Attention.attention_model import Attention_model
warnings.filterwarnings('ignore')

try:
    DATASOURCE = sys.argv[1]
    kernel = sys.argv[2]
    if not os.path.isdir('Results'):
        os.mkdir('Results')
    if not os.path.isdir(f'Results/{DATASOURCE}'):
        os.mkdir(f'Results/{DATASOURCE}')
except Exception as e:
    raise ValueError('Input SP: Sao paulo Data or BE: Beijing Data and kernel (affine, gaussian, inner-product, cosine) as arguments')


class setting:
    if DATASOURCE == 'BE':
        DIR_DATA = '../Preprocessing/Results/data_train.csv'
        test_station = ['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
                   'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan'] 
        selected_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem']

    elif DATASOURCE == 'SP':
        DIR_DATA = '../Preprocessing/Results/data_train_sp_.csv'
        test_station = ['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
            'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
            'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
        selected_features = ['temp', 'press', 'hum', 'wd', 'ndvi', 'ntl', 'dem']
        
    else:
        raise ValueError("Input DATA should be 'SP' or 'BE'")
    num_epoch = 800#100
    lr_1 = 0.01  #0.04 diferent #ant SGD 0.005 #0.2#0.004 #0.004 #ant 0.003
    lr_2 = 0.001 #0.004 #ant SGD 0.005
    for_normalization = selected_features+['PM25']
    cuda = False 
    criterion = nn.MSELoss()


if __name__ == '__main__':
    data = pd.read_csv(setting.DIR_DATA)
    data_labeled = data[~data['PM25'].isna()]
    data_labeled = data_labeled[~data_labeled['ntl'].isna()]
    del data

    scaled_data = data_labeled.copy()
    scalers = {}
    for column in setting.for_normalization:
        scaled, scaler = normalization(data_labeled, column, 1)
        scaled_data[column] = scaled.reshape(len(data_labeled), 1)
        scalers.update({column: scaler})

    current_directory = f'Results/{DATASOURCE}'
    if not os.path.isdir(current_directory):
        os.mkdir(current_directory)
     
    model_directory = current_directory+f'/attention_kernel_{kernel}'
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    for station in setting.test_station:
        print(station)
        
        current_station = f'{model_directory}/station_{station}'
        if not os.path.isdir(current_station):
            os.mkdir(current_station)

        train, test = split_data(scaled_data, station)
        
        data_loader = []
        for date in train.current_date.unique().tolist():
            data_instance = {}
            current_data = train[train['current_date']==date]
            x_train = current_data[setting.for_normalization].values
            y_train = current_data['PM25'].values

            current_test = test[test['current_date']==date]
            x_test = current_test[setting.for_normalization].values
            y_test = current_test['PM25'].values

            x_train = to_tensor(x_train, is_cuda=setting.cuda)
            y_train = to_tensor(y_train, is_cuda=setting.cuda)
            x_test = to_tensor(x_test, is_cuda=setting.cuda)
            y_test = to_tensor(y_test, is_cuda=setting.cuda)

            data_instance.update({"x_train": x_train,
                                "y_train": y_train, 
                                "x_test": x_test, 
                                "y_test": y_test})
            data_loader.append(data_instance)

        number_features = len(setting.selected_features)
        attention_layer = GraphAttentionLayer(number_features, number_features, k=None,
                                    no_feature_transformation=True, graph=None,
                                    kernel=kernel, nonlinearity_1=None, nonlinearity_2=nn.LeakyReLU())
        
        layer_optim = torch.optim.SGD(params=[attention_layer.a], lr=setting.lr_1)

        ann = ANN(number_features*2+1, 1)
        ann = init_net(ann, device='cpu')

        ann_optim = torch.optim.SGD(ann.parameters(), lr=setting.lr_2)
    
        attention_model = Attention_model(attention_layer, ann, None, setting.criterion, layer_optim,
                        ann_optim, setting.num_epoch, scalers['PM25'], kernel, DATASOURCE)

        attention_layer_trained, ann_trained, results, msg_final = attention_model.training(data_loader, current_station)

        for k, v in results.items():
            current_file = f'{current_station}/{k}.npy'
            np.save(current_file, v)

