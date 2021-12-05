# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: experimentation of attention model 
# ******************************************************************************************
from inspect import isclass
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


class setting:
    DIR_DATA = '../Preprocessing/Results/data_train.csv'
    #cuda = torch.cuda.is_available()
    test_station = 'Changping'#['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou',
                   # 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
    selected_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi', 'ntl', 'dem']
    for_normalization = selected_features+['PM25']
    kernel = 'affine' #default: affine kernel
    cuda = False
    hidden_layer_ann = 100#32 #100
    criterion = nn.MSELoss()
    num_epoch = 500#100
    lr = 0.01# MEJOR 0.01


if __name__ == '__main__':
    data = pd.read_csv(setting.DIR_DATA)
    #print(setting.cuda)
    data_labeled = data[~data['PM25'].isna()]
    del data
    print(data_labeled.PM25.describe())
    scaled_data = data_labeled.copy()
    scalers = {}
    for column in setting.for_normalization:
        scaled, scaler = normalization(data_labeled, column, 1)
        scaled_data[column] = scaled.reshape(len(data_labeled), 1)
        scalers.update({column: scaler})

    train, test = split_data(scaled_data, setting.test_station)
    
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
    

    y_test = test.values[:, -1].astype(float)

    y_test = to_tensor(y_test, setting.cuda)
    number_features = len(setting.selected_features)
    attention_layer = GraphAttentionLayer(number_features, number_features, k=None,#10 - feature_subset=torch.LongTensor([0, 1]),
                                no_feature_transformation=True,
                                #graph=torch.LongTensor([[0, 5, 1], [3, 4, 6], [1, 3, 5], [1, 3, 4]]), # array 3x3 3 indices 3 neighbors
                                graph=None,#graph_labeled,  
                                kernel=setting.kernel, nonlinearity_1=nn.LeakyReLU(0.2), nonlinearity_2=nn.LeakyReLU(0.2))#nn.LeakyReLU())     
    #0.1
    layer_optim = torch.optim.SGD(params=[attention_layer.a], lr=setting.lr)

    ann = ANN(number_features*2+1, setting.hidden_layer_ann, 1)
    ann_optim = torch.optim.SGD(ann.parameters(), lr=setting.lr)
    # ADAM MEJOR
    attention_model = Attention_model(attention_layer, ann, None, setting.criterion, layer_optim,
                      ann_optim, setting.num_epoch, scalers['PM25'])

    attention_model.training(data_loader)
