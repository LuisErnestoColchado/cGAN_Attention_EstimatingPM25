import os, sys
from numpy.lib.shape_base import split
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import pandas as pd
from Models.cGAN.cgan import cGAN, normalization, to_tensor, split_data
from Models.cGAN.generator import Generator
from Models.cGAN.discriminator import Discriminator
import torch


class setting:
    knn = 3
    num_epoch = 100
    condition_features = ['temp', 'dewp', 'wd', 'ws', 'ndvi', 
                         'ntl', 'dem']
    to_normalization = ['temp', 'dewp', 'dewp', 'wd', 'ws', 'ndvi', 
                         'ntl', 'dem', 'pollution', 'distances']
    station_test = 'Dongsi'
    g_hidden_size = 100
    z_dim = 100
    g_input_size = len(condition_features)+z_dim
    g_output_size = 1
    d_input_size = 1
    d_hidden_size = 100
    d_output_size = 1
    batch_size = 980 #number of grids qgis 
    f_sig = torch.nn.Sigmoid() 
    f_relu = torch.nn.LeakyReLU()
    g_steps = 3
    d_steps = 1
    g_lr = 0.0001
    d_lr = 0.00001
    spatial_loss = True
    cuda = True


values_knn = []
distance_features = []
for k in range(setting.knn):
    values_knn.append(f'pm25_{k}')
    distance_features.append(f'dist_{k}')
features_training = setting.condition_features+values_knn+distance_features+['PM25']
pollution_features =  values_knn+['PM25']

data_train = pd.read_csv('../Preprocessing/Results/data_train.csv')
train_data = data_train[['station']+features_training]


scaled_data = train_data.copy()
scalers = {}
for x in setting.to_normalization:
    if x == 'pollution':
        column = pollution_features
        size_column = len(column)
    elif x == 'distances':
        column = distance_features
        size_column = len(column)
    else:
        column = x
        size_column = 1
    #print(train_data[column].shape)

    scaled, scaler = normalization(train_data, column, size_column)
    scaled_data[column] = scaled.reshape(len(train_data), size_column)
    scalers.update({x: scaler})


generator = Generator(input_size=setting.g_input_size, hidden_size=setting.g_hidden_size,
                      output_size=setting.g_output_size, f=setting.f_sig, f_relu=setting.f_relu)
discriminator = Discriminator(input_size=setting.d_input_size, hidden_size=setting.d_hidden_size,
                      output_size=setting.d_output_size, f=setting.f_sig, f_relu=setting.f_relu)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=setting.g_lr)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=setting.d_lr)


cgan_model = cGAN(generator, discriminator, setting.z_dim, setting.g_steps, setting.d_steps,
             setting.num_epoch, scalers['pollution'], scalers['distances'], g_optimizer,
            d_optimizer, spatial_loss=setting.spatial_loss)


train, test = split_data(scaled_data, setting.station_test)

features = train[setting.condition_features].values
knn_values = train[values_knn].values
distances = train[distance_features].values
x_real = train[]


features_test = test[setting.condition_features]
x_test = 

#t_features = to_tensor(features, setting.cuda)
#t_knn_values = to_tensor(knn_values, setting.cuda)
#t_distances = to_tensor(distances, setting.cuda)


