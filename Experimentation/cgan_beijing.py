import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pandas as pd
from Models.cGAN.cgan import cGAN, normalization, to_tensor, split_data, load_data_cgan
from Models.cGAN.generator import Generator
from Models.cGAN.discriminator import Discriminator
import torch


class setting:
    DIR_DATA = '../Preprocessing/Results/data_train.csv'
    knn = 3
    train_labels = 11
    num_grid = 952
    num_epoch = 6
    condition_features = ['temp', 'dewp', 'wd', 'ws', 'ndvi', 
                         'ntl', 'dem']
    for_normalization = ['temp', 'dewp', 'dewp', 'wd', 'ws', 'ndvi', 
                         'ntl', 'dem', 'pollution', 'distances']
    station_test = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou',
                   'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
    g_hidden_size = 100
    z_dim = 100
    g_input_size = len(condition_features)+knn+z_dim
    g_output_size = 1
    d_input_size = 11+(len(condition_features)+knn)*11
    d_hidden_size = 100
    d_output_size = 1
    batch_size = 1000 
    f_sig = torch.nn.Sigmoid() 
    f_relu = torch.nn.LeakyReLU()
    g_steps = 1
    d_steps = 1
    g_lr = 0.0001
    d_lr = 0.00001
    spatial_loss = True
    spatial_paramn = 0.4
    cuda = True#torch.cuda.is_avilable()


if __name__ == '__main__':
    values_knn = []
    distance_features = []
    for k in range(setting.knn):
        values_knn.append(f'pm25_{k}')
        distance_features.append(f'dist_{k}')
    features_training = setting.condition_features+values_knn+distance_features+['PM25']
    pollution_features =  values_knn+['PM25']

    data_train = pd.read_csv(setting.DIR_DATA)
    train_data = data_train[['station', 'current_date']+features_training]


    scaled_data = train_data.copy()
    scalers = {}
    for x in setting.for_normalization:
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
                d_optimizer, spatial_loss=setting.spatial_loss, parameter_spatial=setting.spatial_paramn)

    cross_test = {}
    for station_ in setting.station_test:
        print(f'******************\n Test station: {station_}\n******************')
        train, test = split_data(scaled_data, station_)

        # Batch 
        condition = train[setting.condition_features+values_knn].values
        knn_values = train[values_knn].values
        distances = train[distance_features].values
        x_real = train['PM25'].values

        condition_test = test[setting.condition_features+values_knn].values
        x_real_test = test['PM25'].values
        condition_test = to_tensor(condition_test, True)
        x_real_test = to_tensor(x_real_test, True)


        data_loader = []
        for date in train.current_date.unique().tolist():
            data_instance = {}
            current_data = train[train['current_date']==date]
            condition = current_data[setting.condition_features+values_knn].values
            knn_values = current_data[values_knn].values
            distances = current_data[distance_features].values
            x_real = current_data['PM25'].values
            fake_data_batch = np.zeros((len(x_real), 1))

            condition = to_tensor(condition, is_cuda=setting.cuda)
            knn_values = to_tensor(knn_values, is_cuda=setting.cuda)
            distances = to_tensor(distances, is_cuda=setting.cuda)
            fake_data_batch = to_tensor(fake_data_batch, is_cuda=setting.cuda)
            x_real = to_tensor(x_real, is_cuda=setting.cuda)
            data_instance.update({'condition': condition,
                                'fake_data_batch': fake_data_batch, 
                                'knn_values': knn_values,
                                'distances': distances, 
                                'x_real': x_real})
            data_loader.append(data_instance)

        results = cgan_model.training_test(data_loader, condition_test, x_real_test)    
        cross_test.update({station_: results})
        print('******************\n End test \n******************')
