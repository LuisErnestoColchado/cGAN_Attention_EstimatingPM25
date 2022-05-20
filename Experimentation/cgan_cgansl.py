import os, sys
from re import M

from numpy import DataSource

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import pandas as pd
from Models.cGANSL.cgan import cGAN, normalization, to_tensor, split_data, load_data_cgan
from Models.cGANSL.generator import Generator
from Models.cGANSL.discriminator import Discriminator
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    DATASOURCE = sys.argv[1]
    knn = int(sys.argv[2])
    if not os.path.isdir('Results'):
        os.mkdir('Results')
    if not os.path.isdir(f'Results/{DATASOURCE}'):
        os.mkdir(f'Results/{DATASOURCE}')
except Exception as e:
    raise ValueError('Input SP: Sao paulo Data or BE: Beijing Data and knn (3, 5, 7) as arguments')

class setting:
    #DATASOURCE = 'BE' # SP: Sao Paulo, BE: Beijing
    if DATASOURCE == 'BE':
        DIR_DATA = '../Preprocessing/Results/data_train.csv'
        condition_features = ['temp', 'pres', 'dewp', 'wd', 'ws', 'ndvi',
                         'ntl', 'dem']
        train_labels = 11
        station_test = ['Aotizhongxin', 'Dingling', 'Changping', 'Dongsi', 'Gucheng', 'Guanyuan',  'Huairou',
                        'Nongzhanguan', 'Shunyi', 'Wanliu', 'Wanshouxigong', 'Tiantan'] 
    elif DATASOURCE == 'SP':
        DIR_DATA = '../Preprocessing/Results/data_train_sp_.csv'
        condition_features = ['temp', 'press', 'hum', 'wd', 'ndvi', 'ntl', 'dem']
        train_labels = 10
        station_test = ['Parelheiros', 'Ibirapuera', 'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']#['Osasco', 'Pico do Jaraguá', 'Marg.Tietê-Pte Remédios',
            #'Cid.Universitária-USP-Ipen', 'Pinheiros', 'Parelheiros', 'Ibirapuera',
            #'Congonhas', 'Santana', 'Parque D.Pedro II', 'Itaim Paulista']
    else:
        raise ValueError("Input DATA should be 'SP' or 'BE'")


    for_normalization =  condition_features+['pollution', 'distances']
    g_hidden_size = 100
    z_dim = 100 
    g_input_size = len(condition_features)+(knn*2)+z_dim
    g_output_size = 1
    d_input_size = 1+(len(condition_features)+(knn*2)) 
    d_hidden_size = 100# number neuron in hidden layer 
    d_output_size = 1 
    
    f_sig = torch.nn.Sigmoid()  
    f_relu = torch.nn.LeakyReLU() 
    spatial_loss = False #FALSE: cGAN without SLoss (Only adversarial learning) 
    spatial_paramn = 0# cGAN with Sloss 
    adv_paramn = 1
    if not spatial_loss: # Only adverarial learning cGAN
        batch_size = 1000
        spatial_paramn = 0
        adv_paramn = 1
        #!if DATASOURCE == 'BE':
        #!    num_epoch = 800
        #!else:
        num_epoch = 800

        #!g_steps = 3
        #!d_steps = 1
        #!g_lr = 0.001#4#sp 0.000001 # SP 
        #!d_lr = 0.001#4#sp 0.000001 # SP 
    else:
        #!if DATASOURCE == 'BE':
        num_epoch = 100
        #!    g_steps = 3
        #!    d_steps = 1
        #!else:
        #!    num_epoch = 100
        

    #!if DATASOURCE == 'BE':
        #g_steps = 2
        #d_steps = 1
        #SṔ 0.0001#0.000001
    #!    g_lr = 0.001 #0.00001
    #!    d_lr = 0.001
    #!else:
    g_lr = 0.001 #0.00001
    d_lr = 0.001
    g_steps = 2 # 3
    d_steps = 1
    cuda = torch.cuda.is_available()
    

if __name__ == '__main__':  
    values_knn = []
    distance_features = []
    for k in range(knn):
        values_knn.append(f'pm25_{k}')
        distance_features.append(f'dist_{k}')
    features_training = setting.condition_features+values_knn+distance_features+['PM25']
    pollution_features =  values_knn+['PM25']

    data_train = pd.read_csv(setting.DIR_DATA)
    train_data = data_train[['station', 'current_date']+features_training]
    
    train_data = train_data[~train_data['ntl'].isna()]
    
    if not setting.spatial_loss:
        train_data = train_data[~train_data.PM25.isna()].reset_index(drop=True)

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

        scaled, scaler = normalization(train_data, column, size_column)
        scaled_data[column] = scaled.reshape(len(train_data), size_column)
        scalers.update({x: scaler})
    
    print(f'******************\n Training and Testing cGAN for {DATASOURCE} \n******************')
    directory_data = f'Results/{DATASOURCE}'
    if not os.path.isdir(directory_data):
        os.mkdir(directory_data)
    current_directory = f'{directory_data}/cgansl_advloss{setting.adv_paramn}_spl{setting.spatial_paramn}_knn{knn}'
    if not os.path.isdir(current_directory):
        os.mkdir(current_directory)

    cross_test = {}
    for station_ in setting.station_test:
        generator = Generator(input_size=setting.g_input_size, hidden_size=setting.g_hidden_size,
                            output_size=setting.g_output_size, f=setting.f_sig, f_relu=setting.f_relu)
        discriminator = Discriminator(input_size=setting.d_input_size, hidden_size=setting.d_hidden_size,
                            output_size=setting.d_output_size, f=setting.f_sig, f_relu=setting.f_relu)

        g_optimizer = torch.optim.Adam(generator.parameters(), lr=setting.g_lr) # ADAM SP
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=setting.d_lr) #ADAM SP

        cgan_model = cGAN(generator, discriminator, setting.z_dim, setting.g_steps, setting.d_steps, setting.num_epoch, scalers['pollution'], scalers['distances'],
         g_optimizer, d_optimizer, parameter_spatial=setting.spatial_paramn, parameter_adversarial=setting.adv_paramn, 
         spatial_loss=setting.spatial_loss, train_labels=setting.train_labels)


        print(f'******************\n Test station: {station_}\n******************')
        train, test = split_data(scaled_data, station_)

        condition_test = test[setting.condition_features+values_knn+distance_features].values
        x_real_test = test['PM25'].values
        condition_test = to_tensor(condition_test, True)
        x_real_test = to_tensor(x_real_test, True)
        
        data_loader = []
        if setting.spatial_loss:
            for date in train.current_date.unique().tolist():
                data_instance = []
                current_data = train[train['current_date']==date]
                condition = current_data[setting.condition_features+values_knn+distance_features].values
                knn_values = current_data[values_knn].values
                distances = current_data[distance_features].values
                x_real = current_data['PM25'].values
                fake_data_batch = np.zeros((len(x_real), 1))
                if np.isnan(np.sum(condition)):
                    print(date)
                condition = to_tensor(condition, is_cuda=setting.cuda)
                knn_values = to_tensor(knn_values, is_cuda=setting.cuda)
                distances = to_tensor(distances, is_cuda=setting.cuda)
                x_real = to_tensor(x_real, is_cuda=setting.cuda)
                data_instance.append(condition)
                data_instance.append(distances)
                data_instance.append(knn_values)
                data_instance.append(x_real)
                data_loader.append(data_instance)
        else:
            condition = train[setting.condition_features+values_knn+distance_features].values
            knn_values = train[values_knn].values
            distances = train[distance_features].values
            x_real = train['PM25'].values

            condition = to_tensor(condition, is_cuda=setting.cuda)
            knn_values = to_tensor(knn_values, is_cuda=setting.cuda)
            distances = to_tensor(distances, is_cuda=setting.cuda)
            x_real = to_tensor(x_real, is_cuda=setting.cuda)
            dataset_train = TensorDataset(condition, distances, knn_values, x_real)
            data_loader = DataLoader(dataset=dataset_train, batch_size=setting.batch_size, shuffle=False)
            
        current_station = f'{current_directory}/station_{station_}'
        if not os.path.isdir(current_station):
            os.mkdir(current_station)

        results, msg_final = cgan_model.training_test(data_loader, condition_test, x_real_test, knn, current_station, station_)    
        cross_test.update({station_: results})

        for k, v in results.items():
            current_file = f'{current_station}/{k}.npy'
            np.save(current_file, v)
        
        print('******************\n End test \n******************')
    
