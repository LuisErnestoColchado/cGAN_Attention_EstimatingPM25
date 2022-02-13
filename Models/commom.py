from torch.nn import init
import numpy as np
import torch
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def normalization_interpolation(data_all, data, column):
    scaler = preprocessing.MinMaxScaler()
    values_all = data_all[column].values
    values = data[column].values
    scaler = scaler.fit(values_all.reshape(-1, 1))
    scaled_data = scaler.transform(values.reshape(-1, 1))
    return scaled_data, scaler

def normalization(data, column, size_columns):
    scaler = preprocessing.MinMaxScaler()
    values = data[column].values
    scaler = scaler.fit(values.reshape(-1, 1))
    scaled_data = scaler.transform(values.reshape(-1, 1))
    return scaled_data, scaler


def split_data(data, station):
    data_train = data[data['station'] != station]
    data_test = data[data['station'] == station]
    return data_train, data_test


def split_temp_eval(data, min_date_test, max_date_test):
    data_train = data[data['current_date'] < min_date_test]
    data_test = data[(data['current_date'] >= min_date_test) & 
                     (data['current_date'] <= max_date_test)]
    return data_train, data_test


def split_attention(data, station):
    data_train = data.copy()
    data_test = data[data['station'] == station]
    return data_train, data_test
        
def to_tensor(x, is_cuda):
    FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    return FloatTensor(x) 

def load_data_cgan(condition, knn_dist, knn_values, x, batch_size):
    data = TensorDataset(condition, knn_dist, knn_values, x)
    loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    return loader


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(1.)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', init_gain=0.02, device='cuda:0'):
    net.to(device)
    init_weights(net, init_type, gain=init_gain)
    return net


def extract(v):
    return v.data.storage().tolist()


def stats(d):
    return [np.mean(d), np.std(d)]

