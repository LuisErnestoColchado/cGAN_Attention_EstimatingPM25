# **********************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Aritificial Neural Network class
# **********************************************************************************
##

import torch
import torch.nn as nn


class ANN(nn.Module):
    def __init__(self, input_size, num_neurons, output_size):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(input_size, num_neurons)
        self.batch1 = nn.BatchNorm1d(num_neurons)
        self.dropout = nn.Dropout(p=0.0)
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.batch2 = nn.BatchNorm1d(num_neurons)
        self.layer3 = nn.Linear(num_neurons, output_size)
        self.batch3 = nn.BatchNorm1d(num_neurons)
        self.layer4 = nn.Linear(num_neurons, output_size)
        self.f_relu = nn.LeakyReLU()
        self.f = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.f_relu(x) 
        x = self.layer2(x)
        x = self.f_relu(x)
        x = self.layer3(x)
        x = self.f(x)
        return x
##


##

