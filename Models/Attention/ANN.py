# **********************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: MLP class
# **********************************************************************************
##

import torch.nn as nn
##


class ANN(nn.Module):
    def __init__(self, input_size, num_neurons, output_size):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(input_size, num_neurons)
        self.batch1 = nn.BatchNorm1d(num_neurons)
        self.layer2 = nn.Linear(num_neurons, num_neurons)
        self.batch2 = nn.BatchNorm1d(num_neurons)
        self.layer3 = nn.Linear(num_neurons, output_size)
        #self.batch3 = nn.BatchNorm1d(num_neurons)
        #self.layer4 = nn.Linear(num_neurons, output_size)
        self.f_relu = nn.LeakyReLU()#negative_slope=0.001)
    def forward(self, x):
        x = self.layer1(x)
        #x = self.batch1(x)
        #!x = nn.functional.sigmoid(x)
        x = self.f_relu(x)
        x = self.layer2(x)
        #x = self.batch2(x)
        x = self.f_relu(x)
        #!x = nn.functional.sigmoid(x)
        x = self.layer3(x)
        #x = self.batch3(x)
        x = self.f_relu(x)
        #x = self.layer4(x)
        #x = nn.functional.sigmoid(x)
        return x
##
