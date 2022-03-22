# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.com
# Description: Class Discriminator for cGAN
# ******************************************************************************************
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f, f_relu):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.map4 = nn.Linear(hidden_size, hidden_size)
        self.map5 = nn.Linear(hidden_size, output_size)
        self.batch1 = nn.BatchNorm1d(hidden_size)
        self.batch2 = nn.BatchNorm1d(hidden_size)
        self.batch3 = nn.BatchNorm1d(hidden_size)
        self.batch4 = nn.BatchNorm1d(hidden_size)
        self.f = f
        self.f_relu = f_relu
        
    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)   # f_relu
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        x = self.f(x)
        return x