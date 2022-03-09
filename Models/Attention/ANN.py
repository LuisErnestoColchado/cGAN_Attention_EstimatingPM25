# **********************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe
# Description: Aritificial Neural Network class
# **********************************************************************************
##

import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.batch1 = nn.BatchNorm1d(32)
        self.layer2 = nn.Linear(32, 32)
        self.batch2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, 32)
        self.batch3 = nn.BatchNorm1d(32)
        self.layer4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.LeakyReLU()(x)
        x = self.layer2(x)
        x = self.batch2(x)
        x = nn.LeakyReLU()(x)
        x = self.layer3(x)
        x = nn.LeakyReLU()(x)
        x = self.layer4(x)
        x = nn.functional.sigmoid(x)
        return x
