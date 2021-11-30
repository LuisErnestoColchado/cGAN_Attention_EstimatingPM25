import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f, f_relu):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f
        self.f_relu = f_relu

    def forward(self, x):
        x = self.map1(x)
        x = self.f_relu(x)
        x = self.map2(x)
        x = self.f_relu(x)
        x = self.map3(x)
        x = self.f_relu(x)
        return x