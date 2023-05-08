import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.first_layer = torch.nn.Conv2d(3, 64, 4, stride = 2, padding = 1)
        nn.init.normal_(self.first_layer.weight, mean = 0.00, std = 0.02)
        
        self.second_layer = torch.nn.Conv2d(64, 128, 4, stride = 2, padding = 1)
        nn.init.normal_(self.second_layer.weight, mean = 0.00, std = 0.02)
        self.batch2 = torch.nn.BatchNorm2d(128, eps=1e-10)
        
        self.third_layer = torch.nn.Conv2d(128, 256, 4, stride = 2, padding = 1)
        nn.init.normal_(self.third_layer.weight, mean = 0.00, std = 0.02)
        self.batch3 = torch.nn.BatchNorm2d(256, eps=1e-10)
        
        self.fourth_layer = torch.nn.Conv2d(256, 512, 4, stride = 1, padding = 1)
        nn.init.normal_(self.fourth_layer.weight, mean = 0.00, std = 0.02)
        self.batch4 = torch.nn.BatchNorm2d(512, eps=1e-10)
        
        self.fifth_layer = torch.nn.Conv2d(512, 1, 4, stride = 1, padding = 1)
        nn.init.normal_(self.fifth_layer.weight, mean = 0.00, std = 0.02)
        self.batch5 = torch.nn.BatchNorm2d(1, eps=1e-10)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, real, gen):

        img = torch.cat((real, gen), 1)

        first = F.leaky_relu(self.first_layer(img), 0.2)
        second = F.leaky_relu(self.batch2(self.second_layer(first)), 0.2)
        third = F.leaky_relu(self.batch3(self.third_layer(second)), 0.2)
        fourth = F.relu(self.batch4(self.fourth_layer(third)), 0.2)
        fifth = F.leaky_relu(self.batch5(self.fifth_layer(fourth)), 0.2)
        output = self.sigmoid(output)

        return output