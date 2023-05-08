import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.first_layer = nn.Conv2d(3, 64, 7, stride = 1, padding = int((7 - 1)/2))
        init.normal_(self.first_layer.weight, mean=0.0, std=0.02)
        self.first_norm = nn.BatchNorm2d(64)

        self.second_layer = nn.Conv2d(64, 128, 3, stride = 2, padding = int((3 - 1)/2), padding_mode = 'reflect')
        self.second_norm = nn.InstanceNorm2d(128)
        init.normal_(self.second_layer.weight, mean=0.0, std=0.02)

        self.third_layer = nn.Conv2d(128, 256, 3, stride = 2, padding = int((3 - 1)/2), padding_mode = 'reflect')
        self.third_norm = nn.InstanceNorm2d(256)
        init.normal_(self.third_layer.weight, mean=0.0, std=0.02)

        self.first_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.first_res_norm = nn.InstanceNorm2d(256)
        self.first_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.first_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.first_res.weight, mean=0.0, std=0.02)
        init.normal_(self.first_res_1.weight, mean=0.0, std=0.02)

        self.second_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.second_res_norm = nn.InstanceNorm2d(256)
        self.second_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.second_res_norm_1= nn.InstanceNorm2d(256)
        init.normal_(self.second_res.weight, mean=0.0, std=0.02)
        init.normal_(self.second_res_1.weight, mean=0.0, std=0.02)

        self.third_res = nn.Conv2d(256, 256, 3, stride=1, padding = int((3 - 1)/2))
        self.third_res_norm = nn.InstanceNorm2d(256)
        self.third_res_1 = nn.Conv2d(256, 256, 3, stride=1, padding = int((3 - 1)/2))
        self.third_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.third_res.weight, mean=0.0, std=0.02)
        init.normal_(self.third_res_1.weight, mean=0.0, std=0.02)

        self.fourth_res = nn.Conv2d(256, 256, 3, stride=1, padding = int((3 - 1)/2))
        self.fourth_res_norm = nn.InstanceNorm2d(256)
        self.fourth_res_1 = nn.Conv2d(256, 256, 3, stride=1, padding = int((3 - 1)/2))
        self.fourth_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.fourth_res.weight, mean=0.0, std=0.02)
        init.normal_(self.fourth_res_1.weight, mean=0.0, std=0.02)

        self.fifth_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.fifth_res_norm = nn.InstanceNorm2d(256)
        self.fifth_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.fifth_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.fifth_res.weight, mean=0.0, std=0.02)
        init.normal_(self.fifth_res_1.weight, mean=0.0, std=0.02)

        self.sixth_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.sixth_res_norm = nn.InstanceNorm2d(256)
        self.sixth_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.sixth_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.sixth_res.weight, mean=0.0, std=0.02)
        init.normal_(self.sixth_res_1.weight, mean=0.0, std=0.02)

        self.seventh_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.seventh_res_norm = nn.InstanceNorm2d(256)
        self.seventh_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.seventh_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.seventh_res.weight, mean=0.0, std=0.02)
        init.normal_(self.seventh_res_1.weight, mean=0.0, std=0.02)

        self.eighth_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.eighth_res_norm = nn.InstanceNorm2d(256)
        self.eighth_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.eighth_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.eighth_res.weight, mean=0.0, std=0.02)
        init.normal_(self.eighth_res_1.weight, mean=0.0, std=0.02)

        self.nineth_res = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.nineth_res_norm = nn.InstanceNorm2d(256)
        self.nineth_res_1 = nn.Conv2d(256, 256, 3, stride = 1, padding = int((3 - 1)/2))
        self.nineth_res_norm_1 = nn.InstanceNorm2d(256)
        init.normal_(self.nineth_res.weight, mean=0.0, std=0.02)
        init.normal_(self.nineth_res_1.weight, mean=0.0, std=0.02)

        self.first_upsample = nn.ConvTranspose2d(256, 128, 2, stride = 2, padding = int((2 - 1)/2))
        init.normal_(self.first_upsample.weight, mean=0.0, std=0.02)
        self.first_upsample_norm = nn.InstanceNorm2d(64)

        self.second_upsample = nn.ConvTranspose2d(128, 64, 2, stride = 2, padding = int((2 - 1)/2))
        init.normal_(self.second_upsample.weight, mean=0.0, std=0.02)
        self.second_upsample_norm = nn.InstanceNorm2d(64)

        self.last_conv = nn.Conv2d(64, 3, 7, stride = 1, padding = int((7 - 1)/2))
        init.normal_(self.last_conv.weight, mean=0.0, std=0.02)
        self.last_norm = nn.InstanceNorm2d(3)


    def forward(self, img):
        
        x = F.relu(self.first_norm(self.first_layer(img)))
        x = F.relu(self.second_norm(self.second_layer(x)))
        x = F.relu(self.third_norm(self.third_layer(x)))

        x_original = x.clone()
        x = self.first_res_norm_1(self.first_res_1(F.relu(self.first_res_norm(self.first_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.second_res_norm_1(self.second_res_1(F.relu(self.second_res_norm(self.second_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.third_res_norm_1(self.third_res_1(F.relu(self.third_res_norm(self.third_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.fourth_res_norm_1(self.fourth_res_1(F.relu(self.fourth_res_norm(self.fourth_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.fifth_res_norm_1(self.fifth_res_1(F.relu(self.fifth_res_norm(self.fifth_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.sixth_res_norm_1(self.sixth_res_1(F.relu(self.sixth_res_norm(self.sixth_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.seventh_res_norm_1(self.seventh_res_1(F.relu(self.seventh_res_norm(self.seventh_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.eighth_res_norm_1(self.eighth_res_1(F.relu(self.eighth_res_norm(self.eighth_res(x)))))
        x = x + x_original

        x_original = x.clone()
        x = self.nineth_res_norm_1(self.nineth_res_1(F.relu(self.nineth_res_norm(self.nineth_res(x)))))
        x = x + x_original
        
        x = F.relu(self.first_upsample_norm(self.first_upsample(x)))

        x = F.relu(self.second_upsample_norm(self.second_upsample(x)))

        x = F.relu(self.last_norm(self.last_conv(x)))

        return x


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.first_layer = nn.Conv2d(3, 64, 4, stride = 2, padding = 1)
        nn.init.normal_(self.first_layer, mean=0.0, std=0.02)

        self.second_layer = nn.Conv2d(64, 128, 4, stride = 2, padding = 1)
        self.second_instance = nn.InstanceNorm2d(128)
        nn.init.normal_(self.second_layer, mean=0.0, std=0.02)


        self.third_layer = nn.Conv2d(128, 256, 4, stride = 2, padding = 1)
        self.third_instance = nn.InstanceNorm2d(256)
        nn.init.normal_(self.third_layer, mean=0.0, std=0.02)


        self.fourth_layer = nn.Conv2d(256, 512, 4, stride = 2, padding = 1)
        self.fourth_instance = nn.InstanceNorm2d(512)
        nn.init.normal_(self.fourth_layer, mean=0.0, std=0.02)


        self.fifth_layer = nn.Conv2d(512, 1, 4, stride = 2, padding = 1)
        self.fifth_instance = nn.InstanceNorm2d(1)
        nn.init.normal_(self.fifth_layer, mean=0.0, std=0.02)

        self.sigmoid = nn.Sigmoid()


    def forward(self, img):
        x = self.first_layer(img)
        x = F.leaky_relu(self.second_instance(self.second_layer(x)), 0.2)
        x = F.leaky_relu(self.third_instance(self.third_layer(x), 0.2))
        x = F.leaky_relu(self.fourth_instance(self.fourth_layer(x)), 0.2)
        x = F.leaky_relu(self.fifth_instance(self.fifth_layer(x)), 0.2)

        return self.sigmoid(x)

