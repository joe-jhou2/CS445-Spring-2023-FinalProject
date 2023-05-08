import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()

        self.first_layer = torch.nn.Conv2d(3, 64, 4, stride = 2, padding = 1)
        nn.init.normal_(self.first_layer.weight, mean = 0.00, std = 0.02)
        self.batch1 = torch.nn.BatchNorm2d(64, eps=1e-10)

        self.second_layer = torch.nn.Conv2d(64, 128, 4, stride = 2, padding = 1)
        nn.init.normal_(self.second_layer.weight, mean = 0.00, std = 0.02)
        self.batch2 = torch.nn.BatchNorm2d(128, eps=1e-10)

        self.third_layer = torch.nn.Conv2d(128, 256, 4, stride = 2, padding = 1)
        nn.init.normal_(self.third_layer.weight, mean = 0.00, std = 0.02)
        self.batch3 = torch.nn.BatchNorm2d(256, eps=1e-10)

        self.fourth_layer = torch.nn.Conv2d(256, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.fourth_layer.weight, mean = 0.00, std = 0.02)
        self.batch4 = torch.nn.BatchNorm2d(512, eps=1e-10)

        self.fifth_layer = torch.nn.Conv2d(512, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.fifth_layer.weight, mean = 0.00, std = 0.02)
        self.batch5 = torch.nn.BatchNorm2d(512, eps=1e-10)

        self.sixth_layer = torch.nn.Conv2d(512, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.sixth_layer.weight, mean = 0.00, std = 0.02)
        self.batch6 = torch.nn.BatchNorm2d(512, eps=1e-10)

        self.seventh_layer = torch.nn.Conv2d(512, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.seventh_layer.weight, mean = 0.00, std = 0.02)
        self.batch7 = torch.nn.BatchNorm2d(512, eps=1e-10)

        self.eighth_layer = torch.nn.Conv2d(512, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.eighth_layer.weight, mean = 0.00, std = 0.02)
        self.batch8 = torch.nn.BatchNorm2d(512, eps=1e-10)


        self.first_d_layer = torch.nn.ConvTranspose2d(512, 1024, 4, stride = 2, padding = 1)
        nn.init.normal_(self.first_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch1_d = torch.nn.BatchNorm2d(1024)

        self.second_d_layer = torch.nn.ConvTranspose2d(1024, 1024, 4, stride = 2, padding = 1)
        nn.init.normal_(self.second_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch2_d = torch.nn.BatchNorm2d(1024)

        self.third_d_layer = torch.nn.ConvTranspose2d(1024, 1024, 4, stride = 2, padding = 1)
        nn.init.normal_(self.third_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch3_d = torch.nn.BatchNorm2d(1024)

        self.fourth_d_layer = torch.nn.Conv2d(1024, 1024, 4, stride = 2, padding = 1)
        nn.init.normal_(self.fourth_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch4_d = torch.nn.BatchNorm2d(1024)

        self.fifth_d_layer = torch.nn.Conv2d(1024, 512, 4, stride = 2, padding = 1)
        nn.init.normal_(self.fifth_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch5_d = torch.nn.BatchNorm2d(512)

        self.sixth_d_layer = torch.nn.Conv2d(512, 256, 4, stride = 2, padding = 1)
        nn.init.normal_(self.sixth_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch6_d = torch.nn.BatchNorm2d(256)

        self.seventh_d_layer = torch.nn.Conv2d(256, 128, 4, stride = 2, padding = 1)
        nn.init.normal_(self.seventh_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch7_d = torch.nn.BatchNorm2d(128)

        self.eighth_d_layer = torch.nn.Conv2d(128, 3, 4, stride = 2, padding = 1)
        nn.init.normal_(self.eighth_d_layer.weight, mean = 0.00, std = 0.02)
        self.batch8_d = torch.nn.BatchNorm2d(3)
        self.tanh = torch.nn.Tanh()

    def forward(self, img):

        x_1 = F.leaky_relu(self.batch1_d(self.first_layer(img)), 0.2)
        x_2 = F.leaky_relu(self.batch2_d(self.second_layer(x)), 0.2)
        x_3 = F.leaky_relu(self.batch3_d(self.third_layer(x)), 0.2)
        x_4 = F.leaky_relu(self.batch4_d(self.fourth_layer(x)), 0.2)
        x_5 = F.leaky_relu(self.batch5_d(self.fifth_layer(x)), 0.2)
        x_6 = F.leaky_relu(self.batch6_d(self.sixth_layer(x)), 0.2)
        x_7 = F.leaky_relu(self.batch7_d(self.seventh_layer(x)), 0.2)
        x_8 = F.leaky_relu(self.batch8_d(self.eigth_layer(x)), 0.2)

        out = F.dropout(F.relu(self.batch1_d(self.first_d_layer(x_8))), 0.5, training = True)
        out = F.dropout(F.relu(self.batch2_d(self.second_d_layer(torch.cat( (out, x_7), 1)))), 0.5, training = True)
        out = F.dropout(F.relu(self.batch3_d(self.third_d_layer(torch.cat( (out, x_6), 1)))), 0.5, training = True)
        out = F.relu(self.batch4_d(self.fourth_d_layer(torch.cat( (out, x_5), 1))))
        out = F.relu(self.batch5_d(self.fifth_d_layer(torch.cat( (out, x_4), 1))))
        out = F.relu(self.batch6_d(self.sixth_d_layer(torch.cat( (out, x_3), 1))))
        out = F.relu(self.batch7_d(self.seventh_d_layer(torch.cat( (out, x_2), 1))))
        out = F.relu(self.batch8_d(self.eighth_d_layer(torch.cat( (out, x_1), 1))))
        output = self.tanh(out)

        return output
                                                            
                                                            
                                     