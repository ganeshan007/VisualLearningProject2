import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class CustomGAN_Generator(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.conv1 = nn.Conv3d(3, ngf,(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1),bias=False)
        self.conv2 = nn.Conv3d(ngf, ngf *2 ,(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),bias=False)
        self.conv3 = nn.Conv3d(ngf *2, ngf * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv3d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv3d(ngf * 8, ngf * 16, 4, 2, 1, bias=False)
        self.conv6 = nn.Conv3d(ngf * 16, ngf * 16, (2, 4, 4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        
        self.BN2 = nn.BatchNorm3d(ngf * 2)
        self.BN3 = nn.BatchNorm3d(ngf * 4)
        self.BN4 = nn.BatchNorm3d(ngf * 8)
        self.BN5 = nn.BatchNorm3d(ngf * 16)
        self.relu = nn.ReLU(inplace = True)

        self.up1 = nn.ConvTranspose3d(ngf * 16, ngf * 16, (2,4,4), stride=(1, 1, 1), padding=(0, 0, 0), bias=False )
        self.up2 = nn.ConvTranspose3d(ngf * 16, ngf * 8, 4, 2, 1, bias=False)
        self.up3 = nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
        self.up4 = nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
        self.up5 = nn.ConvTranspose3d(ngf * 2, ngf * 1, (4,4,4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
        self.up6 = nn.ConvTranspose3d(ngf * 1, 3, (3,4,4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)
        self.tanh = nn.Tanh()
        self.upBN1 = nn.BatchNorm3d(ngf * 16)
        self.upBN2 = nn.BatchNorm3d(ngf * 8)
        self.upBN3 = nn.BatchNorm3d(ngf * 4)
        self.upBN4 = nn.BatchNorm3d(ngf * 2)
        self.upBN5 = nn.BatchNorm3d(ngf * 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.lrelu(self.BN2(self.conv2(x)))
        x3 = self.lrelu(self.BN3(self.conv3(x2)))
        x4 = self.lrelu(self.BN4(self.conv4(x3)))
        x5 = self.lrelu(self.BN5(self.conv5(x4)))
        x6 = self.conv6(x5)

        up1 = self.relu(self.upBN1(self.up1(x6)))
        up1 = x5 + up1

        up2 = self.relu(self.upBN2(self.up2(up1)))
        up2 += x4

        up3 = self.relu(self.upBN3(self.up3(up2)))
        up3 += x3

        up4 = self.relu(self.upBN4(self.up4(up3)))
        up4 += x2
        
        up5 = self.relu(self.upBN5(self.up5(up4)))
        up5 += x
        
        out = self.tanh(self.up6(up5))
        return out
    

class CustomGAN_Discriminator(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        self.conv1 = nn.Conv3d(3, ndf, (3,4,4), stride=(1,2,2), padding=1, bias=False)
        self.conv2 = nn.Conv3d(ndf, ndf*2, (4,4,4), stride=(2,2,2), padding=1, bias=False)
        self.conv3 = nn.Conv3d(ndf*2, ndf*4, (4,4,4), stride=(2,2,2), padding=1, bias=False)
        self.conv4 = nn.Conv3d(ndf*4, ndf*8, (4,4,4), stride=(2,2,2), padding=1, bias=False)
        self.conv5 = nn.Conv3d(ndf*8, ndf*16, (4,4,4), stride=(2,2,2), padding=1, bias=False)
        # self.conv6 = nn.Conv3d(ndf*16, 1, (2,4,4), stride=(1,1,1), padding=0, bias=False)
        self.conv6 = nn.Conv3d(ndf*16, 1, (8,4,4), stride=(1,1,1), padding=0, bias=False)


        self.BN2 = nn.BatchNorm3d(ndf*2)
        self.BN3 = nn.BatchNorm3d(ndf*4)
        self.BN4 = nn.BatchNorm3d(ndf*8)
        self.BN5 = nn.BatchNorm3d(ndf*16)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        
        x2 = self.lrelu(self.BN2(self.conv2(x1)))
        x2 = self.lrelu(self.BN3(self.conv3(x2)))

        x3 = self.lrelu(self.BN4(self.conv4(x2)))
        x3 = self.lrelu(self.BN5(self.conv5(x3)))

        out = self.sigmoid(self.conv6(x3))
        # return out.view(-1,1), [x2,x1]
        return out.view(-1), [x2,x1]


        