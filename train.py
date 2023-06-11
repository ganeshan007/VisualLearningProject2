from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import sklearn.datasets
import time
from PIL import ImageFile
from PIL import Image
from data_utils import VideoFolder, SubsetRandomSampler
from torch.utils.data import DataLoader
from models import CustomGAN_Generator, CustomGAN_Discriminator
ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--cuda', type=bool, default=False, help='cuda enabled')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--outputDir', required=True, default='./results', help='path to output folder')

args = parser.parse_args()
print('Making output folder... \n')
if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
    print('Done!')

imageSize = 128
train_set = VideoFolder(root=args.dataroot,
		nframes=32,
		transform=transforms.Compose([
					transforms.Resize( (imageSize, imageSize) ),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5),
						(0.5, 0.5, 0.5)),
						]))

print('training data size: ' + str (len(train_set) ) )
train_dataloader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True, pin_memory=True, drop_last=True)


criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.

print('Initializing Models... \n')
net_G = CustomGAN_Generator(32)
net_D = CustomGAN_Discriminator(32)

device = torch.device("cpu")
if args.cuda:
    print('Using GPU...')
    device = torch.device("cuda")
net_G.to(device)
net_D.to(device)


optimizerG = optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))

print('Starting Training... \n')

for epoch in range(1, args.epochs + 1):
    for i, data in enumerate(train_dataloader, 0):
        net_D.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        real_data = real_data[:,:,0,:,:]
        real_data = real_data.unsqueeze(2).repeat(1,1,real_data.size(2),1,1 )
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
        output = net_D(real_data)[0]
        err_D_real = criterion(output, label)  
        print(err_D_real.item())
        err_D_real.backward()

        noise  = torch.randn((batch_size,3,32,imageSize,imageSize), device=device)
        fake = net_G(noise)
        label.fill_(fake_label)
        output = net_D(fake.detach())
        output = output.data.permute(2, 0, 1, 3, 4)
        print(output.shape)










        noise = torch.randn(batch_size, 100, 1, 1, device=device)



