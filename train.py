from __future__ import print_function
import argparse
import os
import wandb
import random
import torch
import numpy as np
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
from model_utils import CheckpointSaver
from models import CustomGAN_Generator, CustomGAN_Discriminator
ImageFile.LOAD_TRUNCATED_IMAGES = True
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--cuda', type=bool, default=False, help='cuda enabled')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--printEvery', type=int, default=50, help='number of steps to print after')
parser.add_argument('--outputDir', required=True, default='./results', help='path to output folder')

args = parser.parse_args()
print('Making output folder... \n')
if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)
    print('Done!')

imageSize = 128
num_frames = 32
train_set = VideoFolder(root=args.dataroot,
		nframes=num_frames,
		transform=transforms.Compose([
					transforms.Resize( (imageSize, imageSize) ),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5),
						(0.5, 0.5, 0.5)),
						]))

print('training data size: ' + str (len(train_set) ) )
train_dataloader = DataLoader(train_set, batch_size=args.batchSize, shuffle=True, pin_memory=True, drop_last=True, num_workers=5)


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
torch.autograd.set_detect_anomaly(True)

print('Initializing Logging at WandB... \n')
run = wandb.init(
    project="VisualLearningProject",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    })
checkpoint_saver = CheckpointSaver(dirpath=args.outputDir, decreasing=True, top_n=5)

G_losses = []
D_losses = []

for epoch in range(1, args.epochs + 1):
    for i, data in enumerate(train_dataloader, 0):
        net_D.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        real_data = real_data[:,:,0,:,:]
        real_data = real_data.unsqueeze(2).repeat(1,1,num_frames,1,1 )
        label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
        output = net_D(real_data)[0]
        errD_real = criterion(output, label)  
        errD_real.backward()
        
        noise  = torch.randn((args.batchSize,3,num_frames,imageSize,imageSize), device=device)
        fake = net_G(noise)
        label.fill_(fake_label)
        output = net_D(fake.detach())[0]
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        net_G.zero_grad()
        label.fill_(real_label)
        output = net_D(fake)[0]
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        if (i%args.printEvery==0):
            print(f'Epoch {epoch} Step {i} \t Generator Loss is {errG} \t Discriminator Loss is {errD}')
        wandb.log({'Train Generator Loss':errG.item(), 'Training Discriminator Loss': errD.item()})
        G_losses.append(errG.item())
        D_losses.append(errD.item())
    checkpoint_saver(net_G, net_D, epoch, np.mean(G_losses))



