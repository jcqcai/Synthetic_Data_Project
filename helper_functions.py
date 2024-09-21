import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import time
import random

def get_datasets():
    transform = transforms.ToTensor()
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def get_trainloader(trainset, batch_size):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader

def get_testloader(testset, batch_size):
    testloader = DataLoader(testset, shuffle=False)
    return testloader

def get_dataloaders(trainset, testset, batch_size):
    trainloader = get_trainloader(trainset, batch_size)
    testloader = get_testloader(testset, batch_size)
    return trainloader, testloader

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_batch(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainiter = iter(loader)
    xs, ys = next(trainiter)
    imshow(make_grid(xs).cpu())
    print(ys)

