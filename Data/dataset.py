"""

Data loader of the dataset in the inputs folder.

"""

import glob
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as t

from Data.transforms import *
from runtime_args import args

Input_root = './Inputs/'
Train_root = Input_root +'Train'
Val_root = Input_root +'Val'
Test_root = Input_root +'Test'

class LoadDataset(Dataset):
    
    def __init__(self, args):
        
        self.size = args.input_size
        self.da = args.data_augmentation
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.trainset = self.train_set()
        self.valset = self.val_set()
        self.testset = self.test_set()
        self.trainloader = self.train_loader()
        self.validloader = self.val_loader()
        self.testloader = self.test_loader()
    
    def train_set(self):
        
        trainset = ImageFolder(
            root=Train_root,
            transform=train_transform(self.size, self.da),
        )
    
        return trainset
    
    def train_loader(self):
        
        trainloader = DataLoader(
            self.trainset,
            batch_size= self.batch_size,
            shuffle= True,
            num_workers= self.num_workers,
            pin_memory= True,
        )
        
        return trainloader
        
    def val_set(self):
        
        valset = ImageFolder(
            root=Val_root,
            transform=val_transform(self.size),
        )
        
        return valset
    
    def val_loader(self):
        
        validloader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return validloader
    
    def test_set(self):
        
        testset = ImageFolder(
            root=Test_root,
            transform=test_transform(self.size),
        )
        
        return testset
    
    def test_loader(self):
        
        testloader = DataLoader(
            self.testset,
            batch_size=150,
            shuffle=False,
            num_workers=1,
        )
        
        return testloader
        
    
    
    
    
        
      
        
    
    