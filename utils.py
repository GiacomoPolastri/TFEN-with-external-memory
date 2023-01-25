from collections import OrderedDict
import os
import sys
import time
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

from Models.mobilenetv2 import mobilenetv2

def swap(img):
    img = img.swapaxes(0, 1)
    img = img.swapaxes(1, 2)

    return img

def rev_swap(img):
    img = img.swapaxes(2, 1)
    img = img.swapaxes(1, 0)

    return img

def get_image(set):
    # Posso pensare di prendere piu esempi e fare un output con diversi esempi e dei cicli
    img, label = [], []
    example = torch.randint(len(set), size=(1,)).item()
    img, label = set[example]

    return img, label, example

def img_to_device(img, device):
    #img = transform(img)
    #img = img.unsqueeze(0)
    img = img.to(device)

    return img

def get_device(gpu_usage=True, eval=True):
    
    device = torch.device('cuda' if (torch.cuda.is_available() & gpu_usage == True) else 'cpu')
    use_multi_gpu = torch.cuda.device_count() > 1
    if eval == True:
        print('[Info] device:{} use_multi_gpu:{}'.format(device, use_multi_gpu))

    return device

def model_to_device(model, device):
    print ('put model into GPUs')
    if torch.cuda.device_count() > 1:
        print ("Let's use ", str(torch.cuda.device_count()), " GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    
    return model

def load_model (model, device, n_model):
    
    try:
        state_model = torch.load('./Train/' + str(n_model) + '.pth', map_location=device)
        state_dict = state_model['model_state_dict']
        
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print('Model loaded')
    
    except Exception as e:
        print(e)
    
    model.eval()
    
    return model
    
        