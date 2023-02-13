
import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch
from torch.nn import Threshold

from Visualize.utils import Layers, get_output_layer, to_processed
from utils import get_device, load_model
from runtime_args import args
from Models.mobilenetv2 import mobilenetv2
from Data.OwnDataset import LoadDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Get device and model, load the model
device = get_device()
model = mobilenetv2().to(device)
model = load_model(model, device, args.number_train)

# Get the test set
test_loader = LoadDataset(args).testloader
dataiter = iter(test_loader)
images, labels = next(dataiter)

output_folder = './Outputs/Visualize'
if not os.path.exists(output_folder): os.mkdir(output_folder)

mnet_block = Layers(args, model).mnet_block
cam_block = Layers(args, model).cam_block
sam_cnn_layer = Layers(args, model).sam_cnn_layer

fig = plt.figure(figsize=(30,50))

cropped_feature_map = []

for i, image in enumerate(images):
    
    image = image.unsqueeze(0).to(device)
    
    x_conv, output = model(image)
    
    _ , pred = torch.max(output, 1)
    
    max_x_conv = torch.max(x_conv.squeeze(0), 0)[0]    
    
    max_x_conv -= torch.min(max_x_conv)
    max_x_conv /= torch.max(max_x_conv)

    t = Threshold(0.5, 0)
    threshold = t(max_x_conv)

    value_to_save = []
    i = 0
    for array in (max_x_conv):
        #print (i)
        #print (array)
        for x in (array):    
            if x >= 0.5: value_to_save.append([i, x])
            i = i + 1
    
    cropped_feature_map.append(value_to_save)
    #print (value_to_save)
    #print (len(value_to_save))
    #number, values = value_to_save[0]
    #print (number)
print (len(cropped_feature_map))