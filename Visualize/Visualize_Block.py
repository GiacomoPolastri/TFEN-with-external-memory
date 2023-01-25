"""
export QT_QPA_PLATFORM=offscreen
"""
import os
import warnings
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import tqdm

from utils import get_device, load_model
from Visualize.utils import Layers, get_output_layer, to_processed, normalize
from Models.mobilenetv2 import mobilenetv2
from Data.classes import classes
from Data.dataset import LoadDataset
from runtime_args import args


warnings.filterwarnings("ignore", category=UserWarning) 

device = get_device()

model = mobilenetv2().to(device)

model = load_model(model, device, args.number_train)

test_loader = LoadDataset(args).testloader
dataiter = iter(test_loader)
images, labels = next(dataiter)

block_list = []

block_list.append(Layers(args, model).mnet_block)
block_list.append(Layers(args, model).cbam_block)
block_list.append(Layers(args, model).cam_block)
block_list.append(Layers(args, model).sam_block)
block_list.append(Layers(args, model).last_cnn_layer)

output_folder = './Outputs/Visualize'
if not os.path.exists(output_folder): os.mkdir(output_folder)

fig = plt.figure(figsize=(30,50))

for i, image in enumerate(images):
    plt.clf()
    image = image.unsqueeze(0).to(device)
    outputs, names = get_output_layer(block_list, image)
    
    processed = to_processed(outputs)
    
    fig.add_subplot(3, 2, 6)
    input_image = image.squeeze(0).permute(1,2,0).cpu().numpy()
    input_image = np.clip(input_image, 0, 1)
    plt.imshow(input_image)
    plt.axis('off')
    
    for j, process in enumerate(processed):
        
        process = cv2.resize(process,(224,224))
        fig.add_subplot(3, 2, 1+j)
        plt.imshow(process)   
        plt.title(names[j], fontsize=40)
        plt.axis('off')
    
    output_single_image = output_folder + "/" + str(i)
    if not os.path.exists(output_single_image): os.mkdir(output_single_image)
    plt.savefig( output_single_image + "/Block_Visualize" + str(i) + ".jpg")
        

    
    