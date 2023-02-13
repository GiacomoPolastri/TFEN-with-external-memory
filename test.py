
import os
import warnings
from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch

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

for i, image in enumerate(images):
    
    image = image.unsqueeze(0).to(device)
    
    cnn_filter, output = model(image)
    
    _ , pred = torch.max(output, 1)
    
    combined_filter = torch.max(cnn_filter.squeeze(0), 0)[0].detach().cpu().numpy()
    
    cv2_combined_filter = cv2.resize(combined_filter, (224,224))
    
    mnet = mnet_block(image)
    cam = cam_block(mnet)
    sam, names = get_output_layer(sam_cnn_layer, cam)
    
    processed = to_processed(sam)
    process = cv2.resize(processed[1], (224,224))
    
    heatmap_x_conv = np.asarray(
        cv2.applyColorMap(
            cv2.normalize(
                cv2_combined_filter, 
                None, 
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX, 
                dtype=cv2.CV_8U),
                cv2.COLORMAP_JET), 
                dtype=np.float32)
    
    heatmap = np.asarray(
    cv2.applyColorMap(
        cv2.normalize(
            process, 
            None, 
            alpha=0, 
            beta=255, 
            norm_type=cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U),
            cv2.COLORMAP_JET), 
            dtype=np.float32)
    
    input_image = image.squeeze(0).permute(1,2,0).cpu().numpy()
    input_image = np.clip(input_image, 0, 1)
    
    heatmap_cnn_x_conv = cv2.addWeighted(
        np.asarray(
            input_image, dtype=np.float32), 
            0.9, 
            heatmap_x_conv, 
            0.0025, 
            0)
    heatmap_cnn_x_conv = np.clip(heatmap_cnn_x_conv, 0,1)
    
    heatmap_cnn = cv2.addWeighted(
    np.asarray(
        input_image, dtype=np.float32), 
        0.9, 
        heatmap, 
        0.0025, 
        0)
    heatmap_cnn = np.clip(heatmap_cnn, 0,1)
    
    fig.add_subplot(2,2,1)
    plt.imshow(process)
    plt.title(names[1], fontsize=40)
    plt.axis('off')
    
    fig.add_subplot(2,2,2)
    plt.imshow(heatmap_cnn)
    plt.title('Heat Map', fontsize=40)
    plt.axis('off')
    
    fig.add_subplot(2,2,3)
    plt.imshow(cv2_combined_filter)
    plt.title('x_conv', fontsize=40)
    plt.axis('off')
    
    fig.add_subplot(2,2,4)
    plt.imshow(heatmap_cnn_x_conv)
    plt.title('Heat Map', fontsize=40)
    plt.axis('off')    

    output_single_image = output_folder + "/" + str(i)
    if not os.path.exists(output_single_image): os.mkdir(output_single_image)
    plt.savefig( output_single_image + "/FinalTest" + str(i) + ".jpg")