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
from Visualize.utils import Layers
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

output_folder = './Outputs/Visualize'
if not os.path.exists(output_folder): os.mkdir(output_folder)

fig = plt.figure(figsize=(30,50))

for i, image in enumerate(images):
    
    plt.clf()
    
    image = image.unsqueeze(0).to(device)
    
    cnn_filter, output = model(image)
    
    _ , pred = torch.max(output, 1)
    
    combined_filter = torch.max(cnn_filter.squeeze(0), 0)[0].detach().cpu().numpy()
    
    cv2_combined_filter = cv2.resize(combined_filter, (224,224))
    
    heatmap = np.asarray(
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
    
    cnn_filter = cnn_filter.squeeze(0)
    out = torch.zeros((7,7)).to(device)
    for filter in cnn_filter:
        out = torch.add(out, filter)
    print (out.size())
    out = out.detach().cpu().numpy()
    
    cv2_out = cv2.resize(out, (224,224))
    
    input_image = image.squeeze(0).permute(1,2,0).cpu().numpy()
    input_image = np.clip(input_image, 0, 1)
    
    heatmap_cnn = cv2.addWeighted(
        np.asarray(
            input_image, dtype=np.float32), 
            0.9, 
            heatmap, 
            0.0025, 
            0)
    
    fig.add_subplot(2,2,1)
    plt.imshow(input_image)
    plt.title(classes[pred])
    plt.axis('off')
    
    fig.add_subplot(2,2,2)
    plt.imshow(heatmap_cnn)
    plt.title('Heat Map')
    plt.axis('off')
    
    fig.add_subplot(2,2,3)
    plt.imshow(cv2_out)
    plt.title('cv2_cnn_filter')
    plt.axis('off')
    
    fig.add_subplot(2,2,4)
    plt.imshow(cv2_combined_filter)
    plt.title('cv2_combined_filter')
    plt.axis('off')
    
    output_single_image = output_folder + "/" + str(i)
    if not os.path.exists(output_single_image): os.mkdir(output_single_image)
    plt.savefig( output_single_image + "/Output" + str(i) + ".jpg")
    

    
    