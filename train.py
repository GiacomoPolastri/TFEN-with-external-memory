

import torch
import torchvision
import fiftyone as fo 

from Models.mobilenetv2 import mobilenetv2
from Data.AMdataset import LoaDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = mobilenetv2
model.to(device)

