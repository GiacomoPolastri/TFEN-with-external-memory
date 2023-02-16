import fiftyone as fo 
from fiftyone import ViewField as F

# Import PyTorch and Detectron2
import torch, detectron2

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from Models.mobilenetv2 import mobilenetv2
from Data.AMdataset import GetDataset
from utils import get_device
from Data.utils import get_fiftyone_dicts

device = get_device()
model = mobilenetv2()
model.to(device)

trainset = GetDataset().train_set

print(trainset)

session = fo.launch_app(trainset, desktop=True)
session.wait()



#print (trainset.match_tags)
#DatasetCatalog.register("trainset", get_fiftyone_dicts(trainset))
