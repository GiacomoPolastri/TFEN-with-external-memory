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
from detectron2.utils.events import EventStorage
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model

from Models.mobilenetv2 import mobilenetv2
from Models.utils import get_number_model
from Data.AMdataset import GetDataset
from utils import get_device
from Data.utils import get_fiftyone_dicts
from Checkpoint.detection_checkpoint import DetectionCheckpointer

# Set model and device
"""
device = get_device()
model = mobilenetv2()
model.to(device)
# Use Model
# 
# From a yacs config object, 
# models can be built by functions: 
# build_model, build_backbone, build_roi_heads
# 
# build model: only builds the model structure 
# and fills it with random parameters. 
model = build_model(cfg)
"""

# Load / Save a Checkpoint
"""
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) # file with weights saved
numberOfCheckpoint = get_number_model()
checkpointer = DetectionCheckpointer(model, save_dir = "Train")
checkpointer.save("training_number_" + str(numberOfCheckpoint))
"""

# Load dataset and prepare the dataset for Detectron2
trainset = fo.load_dataset('Aerial_Maritime_trainset')
view = trainset.match_tags('train')
DatasetCatalog.register('train', lambda view = view: get_fiftyone_dicts(view))
MetadataCatalog.get('train')
metadata = MetadataCatalog.get('train')

# Detectron configuration

# Training

# Use a Model
# When in training mode, all models are required to be used under an 
# EventStorage. 
# The traing statistic will be put into the storage
# EventStorage example
"""
with EventStorage() as storage:
    losses = model(inputs) # input is a list[dict]
"""
