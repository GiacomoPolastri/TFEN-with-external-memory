# import FiftyOne
import fiftyone as fo
import fiftyone.zoo as foz

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
from Data.AMdataset import GetDataset
from fiftyone import ViewField as F
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from Data.utils import get_fiftyone_dicts

# Load dataset and prepare the dataset for Detectron2
trainset = fo.load_dataset('Aerial_Maritime_trainset')
view = trainset.match_tags('train')
DatasetCatalog.register('train', lambda view = view: get_fiftyone_dicts(view))
MetadataCatalog.get('train')
metadata = MetadataCatalog.get('train')

print (metadata)

# Visualize some samples to make sure everything is being loaded properly
"""
dataset_dicts = get_fiftyone_dicts(view)
ids = [dd["image_id"] for dd in dataset_dicts]

view = trainset.select(ids)
session = fo.launch_app(view)
session.wait()
"""

