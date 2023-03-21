# Add Tags:
# train, valid and test

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
### TRAINSET ###
trainset = GetDataset().train_set
#trainset.tags.append('train')
#for sample in trainset.iter_samples(progress=True):
#    sample.tags.append('train')
#    sample.save()
#print (trainset)
#print (trainset.first())
#print (trainset.last())

### VALIDSET ###
validset = GetDataset().valid_set
#validset.tags.append('valid')
#for sample in validset.iter_samples(progress=True):
#    sample.tags.append('valid')
#    sample.save()
#validset.persistent = True
#view = validset.match_tags('valid')
#DatasetCatalog.register('valid', lambda view = view: get_fiftyone_dicts(view))
#MetadataCatalog.get('valid')
#metadata = MetadataCatalog.get('valid')
#fo.pprint(validset.stats(include_media=True))
#print (validset)
#print (validset.first())
#print (validset.last())

### TESTSET ###
testset = GetDataset().test_set
testset.tags.append('test')
for sample in testset.iter_samples(progress=True):
    sample.tags.append('test')
    sample.save()
testset.persistent = True
"""
view = testset.match_tags('test')
DatasetCatalog.register('test', lambda view = view: get_fiftyone_dicts(view))
MetadataCatalog.get('test')
metadata = MetadataCatalog.get('test')
fo.pprint(testset.stats(include_media=True))
### VISUALIZATION ###
# Visualize some samples to make sure everything is being loaded properly
dataset_dicts = get_fiftyone_dicts(view)
ids = [dd["image_id"] for dd in dataset_dicts]
view = testset.select(ids)
session = fo.launch_app(view)
"""
