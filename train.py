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
from runtime_args import args

# Set model and device
device = get_device()
model = mobilenetv2()
model.to(device)
"""
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

# Load trainset and prepare the dataset for Detectron2
trainset = fo.load_dataset('Aerial_Maritime_trainset')
view_train = trainset.match_tags('train')
DatasetCatalog.register('train', lambda view = view_train: get_fiftyone_dicts(view))
MetadataCatalog.get('train')
metadata_train = MetadataCatalog.get('train')

# Load validationset and prepare the dataset for Detectron2
validset = fo.load_dataset('Aerial_Maritime_validset')
view_valid = trainset.match_tags('valid')
DatasetCatalog.register('valid', lambda view = view_valid: get_fiftyone_dicts(view))
MetadataCatalog.get('valid')
metadata_valid = MetadataCatalog.get('valid')

# Detectron configuration
# Load config from file and command-line arguments
cfg = get_cfg()
# Set all the configuration
cfg.merge_from_file()
cfg.DATASETS.TRAIN = ("train")
cfg.DATASETS.TEST = () # Maybe put here the validation set
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.WEIGHTS = ()
cfg.SOLVER.IMS_PER_BATCH =   # Number of training examples per batch utilized in one iteration
cfg.SOLVER.BASE_LR = args.learning_rate #example 0.00025
cfg.SOLVER.MAX_ITER = args.ephocs
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.TEST.DETECTIONS_PER_IMAGE = args.number_detections


# Training
"""
os.makedirs(cfg.OUTPUT_DIR, exist = True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
"""

# Use a Model
# When in training mode, all models are required to be used under an 
# EventStorage. 
# The traing statistic will be put into the storage
# EventStorage example
"""
with EventStorage() as storage:
    losses = model(inputs) # input is a list[dict]
"""
