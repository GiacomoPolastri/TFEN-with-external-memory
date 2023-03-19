import logging
import fiftyone as fo 
from fiftyone import ViewField as F

# Import PyTorch and Detectron2
import torch, detectron2
from detectron2.engine.train_loop import TrainerBase
from detectron2.utils import registry
"""TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
"""
# Setup detectron2 logger
import detectron2
from detectron2.engine.defaults import DefaultTrainer
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
from detectron2.modeling import build_model, build_backbone

from Models.utils import get_number_model
from Data.AMdataset import GetDataset
from utils import get_device, model_to_device
from Data.utils import get_fiftyone_dicts
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from runtime_args import args
from Models.mobilenetv2 import build_mnv2_backbone
#from Models.fujiMNV2 import build_mnv2_backbone
# Load trainset and prepare the dataset for Detectron2
trainset = fo.load_dataset('trainset')
view_train = trainset.match_tags('train')
DatasetCatalog.register('train', lambda view = view_train: get_fiftyone_dicts(view_train))
MetadataCatalog.get('train')
metadata_train = MetadataCatalog.get('train')

# Load validationset and prepare the dataset for Detectron2
validset = fo.load_dataset('validset')
view_valid = validset.match_tags('valid')
DatasetCatalog.register('valid', lambda view = view_valid: get_fiftyone_dicts(view_valid))
MetadataCatalog.get('valid')
metadata_valid = MetadataCatalog.get('valid')
 
# Detectron configuration

# Load config from file and command-line arguments
cfg = get_cfg()
# Set all the configuration
cfg.DATASETS.TRAIN = ("train","valid")
cfg.DATASETS.TEST = () # Maybe put here the validation set
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2  # Number of training examples per batch utilized in one iteration
cfg.SOLVER.BASE_LR = 4e-5 #args.learning_rate #example 0.00025
cfg.SOLVER.MAX_ITER = 300 #args.epochs
cfg.SOLVER.STEPS = []
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Add backbone
cfg.MODEL.BACKBONE.NAME = "build_mnv2_backbone"
# Add ROI Head


"""
Add config for ResNeSt.
"""
# Apply deep stem 
cfg.MODEL.RESNETS.DEEP_STEM = False
# Apply avg after conv2 in the BottleBlock
# When AVD=True, the STRIDE_IN_1X1 should be False
cfg.MODEL.RESNETS.AVD = False
# Apply avg_down to the downsampling layer for residual path 
cfg.MODEL.RESNETS.AVG_DOWN = False
# Radix in ResNeSt setting RADIX: 2
cfg.MODEL.RESNETS.RADIX = 2
# Bottleneck_width in ResNeSt
cfg.MODEL.RESNETS.BOTTLENECK_WIDTH = 64

cfg.MODEL.FPN.REPEAT = 2


device = get_device()
model = build_model(cfg)
print (cfg)


model = model_to_device(model,device)
#print (model.eval())
# Training

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
#trainer = model_to_device(trainer, device)
#trainer.resume_or_load(resume=False)
trainer.train()