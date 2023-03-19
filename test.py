import torch
import fiftyone as fo 
import detectron2
import detectron2_backbone
### DATASET ###
from detectron2.data import MetadataCatalog, DatasetCatalog
from Data.utils import get_fiftyone_dicts
### D2 CONFIGURATION ###
from detectron2.config.config import get_cfg
### BACKBONE ###
#from Models.mobilenetv2 import build_mnv2_backbone
from Models.prova import build_mnv2_CBAM_backbone
from detectron2_backbone.config import add_backbone_config
### MODEL ###
from detectron2.modeling import build_model
import os
from utils import get_device, model_to_device
from detectron2.engine.defaults import DefaultTrainer

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

### Detectron configuration ###
cfg = get_cfg()
# Set all the configuration
cfg.SOLVER.IMS_PER_BATCH = 2  # Number of training examples per batch utilized in one iteration
cfg.DATASETS.TEST = () # Maybe put here the validation set
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2  # Number of training examples per batch utilized in one iteration
cfg.SOLVER.BASE_LR = 4e-5 #args.learning_rate #example 0.00025
cfg.SOLVER.MAX_ITER = 300 #args.epochs
cfg.SOLVER.STEPS = []
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 24 #64 original 256
# Dataset
cfg.DATASETS.TRAIN = ("train","valid")
# Backbone
add_backbone_config(cfg)
cfg.MODEL.BACKBONE.NAME = "build_mnv2_CBAM_backbone"
# Proposal generator options
cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# ROI Heads
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Train
model = build_model(cfg)
device = get_device()
model = model_to_device(model,device)
print (cfg)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.train()