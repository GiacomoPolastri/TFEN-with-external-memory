import torch
import fiftyone as fo 
import detectron2
from detectron2 import model_zoo
#import detectron2_backbone
import os
### DATASET ###
from detectron2.data import MetadataCatalog, DatasetCatalog
from Data.utils import get_fiftyone_dicts
### DETECTRON2 CONFIGURATION ###
from detectron2.config.config import get_cfg
### BACKBONE ###
#from Models.mobilenetv2 import build_mnv2_backbone
#from Models.prova import build_mnv2_CBAM_backbone
from Models.mobilenetv2 import build_mnv2_classCBAM_backbone
#from detectron2_backbone.config import add_backbone_config
### MODEL ###
from detectron2.modeling import build_model
from utils import get_device, model_to_device
from detectron2.engine.defaults import DefaultTrainer
############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist1')
###################################################
### INFERENCE ###
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load trainset and prepare the dataset for Detectron2
trainset = fo.load_dataset('trainset')
view_train = trainset.match_tags('train')
DatasetCatalog.register('trainset', lambda view = view_train: get_fiftyone_dicts(view_train))
MetadataCatalog.get("trainset").thing_classes = ["docks", "boats", "lifts", "jetskis", "cars"]
#metadata_train = MetadataCatalog.get('train')

# Load validationset and prepare the dataset for Detectron2
validset = fo.load_dataset('validset')
view_valid = validset.match_tags('valid')
DatasetCatalog.register('validset', lambda view = view_valid: get_fiftyone_dicts(view_valid))
MetadataCatalog.get("validset").thing_classes = ["docks", "boats", "lifts", "jetskis", "cars"]
#metadata_valid = MetadataCatalog.get('valid')

### DETECTRON2 CONFIGURATION ###
cfg = get_cfg()
# Set all the configuration
cfg.SOLVER.IMS_PER_BATCH = 2  # Number of training examples per batch utilized in one iteration
cfg.DATASETS.TEST = () # Maybe put here the validation set
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2 # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.01      # pick a good LR default: 0.00025 
cfg.SOLVER.MAX_ITER = 30001    # 300 iterations seems good enough for this toy dataset; 
                             # you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 24 #original 256
cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0001
# Dataset
cfg.DATASETS.TRAIN = ("trainset","validset")
cfg.OUTPUT_DIR = "./output2"
# Backbone
#add_backbone_config(cfg)
cfg.MODEL.BACKBONE.NAME = "build_mnv2_classCBAM_backbone"
# Proposal generator options
cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# ROI Heads
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
# The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Create OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
### INFERENCE ###
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
weights_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
if os.path.exists(weights_path):
    cfg.MODEL.WEIGHTS = weights_path  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(cfg)
trainer.train()
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("validset", cfg, False, cfg.OUTPUT_DIR)
#'validset', cfg, False, cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "validset")
inference_on_dataset(trainer.model, val_loader, evaluator)

### PREDICTION TO FIFTYONE ###
# we generate predictions on each sample in the validation set 
# and convert the outputs from Detectron2 to FiftyOne format, 
# then add them to our FiftyOne dataset.
"""predictions = {}
for d in validset_dict:
    #print (d.file_name)
    img_w = d["width"]
    img_h = d["height"]
    img = cv2.imread(d["file_name"])
    #print (img)
    outputs = predictor(img)
    print (outputs)
    break"""
"""
    detections = detectron_to_fo(outputs, img_w, img_h)
    predictions[d["image_id"]] = detections

testset.set_values("predictions", predictions, key_field="id")

session = fo.launch_app(testset, desktop=True)
session.wait()
"""
