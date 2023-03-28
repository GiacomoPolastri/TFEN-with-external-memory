from detectron2.structures.image_list import ImageList
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
from Models.prova import build_mnv2_CBAM_backbone
#from detectron2_backbone.config import add_backbone_config
### MODEL ###
from detectron2.modeling import build_model
from utils import get_device, model_to_device
from detectron2.engine.defaults import DefaultTrainer
### INFERENCE ###
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from Models.CBAM import CBAM
import cv2
from Visualize.utils import detectron_to_fo
### CROP ###
from torch.nn import Threshold

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load validationset and prepare the dataset for Detectron2
validset = fo.load_dataset('validset')
view_valid = validset.match_tags('valid')
DatasetCatalog.register('validset', lambda view = view_valid: get_fiftyone_dicts(view_valid))
MetadataCatalog.get("validset").thing_classes = ["docks", "boats", "lifts", "jetskis", "cars"]
metadata_valid = MetadataCatalog.get('valid')

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
#cfg.DATASETS.TEST("validset")
cfg.INPUT.FORMAT = "BGR"
cfg.OUTPUT_DIR = "./output1"
# Backbone
#add_backbone_config(cfg)
cfg.MODEL.BACKBONE.NAME = "build_mnv2_CBAM_backbone"
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
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold

model = build_model(cfg)
model.eval()
#print (model)
predictor = DefaultPredictor(cfg)


    
# Useless
def take_module(model):
    blocks = list(model.children())
    for i, block in enumerate(blocks):
        #print (str(type(block)) + "\n")
        if i == 0:
            for module in block.modules():
                #print (type(module))
                if type(module) == CBAM:
                    cbam = module
                    #print (str(type(module)))
                    #print (module)
    return cbam

# TODO: use the crop.py to extract the good values of the feature map

validset_dict = get_fiftyone_dicts(view_valid)
predictions = {}
for d in validset_dict:
    
    im = cv2.imread(d['file_name'])
    img = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    width = d["width"]
    height = d["height"]
    inputs = [{"image": img, "height":height, "width": width}]
    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
    
    # feature map of CBAM: res4
    cbam = features['res4']
    ################################################
    ### MEMORY USAGE TENSOR ###
    size_one_element_byte = cbam.element_size()
    number_element = cbam.nelement()
    size_tensor_memory = size_one_element_byte * number_element
    print (size_tensor_memory)
    ################################################################

    
    cbam = cbam.squeeze(0)
    crop_tensor = torch.zeros(cbam.size())
    
    for i in range(len(cbam)):
        
        tensor = cbam[i]
        tensor -= torch.min(tensor)
        tensor /= torch.max(tensor)

        t = Threshold(0.5,0)
        threshold = t(tensor)
        crop_tensor[i] = threshold
    
    #print(crop_tensor)
    #print (cbam)
    ################################################################
    ### MEMORY USAGE CROP TENSOR ###
    crop_size_one_element_byte = crop_tensor.element_size()
    crop_number_element = crop_tensor.nelement()
    crop_size_tensor_memory = crop_size_one_element_byte * number_element
    print (crop_size_tensor_memory)
    ################################################################
    break
