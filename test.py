### DATASET ###
import fiftyone as fo
from Data.utils import get_fiftyone_dicts
from detectron2.data import MetadataCatalog, DatasetCatalog
### DETECTRON2 CONFIGURATION ###
from detectron2.config.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
from detectron2_backbone.config import add_backbone_config
from Models.prova import build_mnv2_CBAM_backbone
### PREDICTION TO FIFTYONE ###
import cv2
from Visualize.utils import detectron_to_fo
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# validset
validset = fo.load_dataset('validset')
view = validset.match_tags('valid')
validset_dict = get_fiftyone_dicts(view)

cfg = get_cfg()
# Create OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

### INFERENCE ###
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

predictor = DefaultPredictor(cfg)

### PREDICTION TO FIFTYONE ###
# we generate predictions on each sample in the validation set 
# and convert the outputs from Detectron2 to FiftyOne format, 
# then add them to our FiftyOne dataset.
predictions = {}
for d in validset_dict:
    print (d)
    img_w = d["width"]
    img_h = d["height"]
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    print (outputs)
    break
    """
    detections = detectron_to_fo(outputs, img_w, img_h)
    predictions[d["image_id"]] = detections

testset.set_values("predictions", predictions, key_field="id")

session = fo.launch_app(testset, desktop=True)
session.wait()
"""
