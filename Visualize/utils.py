import numpy as np
import torch
import torch.nn as nn
from Models import CBAM as convbam
from runtime_args import args   
import fiftyone as fo  

class Layers ():
    
    def __init__ (self, args, model):
        
        self.layers = args.layers
        self.mnet = args.mnet
        self.cbam = args.cbam
        self.sam = args.sam
        self.cam = args.cam
        self.last_cnn_layer = args.last_cnn_layer
        self.model = model
        self.cnn_layers, self.mnet_block, self.cbam_block, self.cam_block, self.sam_block, self.last_cnn_layer, self.sam_cnn_layer = self.get_layers()
        
    def get_layers (self):

        cnn_layers, sam_cnn_layer = [], []
        mnet_block, cbam_block, cam_block, sam_block, last_cnn_layer = None, None, None, None, None
        
        blocks = list(self.model.children())

        for i, block in enumerate(blocks):
    
            if i == 0: 
                mnet_block = block
                for layer in block.modules():
                    if (type(layer) == nn.Conv2d):
                        cnn_layers.append(layer)
            if i == 1: 
                cbam_block = block
                for child in block.children():
                    if (type(child) == convbam.ChannelGate): cam_block = child
                    if (type(child) == convbam.SpatialGate): 
                        sam_block = child
                        for layer in sam_block.children():
                            if (type(layer) == convbam.ChannelPool): sam_cnn_layer.append(layer)
                            if (type(layer) == convbam.BasicConv):
                                for l in layer.children():
                                    if (type(l) == nn.Conv2d): sam_cnn_layer.append(l)
            if i == 2: last_cnn_layer = block      
        
        return cnn_layers, mnet_block, cbam_block, cam_block, sam_block, last_cnn_layer, sam_cnn_layer
    
    
def get_output_layer(layers, image):
    
    output = image
    outputs, names = [], []
    for layer in layers:
        output = layer(output)
        outputs.append(output)
        names.append(str(type(layer))) #type
        
    return outputs, names

def to_processed (outputs):
    
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    
    return processed
    
    #processed = []
    #for fm in outputs:
    #    fm = fm.squeeze(0).detach().cpu().numpy()
    #    fm = np.clip(fm, 0,1)
    #    processed.append(fm)
    #return fm



def normalize (input):
    
    for i in input:
        for j in range(len(input)):
            input[j] -= np.amin(input[j])
            input[j] += np.amin(input[j])
    return input
            

def detectron_to_fo(outputs, img_w, img_h):
    # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    detections = []
    instances = outputs["instances"].to("cpu")
    for pred_box, score, c in zip(
        instances.pred_boxes, instances.scores, instances.pred_classes,
    ):
        x1, y1, x2, y2 = pred_box
        bbox = [float(x1)/img_w, float(y1)/img_h, float(x2-x1)/img_w, float(y2-y1)/img_h]
        detection = fo.Detection(label = pred_classes_to_label(c), confidence=float(score), bounding_box=bbox)
        
        detections.append(detection)

    return fo.Detections(detections=detections)

def pred_classes_to_label(pc):
    label = 'none'
    if pc == 1:
        label = 'boat'
        return label
    if pc == 2:
        label = 'car'
        return label
    if pc == 3:
        label = 'dock'
        return label
    if pc == 4:
        label = 'jetski'
        return label
    if pc == 0:
        label = 'lift'
        return label