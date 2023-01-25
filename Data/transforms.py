"""

Tranforms for Train, Val and Test set

"""

from torchvision import transforms as t

def train_transform(size, da):
    
    transform = t.Compose([
        t.Resize(256),
        t.RandomCrop(size),
        t.RandomHorizontalFlip(da),
        t.RandomRotation((da*100)),
        t.RandomInvert(da),
        t.RandomAutocontrast(da),
        t.RandomEqualize(da),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),  
    ])
    
    return transform

def val_transform(size):
    
    transform = t.Compose([
        t.Resize(256),
        t.RandomCrop(size),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
    ])
    
    return transform

def test_transform(size):
    
    transform = t.Compose([
        t.Resize(size),
        t.CenterCrop(size),
        t.ToTensor(),
        t.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
    ])
    
    return transform
    
    
    