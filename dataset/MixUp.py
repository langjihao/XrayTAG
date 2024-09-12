import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import random
import os

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

def mixup_data(x1, y1, x2, y2 ,alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y
class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root,  max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        self.mixup_alpha = args.mixup_alpha
        # clip_features = np.load(args.clip_features_path)
        # self.clip_features = clip_features['arr_0']
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0].replace('jpg','png'))).convert('RGB')
        image = self.transform(image)
        
        labels = ann['labels'][:14]
        labels = [0 if label in [0, 2] else 1 for label in labels]
        labels = torch.from_numpy(np.array(labels)).long()
                # MixUp
        if self.mixup_alpha > 0:
            index2 = random.randint(0, len(self.ann) - 1)
            ann2 = self.ann[index2]
            image_path2 = ann2['image_path']
            image2 = Image.open(os.path.join(self.image_root, image_path2[0].replace('jpg','png'))).convert('RGB')
            image2 = self.transform(image2)
            
            labels2 = ann2['labels'][:14]
            labels2 = [0 if label in [0, 2] else 1 for label in labels2]
            labels2 = torch.from_numpy(np.array(labels2)).long()
            
            image, labels = mixup_data(image, labels, image2, labels2, self.mixup_alpha)


        return image, labels
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=100, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.dataset = dataset
        self.args = args
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0].replace('jpg','png'))).convert('RGB')
        image = self.transform(image)


        labels = ann['labels'][:14]
        labels = [0 if label in [0, 2] else 1 for label in labels]
        labels = torch.from_numpy(np.array(labels))

        return image, labels