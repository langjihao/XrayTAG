import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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


class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root,  max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        clip_features = np.load("data/clip.npz")
        self.clip_features = clip_features['arr_0']
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0].replace('jpg','png'))).convert('RGB')
        image = self.transform(image)
        
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, cls_labels, clip_memory
    
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
        clip_features = np.load("data/clip.npz")
        self.clip_features = clip_features['arr_0']
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0].replace('jpg','png'))).convert('RGB')
        image = self.transform(image)

        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, cls_labels, clip_memory
