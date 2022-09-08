import os
import random
import torch.utils.data as data
from torch import distributed
import torchvision as tv
import numpy as np
from .utils import Subset, filter_images, group_images
import torch

from PIL import Image

from .coco_base import COCOSeg
from .coco_20i import COCO20iReader
from .baseset import base_set

cfg = {'DATASET': {
           'TRANSFORM': {
                'TRAIN': {
                    'transforms': ('normalize', ),
                    'joint_transforms': ('joint_random_scale_crop', 'joint_random_horizontal_flip'),
                    'TRANSFORMS_DETAILS': {
                        'NORMALIZE': {
                            'mean': (0.485, 0.456, 0.406),
                            'sd': (0.229, 0.224, 0.225),
                        },
                        'crop_size': (512, 512)
                    }
                },
                'TEST': {
                    'transforms': ('normalize', ),
                    'TRANSFORMS_DETAILS': {
                        'NORMALIZE': {
                            'mean': (0.485, 0.456, 0.406),
                            'sd': (0.229, 0.224, 0.225),
                        },
                        'crop_size': (512, 512)
                    }
                }}}}

class COCOSegmentationIncremental(data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 labels=None,
                 labels_old=None,
                 idxs_path=None,
                 masking=True,
                 overlap=True,
                 step=0,
                 few_shot=False,
                 num_shot=5,
                 batch_size=24):

        COCO_PATH = os.path.join(root, "COCO2017")
        folding = 3

        if step == 0:
            ds = COCO20iReader(COCO_PATH, folding, True, exclude_novel=True)
            self.dataset = base_set(ds, "train", cfg)
        else:
            ds = COCOSeg(COCO_PATH, True)
            self.dataset = base_set(ds, "test", cfg) # Use test config to keep original scale of the image.

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

