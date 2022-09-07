import os
from copy import deepcopy
import torch
from torch import distributed
import torchvision

import sys
import torch.utils.data as data
import numpy as np
import json
import random

from PIL import Image
from .utils import Subset, filter_images, group_images

# from utils.tasks import get_dataset_list, get_tasks

import shutil
from tqdm import trange

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

# import utils
# from .baseset import base_set

class COCOSegmentation(data.Dataset):
    def __init__(self,
                 root,
                 image_set='train',
                 is_aug=True,
                 transform=None):

        self.root = os.path.expanduser(root)
        # self.year = "2012"
        # self.min_area = 200  # small areas are marked as crowded
        self.transform = transform

        self.image_set = image_set

        base_dir = "COCO2017"
        coco_root = os.path.join(self.root, base_dir)
        if not os.path.isdir(coco_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.annotation_path = os.path.join(coco_root, 'annotations',
                                            'instances_{}2017.json'.format(image_set))
        assert os.path.exists(
            self.annotation_path), "SegmentationClassAug not found"

        self.img_dir = os.path.join(coco_root, '{}2017'.format(image_set))
        # splits_dir = os.path.join(voc_root, 'splits')
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())

        # # COCO class
        # class_list = sorted([i for i in self.coco.cats.keys()])

        # # The instances labels in COCO dataset is not dense
        # # e.g., total 80 classes. Some objects are labeled as 82
        # # but they are 73rd class; while none is labeled as 83.
        # self.class_map = {}
        # for i in range(len(class_list)):
        #     self.class_map[class_list[i]] = i + 1
        #
        # # Given a class idx (1-80), self.instance_class_map gives the list of images that contain
        # # this class idx
        # class_map_dir = os.path.join(coco_root, 'instance_seg_class_map', image_set)
        # if not os.path.exists(class_map_dir):
        #     # Merge VOC and SBD datasets and create auxiliary files
        #     try:
        #         self.create_coco_class_map(class_map_dir)
        #     except (Exception, KeyboardInterrupt) as e:
        #         # Dataset creation fail for some reason...
        #         shutil.rmtree(class_map_dir)
        #         raise e
        #
        # self.instance_class_map = {}
        # for c in range(1, 81):
        #     class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
        #     with open(class_map_path, 'r') as f:
        #         class_idx_list = f.readlines()
        #     class_idx_list = [int(i.strip()) for i in class_idx_list if i]
        #     self.instance_class_map[c] = class_idx_list
        #
        # self.CLASS_NAMES_LIST = ['background']
        # for i in range(len(class_list)):
        #     cls_name = self.coco.cats[class_list[i]]['name']
        #     self.CLASS_NAMES_LIST.append(cls_name)


    # def create_coco_class_map(self, class_map_dir):
    #     assert not os.path.exists(class_map_dir)
    #     os.makedirs(class_map_dir)
    #
    #     instance_class_map = {}
    #     for c in range(1, 81):
    #         instance_class_map[c] = []
    #
    #     print("Computing COCO class-object masks...")
    #     for i in trange(len(self)):
    #         img_id = self.img_ids[i]
    #         mask = self._get_mask(img_id)
    #         contained_labels = torch.unique(mask)
    #         for c in contained_labels:
    #             c = int(c)
    #             if c == 0 or c == -1:
    #                 continue  # background or ignore_mask
    #             instance_class_map[c].append(str(i))  # use string to format integer to write to txt
    #
    #     for c in range(1, 81):
    #         with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
    #             f.write('\n'.join(instance_class_map[c]))

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')

    def _get_mask(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        label_fname = img_fname.replace('.jpg', '.png')
        img_fpath = os.path.join(self.img_dir, label_fname)
        return self._get_seg_mask(img_fpath)

    def _get_seg_mask(self, fname: str):
        deleted_idx = [91, 83, 71, 69, 68, 66, 45, 30, 29, 26, 12]
        raw_lbl = np.array(Image.open(fname)).astype(np.int)
        ignore_idx = (raw_lbl == 255)
        raw_lbl += 1
        raw_lbl[raw_lbl > 91] = 0 # STUFF classes are mapped to background
        for d_idx in deleted_idx:
            raw_lbl[raw_lbl > d_idx] -= 1
        raw_lbl[ignore_idx] = -1
        return torch.tensor(raw_lbl)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img = self._get_img(img_id)
        seg_mask = self._get_mask(img_id)  # tensor
        return img, seg_mask

    def __len__(self):
        return len(self.coco.imgs)


###############################################################################


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

        full_coco = COCOSegmentation(root, 'train' if train else 'val', is_aug=True, transform=None)

        ########################################################
        # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
        # Given fold number, define L_{test}
        # self.val_label_set = labels
        # self.train_label_set = [i for i in range(
        #     1, 81) if i not in self.val_label_set]
        #
        # self.visible_labels = self.train_label_set
        # self.invisible_labels = self.val_label_set
        ########################################################

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
            # Given fold number, define L_{test}
            labels_old = labels_old if labels_old is not None else []
            self.labels = [0] + labels
            self.labels_old = [0] + labels_old
            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_coco, labels, labels_old, overlap=overlap)
                if idxs_path is not None and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            if train:
                masking_value = 0
            else:
                masking_value = 255

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            self.inverted_order[255] = masking_value

            reorder_transform = torchvision.transforms.Lambda(
                lambda t: t.apply_(
                    lambda x: self.inverted_order[x] if x in self.inverted_order else masking_value))

            if masking:
                tmp_labels = self.labels + [255]
                target_transform = torchvision.transforms.Lambda(
                    lambda t: t.apply_(
                        lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
            else:
                target_transform = reorder_transform

            ####################################################################################
            final_file_name = []
            if few_shot and step > 0 and train:
                seed = 2022
                np.random.seed(seed)
                random.seed(seed)
                torch.manual_seed(seed)
                for _ in range(num_shot):
                    idx = random.choice(idxs)
                    while True:
                        if idx not in final_file_name:
                            final_file_name.append(idx)
                            break
                        else:
                            idx = random.choice(idxs)
            else:
                final_file_name = idxs

            idxs = final_file_name

            while len(idxs) < batch_size:
                if num_shot == 5:
                    idxs = idxs * 20
                elif num_shot == 1:
                    idxs = idxs * 100
                else:
                    idxs = idxs * 5
            ####################################################################################

            # make the subset of the dataset
            self.dataset = Subset(full_coco, idxs, transform, target_transform)
        else:
            self.dataset = full_coco

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)