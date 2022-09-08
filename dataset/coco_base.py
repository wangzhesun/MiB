# import os
# from copy import deepcopy
# import torch
# from torch import distributed
# import torchvision
#
# import sys
# import torch.utils.data as data
# import numpy as np
# import json
# import random
#
# from PIL import Image
# from .utils import Subset, filter_images, group_images
#
# # from utils.tasks import get_dataset_list, get_tasks
#
# import shutil
# from tqdm import trange
#
# from pycocotools import mask as coco_mask
# from pycocotools.coco import COCO
#
#
# # import utils
# # from .baseset import base_set
#
# novel_dict = {
#     0: [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77],
#     1: [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78],
#     2: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79],
#     3: [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80]
# }
#
#
# class COCOSegmentation(data.Dataset):
#     def __init__(self,
#                  root,
#                  image_set='train',
#                  is_aug=True,
#                  transform=None):
#
#         self.root = os.path.expanduser(root)
#         # self.year = "2012"
#         # self.min_area = 200  # small areas are marked as crowded
#         self.transform = transform
#
#         self.image_set = image_set
#
#         base_dir = "COCO2017"
#         coco_root = os.path.join(self.root, base_dir)
#         if not os.path.isdir(coco_root):
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#
#         self.annotation_path = os.path.join(coco_root, 'annotations',
#                                             'instances_{}2017.json'.format(image_set))
#         assert os.path.exists(
#             self.annotation_path), "SegmentationClassAug not found"
#
#         self.img_dir = os.path.join(coco_root, '{}2017'.format(image_set))
#         # splits_dir = os.path.join(voc_root, 'splits')
#         self.coco = COCO(self.annotation_path)
#         self.img_ids = list(self.coco.imgs.keys())
#
# #########################################################
#         # COCO class
#         class_list = sorted([i for i in self.coco.cats.keys()])
#
#         # The instances labels in COCO dataset is not dense
#         # e.g., total 80 classes. Some objects are labeled as 82
#         # but they are 73rd class; while none is labeled as 83.
#         self.class_map = {}
#         for i in range(len(class_list)):
#             self.class_map[class_list[i]] = i + 1
#
#         # Given a class idx (1-80), self.instance_class_map gives the list of images that contain
#         # this class idx
#         class_map_dir = os.path.join(coco_root, 'instance_seg_class_map', image_set)
#         if not os.path.exists(class_map_dir):
#             # Merge VOC and SBD datasets and create auxiliary files
#             try:
#                 self.create_coco_class_map(class_map_dir)
#             except (Exception, KeyboardInterrupt) as e:
#                 # Dataset creation fail for some reason...
#                 shutil.rmtree(class_map_dir)
#                 raise e
#
#         self.instance_class_map = {}
#         for c in range(1, 81):
#             class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
#             with open(class_map_path, 'r') as f:
#                 class_idx_list = f.readlines()
#             class_idx_list = [int(i.strip()) for i in class_idx_list if i]
#             self.instance_class_map[c] = class_idx_list
#
#         self.CLASS_NAMES_LIST = ['background']
#         for i in range(len(class_list)):
#             cls_name = self.coco.cats[class_list[i]]['name']
#             self.CLASS_NAMES_LIST.append(cls_name)
#
#     def create_coco_class_map(self, class_map_dir):
#         assert not os.path.exists(class_map_dir)
#         os.makedirs(class_map_dir)
#
#         instance_class_map = {}
#         for c in range(1, 81):
#             instance_class_map[c] = []
#
#         print("Computing COCO class-object masks...")
#         for i in trange(len(self)):
#             img_id = self.img_ids[i]
#             mask = self._get_mask(img_id)
#             contained_labels = torch.unique(mask)
#             for c in contained_labels:
#                 c = int(c)
#                 if c == 0 or c == -1:
#                     continue  # background or ignore_mask
#                 instance_class_map[c].append(str(i))  # use string to format integer to write to txt
#
#         for c in range(1, 81):
#             with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
#                 f.write('\n'.join(instance_class_map[c]))
#
#     def get_class_map(self, class_id):
#         return deepcopy((self.instance_class_map[class_id]))
#
#     def get_label_range(self):
#         return [i + 1 for i in range(80)]
#     ##############################################################
#
#     def _get_img(self, img_id):
#         img_desc = self.coco.imgs[img_id]
#         img_fname = img_desc['file_name']
#         img_fpath = os.path.join(self.img_dir, img_fname)
#         return Image.open(img_fpath).convert('RGB')
#
#     def _get_mask(self, img_id):
#         img_desc = self.coco.imgs[img_id]
#         img_fname = img_desc['file_name']
#         label_fname = img_fname.replace('.jpg', '.png')
#         img_fpath = os.path.join(self.img_dir, label_fname)
#         return self._get_seg_mask(img_fpath)
#
#     def _get_seg_mask(self, fname: str):
#         deleted_idx = [91, 83, 71, 69, 68, 66, 45, 30, 29, 26, 12]
#         raw_lbl = np.array(Image.open(fname)).astype(np.int)
#         ignore_idx = (raw_lbl == 255)
#         raw_lbl += 1
#         raw_lbl[raw_lbl > 91] = 0  # STUFF classes are mapped to background
#         for d_idx in deleted_idx:
#             raw_lbl[raw_lbl > d_idx] -= 1
#         raw_lbl[ignore_idx] = -1
#         return torch.tensor(raw_lbl)
#
#     def __getitem__(self, index):
#         img_id = self.img_ids[index]
#         img = self._get_img(img_id)
#         seg_mask = self._get_mask(img_id)  # tensor
#         return img, seg_mask
#
#     def __len__(self):
#         return len(self.coco.imgs)
#
#
# ############################################################################
# class COCO20iBase(torchvision.datasets.vision.VisionDataset):
#     def __init__(self, root, labels, labels_old, split, exclude_novel=False, vanilla_label=False):
#         super(COCO20iBase, self).__init__(root, None, None, None)
#         # assert fold >= 0 and fold <= 3
#         assert split in [True, False]
#         if vanilla_label:
#             assert exclude_novel
#         self.vanilla_label = vanilla_label
#
#         # Get augmented VOC dataset
#         self.vanilla_ds = COCOSegmentation(root, split)
#         self.CLASS_NAMES_LIST = self.vanilla_ds.CLASS_NAMES_LIST
#
#         # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
#         # Given fold number, define L_{test}
#         self.val_label_set = labels
#         self.train_label_set = labels_old
#         self.train_label_set.remove(0)
#
#         self.visible_labels = self.train_label_set
#         self.invisible_labels = self.val_label_set
#
#         # Pre-training or meta-training
#         if exclude_novel:
#             # Exclude images containing invisible classes and use rest
#             novel_examples_list = []
#             for label in self.invisible_labels:
#                 novel_examples_list += self.vanilla_ds.get_class_map(label)
#             self.subset_idx = [i for i in range(len(self.vanilla_ds))]
#             self.subset_idx = list(set(self.subset_idx) - set(novel_examples_list))
#         else:
#             # Use images containing at least one pixel from relevant classes
#             examples_list = []
#             for label in self.visible_labels:
#                 examples_list += self.vanilla_ds.get_class_map(label)
#             self.subset_idx = list(set(examples_list))
#
#         # Sort subset idx to make dataset deterministic (because set is unordered)
#         self.subset_idx = sorted(self.subset_idx)
#
#         # Generate self.class_map
#         self.class_map = {}
#         for c in range(1, 81):
#             self.class_map[c] = []
#             real_class_map = self.vanilla_ds.get_class_map(c)
#             real_class_map_lut = {}
#             for idx in real_class_map:
#                 real_class_map_lut[idx] = True
#             for subset_i, real_idx in enumerate(self.subset_idx):
#                 if real_idx in real_class_map_lut:
#                     self.class_map[c].append(subset_i)
#
#         self.remap_dict = {}
#         map_idx = 1
#         for c in range(1, 81):
#             if c in self.val_label_set:
#                 self.remap_dict[c] = 0  # novel classes are masked as background
#             else:
#                 assert c in self.train_label_set
#                 self.remap_dict[c] = map_idx
#                 map_idx += 1
#
#     def __len__(self):
#         return len(self.subset_idx)
#
#     def get_class_map(self, class_id):
#         """
#         class_id here is subsetted. (e.g., class_idx is 12 in vanilla dataset may get translated to 2)
#         """
#         assert class_id > 0
#         assert class_id < (len(self.visible_labels) + 1)
#         # To access visible_labels, we translate class_id back to 0-indexed
#         return deepcopy(self.class_map[self.visible_labels[class_id - 1]])
#
#     def get_label_range(self):
#         return [i + 1 for i in range(len(self.visible_labels))]
#
#     def __getitem__(self, idx: int):
#         assert 0 <= idx and idx < len(self.subset_idx)
#         img, target_tensor = self.vanilla_ds[self.subset_idx[idx]]
#         if not self.vanilla_label:
#             target_tensor = self.mask_pixel(target_tensor)
#         return img, target_tensor
#
#     def mask_pixel(self, target_tensor):
#         """
#         Following OSLSM, we mask pixels not in current label set as 0. e.g., when
#         self.train = True, pixels whose labels are in L_{test} are masked as background
#
#         Parameters:
#             - target_tensor: segmentation mask (usually returned array from self.load_seg_mask)
#
#         Return:
#             - Offseted and masked segmentation mask
#         """
#         # Use the property that validation label split is contiguous to accelerate
#         label_set = torch.unique(target_tensor)
#         for l in label_set:
#             l = int(l)
#             if l == 0 or l == -1:
#                 continue  # background and ignore_label are unchanged
#             src_label = l
#             target_label = self.remap_dict[l]
#             target_tensor[target_tensor == src_label] = target_label
#         return target_tensor
#
#
# ###############################################################################
#
#
# class COCOSegmentationIncremental(data.Dataset):
#     def __init__(self,
#                  root,
#                  train=True,
#                  transform=None,
#                  labels=None,
#                  labels_old=None,
#                  idxs_path=None,
#                  masking=True,
#                  overlap=True,
#                  step=0,
#                  few_shot=False,
#                  num_shot=5,
#                  batch_size=24):
#
# ################################################################
#         if step == 0:
#             coco_root = os.path.join(root, "COCO2017")
#
#             self.dataset = COCO20iBase(coco_root, labels=labels, labels_old=labels_old, split=True, exclude_novel=True)
#         else:
# #################################################################
#
#             full_coco = COCOSegmentation(root, 'train' if train else 'val', is_aug=True, transform=None)
#
#             ########################################################
#             # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
#             # Given fold number, define L_{test}
#             # self.val_label_set = labels
#             # self.train_label_set = [i for i in range(
#             #     1, 81) if i not in self.val_label_set]
#             #
#             # self.visible_labels = self.train_label_set
#             # self.invisible_labels = self.val_label_set
#             ########################################################
#
#             self.labels = []
#             self.labels_old = []
#
#             if labels is not None:
#                 # Split dataset based on folding. Refer to https://arxiv.org/pdf/1709.03410.pdf
#                 # Given fold number, define L_{test}
#                 labels_old = labels_old if labels_old is not None else []
#                 self.labels = [0] + labels
#                 self.labels_old = [0] + labels_old
#                 self.order = [0] + labels_old + labels
#
#                 # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
#                 if idxs_path is not None and os.path.exists(idxs_path):
#                     idxs = np.load(idxs_path).tolist()
#                 else:
#                     idxs = filter_images(full_coco, labels, labels_old, overlap=overlap)
#                     if idxs_path is not None and distributed.get_rank() == 0:
#                         np.save(idxs_path, np.array(idxs, dtype=int))
#
#                 if train:
#                     masking_value = 0
#                 else:
#                     masking_value = 255
#
#                 self.inverted_order = {label: self.order.index(label) for label in self.order}
#                 self.inverted_order[255] = masking_value
#
#                 reorder_transform = torchvision.transforms.Lambda(
#                     lambda t: t.apply_(
#                         lambda x: self.inverted_order[
#                             x] if x in self.inverted_order else masking_value))
#
#                 if masking:
#                     tmp_labels = self.labels + [255]
#                     target_transform = torchvision.transforms.Lambda(
#                         lambda t: t.apply_(
#                             lambda x: self.inverted_order[x] if x in tmp_labels else masking_value))
#                 else:
#                     target_transform = reorder_transform
#
#                 final_file_name = []
#                 if few_shot and step > 0 and train:
#                     seed = 2022
#                     np.random.seed(seed)
#                     random.seed(seed)
#                     torch.manual_seed(seed)
#                     for _ in range(num_shot):
#                         idx = random.choice(idxs)
#                         while True:
#                             if idx not in final_file_name:
#                                 final_file_name.append(idx)
#                                 break
#                             else:
#                                 idx = random.choice(idxs)
#                 else:
#                     final_file_name = idxs
#
#                 idxs = final_file_name
#
#                 while len(idxs) < batch_size:
#                     if num_shot == 5:
#                         idxs = idxs * 20
#                     elif num_shot == 1:
#                         idxs = idxs * 100
#                     else:
#                         idxs = idxs * 5
#
#                 # make the subset of the dataset
#                 self.dataset = Subset(full_coco, idxs, transform, target_transform)
#             else:
#                 self.dataset = full_coco
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is the image segmentation.
#         """
#         return self.dataset[index]
#
#     def __len__(self):
#         return len(self.dataset)
#
#     @staticmethod
#     def __strip_zero(labels):
#         while 0 in labels:
#             labels.remove(0)

import os
import numpy as np
import torch
import shutil
from tqdm import trange
from copy import deepcopy
from torchvision import datasets, transforms
from PIL import Image

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import utils
from .baseset import base_set

COCO_PATH = os.path.join(utils.get_dataset_root(), "COCO2017")


# 2017 train images normalization constants
#   mean: 0.4700, 0.4468, 0.4076
#   sd: 0.2439, 0.2390, 0.2420

class COCOSeg(datasets.vision.VisionDataset):
    def __init__(self, root, train=True):
        super(COCOSeg, self).__init__(root, None, None, None)
        self.min_area = 200  # small areas are marked as crowded
        split_name = "train" if train else "val"
        self.annotation_path = os.path.join(root, 'annotations',
                                            'instances_{}2017.json'.format(split_name))
        self.img_dir = os.path.join(root, '{}2017'.format(split_name))
        self.coco = COCO(self.annotation_path)
        self.img_ids = list(self.coco.imgs.keys())

        # COCO class
        class_list = sorted([i for i in self.coco.cats.keys()])

        # The instances labels in COCO dataset is not dense
        # e.g., total 80 classes. Some objects are labeled as 82
        # but they are 73rd class; while none is labeled as 83.
        self.class_map = {}
        for i in range(len(class_list)):
            self.class_map[class_list[i]] = i + 1

        # Given a class idx (1-80), self.instance_class_map gives the list of images that contain
        # this class idx
        class_map_dir = os.path.join(root, 'instance_seg_class_map', split_name)
        if not os.path.exists(class_map_dir):
            # Merge VOC and SBD datasets and create auxiliary files
            try:
                self.create_coco_class_map(class_map_dir)
            except (Exception, KeyboardInterrupt) as e:
                # Dataset creation fail for some reason...
                shutil.rmtree(class_map_dir)
                raise e

        self.instance_class_map = {}
        for c in range(1, 81):
            class_map_path = os.path.join(class_map_dir, str(c) + ".txt")
            with open(class_map_path, 'r') as f:
                class_idx_list = f.readlines()
            class_idx_list = [int(i.strip()) for i in class_idx_list if i]
            self.instance_class_map[c] = class_idx_list

        self.CLASS_NAMES_LIST = ['background']
        for i in range(len(class_list)):
            cls_name = self.coco.cats[class_list[i]]['name']
            self.CLASS_NAMES_LIST.append(cls_name)

    def create_coco_class_map(self, class_map_dir):
        assert not os.path.exists(class_map_dir)
        os.makedirs(class_map_dir)

        instance_class_map = {}
        for c in range(1, 81):
            instance_class_map[c] = []

        print("Computing COCO class-object masks...")
        for i in trange(len(self)):
            img_id = self.img_ids[i]
            mask = self._get_mask(img_id)
            contained_labels = torch.unique(mask)
            ############################################
            # print('printing contained labels: ')
            # print(contained_labels)
            ############################################
            for c in contained_labels:
                ##################################3333
                # if c not in range(0,81):
                #     continue
                ######################################
                c = int(c)
                if c == 0 or c == -1:
                    continue  # background or ignore_mask
                ###################################################################
                # print('printing c and i: ')
                # print(c)
                # print(i)
                ####################################################################
                instance_class_map[c].append(str(i))  # use string to format integer to write to txt

        for c in range(1, 81):
            with open(os.path.join(class_map_dir, str(c) + '.txt'), 'w') as f:
                f.write('\n'.join(instance_class_map[c]))

    def _get_img(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        img_fpath = os.path.join(self.img_dir, img_fname)
        return Image.open(img_fpath).convert('RGB')

    def _get_mask(self, img_id):
        img_desc = self.coco.imgs[img_id]
        img_fname = img_desc['file_name']
        ##################################
        label_fname = img_fname.replace('.jpg', '.png')
        img_fpath = os.path.join(self.img_dir, label_fname)
        # img_fpath = os.path.join(self.img_dir, img_fname)
        ##################################
        return self._get_seg_mask(img_fpath)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img = self._get_img(img_id)
        seg_mask = self._get_mask(img_id)  # tensor
        return (img, seg_mask)

    def _get_seg_mask(self, fname: str):
        deleted_idx = [91, 83, 71, 69, 68, 66, 45, 30, 29, 26, 12]
        raw_lbl = np.array(Image.open(fname)).astype(np.int)
        ignore_idx = (raw_lbl == 255)
        raw_lbl += 1
        raw_lbl[raw_lbl > 91] = 0  # STUFF classes are mapped to background
        for d_idx in deleted_idx:
            raw_lbl[raw_lbl > d_idx] -= 1
        raw_lbl[ignore_idx] = -1
        return torch.tensor(raw_lbl)

    def get_class_map(self, class_id):
        return deepcopy((self.instance_class_map[class_id]))

    def get_label_range(self):
        return [i + 1 for i in range(80)]

    def __len__(self):
        return len(self.coco.imgs)


def get_train_set(cfg):
    ds = COCOSeg(COCO_PATH, True)
    return base_set(ds, "train", cfg)


def get_val_set(cfg):
    ds = COCOSeg(COCO_PATH, False)
    return base_set(ds, "test", cfg)