'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import *
import torch
import random

import torchvision.transforms as transforms


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

@DATASETS.register_module()
class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        split = config.subset
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot_depth', f'{self.fold}_depth.pkl')

        self.classes = None
        print_log('Load processed data from %s...' % self.pickle_path, logger = 'ModelNetFewShot')

        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        with open(self.pickle_path, 'rb') as f:
            pkl_file  = pickle.load(f) #[self.subset]

        self.dataset = pkl_file[self.subset]
        if self.subset == 'train':
            self.images = pkl_file['train_im']

        elif self.subset == 'test':
            self.images = pkl_file['test_im']

        print_log('The size of %s data is %d' % (split, len(self.dataset)), logger = 'ModelNetFewShot')



    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        im = self.images[index]

        im = self.transform(im)[:3]

        points[:, 0:3] = pc_normalize(points[:, 0:3])
        if not self.use_normals:
            points = points[:, 0:3]

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label,im)