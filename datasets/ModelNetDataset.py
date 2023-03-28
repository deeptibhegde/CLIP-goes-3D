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

from utils import misc

from PIL import Image

import torchvision.transforms as transforms


warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

@DATASETS.register_module()
class ModelNet(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = config.PROCESS
        self.uniform = True
        self.depth = config.DEPTH
        split = config.subset
        self.split = split
        self.subset = config.subset

        self.image = config.DEPTH

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger = 'ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print_log('Processing data %s (only running in the first time)...' % self.save_path, logger = 'ModelNet')
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                self.list_of_filenames = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    # print(fn)
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)


                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls
                    
                    p = fn[1].split('/')
                    file_name = p[-1][:-4]
                    # print(np.array(file_name.replace(fn[0] + '_','')).astype(np.int32))
                    self.list_of_filenames[index] = np.array(file_name.replace(fn[0] + '_','')).astype(np.int32)

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print_log('Load processed data from %s...' % self.save_path, logger = 'ModelNet')
                with open(self.save_path, 'rb') as f:
                    
                    self.list_of_points, self.list_of_labels = pickle.load(f)
                    

        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
        
            
            
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
            fn = self.datapath[index]

            if self.image:
                root = self.root.split('/')
                root = '/'.join(root[:-1])
                img_root = os.path.join(root,"mn40_depth_views/")


                i = 0
                im = pil_loader(os.path.join(img_root,fn[0],fn[0] + '_%04d'%file_name,'view%d.png'%(i+1)))
                im = self.transform(im)[:3]
            else:
                im = torch.tensor(0.0)        
            
        else:
            fn = self.datapath[index]

            


            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            p = fn[1].split('/')
            file_name = p[-1][:-4]

            file_name = np.array(file_name.replace(fn[0] + '_','')).astype(np.int32)
            


            if self.image:
                root = self.root.split('/')
                root = '/'.join(root[:-1])
                img_root = os.path.join(root,"mn40_depth_views/")


                i = 0
                im = pil_loader(os.path.join(img_root,fn[0],fn[0] + '_%04d'%file_name,'view%d.png'%(i+1)))
                im = self.transform(im)[:3]
            else:
                im = torch.tensor(0.0)   
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        
        return point_set, label[0], im
        
        


    def __getitem__(self, index):
        points, label , im = self._get_item(index)

        pt_idxs = np.arange(0, points.shape[0])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return 'ModelNet', 'sample', (current_points, label,im)

