import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torchvision.transforms as transforms

from PIL import Image

import random
from augment import *
from utils.augment import *

prompt_templates =[
        "There is a {category} in the scene.",
        "There is the {category} in the scene.",
        "a photo of a {category} in the scene.",
        "a photo of the {category} in the scene.",
        "a photo of one {category} in the scene.",
        "itap of a {category}.",
        "itap of my {category}.",
        "itap of the {category}.",
        "a photo of a {category}.",
        "a photo of my {category}.",
        "a photo of the {category}.",
        "a photo of one {category}.",
        "a photo of many {category}.",
        "a good photo of a {category}.",
        "a good photo of the {category}.",
        "a bad photo of a {category}.",
        "a bad photo of the {category}.",
        "a photo of a nice {category}.",
        "a photo of the nice {category}.",
        "a photo of a cool {category}.",
        "a photo of the cool {category}.",
        "a photo of a weird {category}.",
        "a photo of the weird {category}.",
        "a photo of a small {category}.",
        "a photo of the small {category}.",
        "a photo of a large {category}.",
        "a photo of the large {category}.",
        "a photo of a clean {category}.",
        "a photo of the clean {category}.",
        "a photo of a dirty {category}.",
        "a photo of the dirty {category}.",
        "a bright photo of a {category}.",
        "a bright photo of the {category}.",
        "a dark photo of a {category}.",
        "a dark photo of the {category}.",
        "a photo of a hard to see {category}.",
        "a photo of the hard to see {category}.",
        "a low resolution photo of a {category}.",
        "a low resolution photo of the {category}.",
        "a cropped photo of a {category}.",
        "a cropped photo of the {category}.",
        "a close-up photo of a {category}.",
        "a close-up photo of the {category}.",
        "a jpeg corrupted photo of a {category}.",
        "a jpeg corrupted photo of the {category}.",
        "a blurry photo of a {category}.",
        "a blurry photo of the {category}.",
        "a pixelated photo of a {category}.",
        "a pixelated photo of the {category}.",
        "a black and white photo of the {category}.",
        "a black and white photo of a {category}",
        "a plastic {category}.",
        "the plastic {category}.",
        "a toy {category}.",
        "the toy {category}.",
        "a plushie {category}.",
        "the plushie {category}.",
        "a cartoon {category}.",
        "the cartoon {category}.",
        "an embroidered {category}.",
        "the embroidered {category}.",
        "a painting of the {category}.",
        "a painting of a {category}.",
    ]



def generate_caption(class_name):    
    prompt = random.choice(prompt_templates)
    return prompt.replace("{category}",class_name)

@DATASETS.register_module()
class ScanObjectNN(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT

        self.classes = {'bag': 0, 'bin': 1, 'box': 2, 'cabinet': 3, 'chair': 4, 'desk': 5, 'display': 6, 'door': 7, 'shelf': 8, 'table': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'sofa': 13, 'toilet': 14}
        self.prob = 0.3
        self.key_list = list(self.classes.keys())
        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            try:
                self.images = np.array(h5['images']).astype(np.float32)
            except:
                pass
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def farthest_point_sample(self,point, npoint):
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

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        image = self.images[idx].copy()

        image = Image.fromarray(np.uint8(image*255.0))

        image = self.transform(image)


        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_point_cloud_90(current_points)

        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_point_cloud(current_points)
        
        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_perturbation_point_cloud(current_points)
        
        if np.random.uniform(0,1) < self.prob:
            current_points = jitter_point_cloud(current_points)

        if np.random.uniform(0,1) < self.prob:
            current_points = drop_random_points(current_points)


        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        
        class_name = self.key_list[int(label)]

        caption = generate_caption(class_name)




        

        return 'ScanObjectNN', 'sample', (current_points, label,image,caption)

    def __len__(self):
        return self.points.shape[0]

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


@DATASETS.register_module()
class ScanObjectNN_hardest(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT

        self.classes = {'bag': 0, 'bin': 1, 'box': 2, 'cabinet': 3, 'chair': 4, 'desk': 5, 'display': 6, 'door': 7, 'shelf': 8, 'table': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'sofa': 13, 'toilet': 14}
        self.key_list = list(self.classes.keys())
        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        self.prob = 0.3
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
         
            self.images = np.array(h5['images']).astype(np.float32)
           
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.images = np.array(h5['images']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def farthest_point_sample(self,point, npoint):
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


    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        
        current_points = pc_normalize(current_points)
        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_point_cloud_90(current_points)

        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_point_cloud(current_points)
        
        if np.random.uniform(0,1) < self.prob:
            current_points = rotate_perturbation_point_cloud(current_points)
        
        if np.random.uniform(0,1) < self.prob:
            current_points = jitter_point_cloud(current_points)

        if np.random.uniform(0,1) < self.prob:
            current_points = drop_random_points(current_points)

        current_points = torch.from_numpy(current_points).float()

        image = self.images[idx].copy()

        image = Image.fromarray(np.uint8(image*255.0))

        image = self.transform(image)
        
        label = self.labels[idx]

        class_name = self.key_list[int(label)]

        caption = generate_caption(class_name)
        
        return 'ScanObjectNN', 'sample', (current_points, label,image,caption)

    

    def __len__(self):
        return self.points.shape[0]