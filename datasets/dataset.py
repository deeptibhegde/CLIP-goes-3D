import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

from utils.augment import *

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import random

import json

import clip

from augment import *


import torchvision.transforms as transforms


MAP = {'uniform': uniform_noise,
       'gaussian': gaussian_noise,
       'background': background_noise,
       'impulse': impulse_noise,
       'scale': scale,
    #    'upsampling': upsampling,
       'shear': shear,
       'rotation': rotation,
       'cutout': cutout,
    #    'density': density,
    #    'distortion': ffd_distortion,
       'distortion_rbf': rbf_distortion,
    #    'distortion_rbf_inv': rbf_distortion_inv,
    #    'occlusion': occlusion,
    #    'lidar': lidar,
    #    'original': None,
}




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

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')    

@DATASETS.register_module()
class ShapenetCG3D(data.Dataset):
    def __init__(self, args,config,train=True,real=False):
        self.dataset_root = args.dataset_root
        
        self.npoints = 8192
        
        self.data_list_file = os.path.join(self.dataset_root, 'shapenet_render/train_img.txt')
        self.test_data_list_file = os.path.join(self.dataset_root, 'shapenet_render/test_img.txt')

        self.real = real

        self.real_caption = args.real_caption

        self.depth = args.depth

        f = open(os.path.join(self.dataset_root,"shapenet_render/shape_names.txt"))

        self.classes = f.readlines()
        self.prob = args.aug_prob

        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i][:-1]

        f.close()
        
        f = open(os.path.join(self.dataset_root,"shapenet_render/taxonomy.json"))


        self.tax = json.load(f)

        f.close()



        self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        if train:
            with open(self.data_list_file, 'r') as f:
                lines = f.readlines()
        else:
            with open(self.test_data_list_file, 'r') as f:
                lines = f.readlines()

        self.sample_points_num = self.npoints

        self.sample_sizes = [1024,2048,8192]

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        # with open(self.data_list_file, 'r') as f:
        #     lines = f.readlines()
        # if self.whole:
        #     with open(test_data_list_file, 'r') as f:
        #         test_lines = f.readlines()
        #     print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
        #     lines = test_lines + lines
        self.file_list = []
        
        for line in lines:
            view = np.random.randint(0,5,1)[0]
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            temp = line.removeprefix(taxonomy_id + '-')
            model_id = temp.split('.')[0]

            img_file_path = os.path.join(self.dataset_root,"shapenet_render/img",taxonomy_id,model_id,'%03d.png'%view)

            choice = np.random.randint(1,3,1)[0]
            depth_img_file_path = os.path.join(self.dataset_root,"shapenet_depth_views",taxonomy_id,model_id,'view%d.png'%choice)

            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line,
                'img_file_path': img_file_path,
                'depth_img_file_path': depth_img_file_path,
            })
        if not train:
            self.file_list = self.file_list[:3000]
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        
    
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


    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        # print(idx)

        points = IO.get(os.path.join(self.dataset_root,'ShapeNet55/shapenet_pc', sample['file_path'])).astype(np.float32)

        # points = self.random_sample(points, self.sample_points_num)
        sample_size = random.choice(self.sample_sizes)
        # points = self.farthest_point_sample(points,self.sample_points_num)
        # points = self.farthest_point_sample(points,sample_size)
        points = self.pc_norm(points)

        #augmentations
        prob = self.prob
        # severity = 3
        # corruptions = [c for c in MAP.keys()]
        # corruption = random.choice(corruptions)
        # for corruption in MAP.keys():
        #     if np.random.uniform(0,1) < self.prob:
        #         points = MAP[corruption](points,severity)
                # print(points.shape)

        # points = MAP[corruption](points,severity)



        index = np.random.choice(points.shape[0],self.npoints)

        points = points[index]
        if np.random.uniform(0,1) < self.prob:
            points = rotate_point_cloud_90(points)

        if np.random.uniform(0,1) < self.prob:
            points = rotate_point_cloud(points)
        
        if np.random.uniform(0,1) < self.prob:
            points = rotate_perturbation_point_cloud(points)
        
        if np.random.uniform(0,1) < self.prob:
            points = jitter_point_cloud(points)

        if np.random.uniform(0,1) < self.prob:
            points = drop_random_points(points)




        points = torch.from_numpy(points).float()


        synset = sample['taxonomy_id']
        syn_dict = next(item for item in self.tax if item["synsetId"] == synset)
        names = syn_dict['name'].split(',')
        class_name = names[0]

        if class_name == 'display':
                class_name = 'monitor'
        elif class_name == 'vessel':
                class_name = 'ship'
        elif class_name == 'ashcan':
                class_name = 'trashcan'

        if self.real_caption:
            real_caption_path = os.path.join(self.dataset_root,'ShapeNetCore.v2_images',class_name + '.json')
            with open(real_caption_path) as json_file:
                cap_data = json.load(json_file)
                
            real_captions = cap_data['caption']
            caption = random.choice(real_captions) 

        else:
            caption = generate_caption(class_name)


        if not self.real:
            # print(sample['img_file_path'])
            im = pil_loader(sample['img_file_path'])
            im = self.transform(im)[:3]

            

            depth_im = pil_loader(sample['depth_img_file_path'])
            depth_im = self.transform(depth_im)[:3]

        else:
            

            if np.random.uniform(0,1) < 0.5:
                real_list = os.listdir(os.path.join(self.dataset_root,'ShapeNetCore.v2_images',class_name))

                choice = random.choice(real_list)
                im = pil_loader(os.path.join(self.dataset_root,'ShapeNetCore.v2_images',class_name,choice))
                im = self.transform(im)
                if im.shape[0] != 3:
                    im = torch.cat((im,im,im),dim=0)
            else:
                im = pil_loader(sample['img_file_path'])
                im = self.transform(im)[:3]

            depth_im = pil_loader(sample['depth_img_file_path'])
            depth_im = self.transform(depth_im)[:3]

            

        return 0, 0, points, (im,depth_im),caption, class_name, int(self.classes.index(class_name))

    def __len__(self):
        return len(self.file_list)
