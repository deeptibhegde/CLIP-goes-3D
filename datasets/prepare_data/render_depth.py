# import logging
import multiprocessing
from tqdm import tqdm 
import random

import argparse


import os 
from .utils.pc_util import * 

import numpy as np

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logging.getLogger('requests').setLevel(logging.CRITICAL)
# logger = logging.getLogger(__name__)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

def do_augment(args1):
    args = args1['args']
    for line in args1['file_list']:
        
        line = line.strip()
        taxonomy_id = line.split('-')[0]
        temp = line.replace(taxonomy_id + '-',"")
        model_id = temp.split('.')[0]

        # /media/SSD/3d/clasp_pointnet/ShapeNetCore.v2/02747177/1b7d468a27208ee3dad910e221d16b18/models/model_normalized.obj

        if not os.path.isdir(os.path.join(save_path,taxonomy_id)):
            os.mkdir(os.path.join(save_path,taxonomy_id))
        if not os.path.isdir(os.path.join(save_path,taxonomy_id,model_id)):
            os.mkdir(os.path.join(save_path,taxonomy_id,model_id))


            points = IO.get(os.path.join(args.dataset_path, line)).astype(np.float32)

            indices = np.random.choice(np.arange(points.shape[0]),8192)

            points = points[indices]

            img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
            img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
            img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)

            img1 = Image.fromarray(np.uint8(img1*255.0))
            img2 = Image.fromarray(np.uint8(img2*255.0))
            img3 = Image.fromarray(np.uint8(img3*255.0))

            img1.save(os.path.join(save_path,taxonomy_id,model_id,'view1.png'))
            img2.save(os.path.join(save_path,taxonomy_id,model_id,'view2.png'))
            img3.save(os.path.join(save_path,taxonomy_id,model_id,'view3.png')) 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type = str, default='/media/SSD/3d/clasp_pointnet/', help = '')
    parser.add_argument('--dataset_path', type = str, default='/media/SSD/3d/clasp_pointnet/ShapeNet55/shapenet_pc', help = '')
    parser.add_argument('--save_path', type = str, default='shapenet_depth_views', help = '')
    parser.add_argument('--name', type = str, default='shapenet', help = '')


args = parser.get_args()


save_path = os.path.join(args.dataset_root,args,save_path)

if not os.path.isdir(save_path):
    os.mkdir(save_path)

if args.name == 'shapenet':
    data_list_file = os.path.join('ShapeNet','train_img.txt')
    test_data_list_file = os.path.join('ShapeNet','test_img.txt')
else:
    data_list_file = os.path.join('ModelNet','train_img.txt')
    test_data_list_file = os.path.join('ModelNet','test_img.txt')

with open(data_list_file, 'r') as f:
        lines = f.readlines()

num_processes = 1
start_id = 0
num_sub_files = int(len(lines)/num_processes)
file_list = lines
jobs = []
for i in range(num_processes):
	file_list_sub = file_list[start_id: start_id+num_sub_files] 
	dict_input = {'file_list':file_list_sub, 'thread_id' :i,'args' : args}
	p = multiprocessing.Process(target=do_augment, args = (dict_input,))
	jobs.append(p)
	p.start()
	start_id += num_sub_files



# file_list_sub = file_list[start_id:]
# do_augment(lines,rate)


