# import logging
import multiprocessing
from tqdm import tqdm 
import random


import os 
from pc_util import * 

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
        return np.loadtxt(file_path, delimiter=',').astype(np.float32)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

def do_augment(args1):
    for line in args1['file_list']:
        line = line[:-1]
        
        class_name = line.split('_')[:-1]
        class_name = '_'.join(class_name)
        filename = line + '.txt'

        if not os.path.isdir(os.path.join(save_path,class_name)):
            os.mkdir(os.path.join(save_path,class_name))
        if not os.path.isdir(os.path.join(save_path,class_name,line)):
            os.mkdir(os.path.join(save_path,class_name,line))


            points = IO.get(os.path.join(data_path,'modelnet40_normal_resampled/',class_name, filename)).astype(np.float32)

            indices = np.random.choice(np.arange(points.shape[0]),8192)

            points = points[indices]

            img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi)
            img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi)
            img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi)

            img1 = Image.fromarray(np.uint8(img1*255.0))
            img2 = Image.fromarray(np.uint8(img2*255.0))
            img3 = Image.fromarray(np.uint8(img3*255.0))

            img1.save(os.path.join(save_path,class_name,line,'view1.png'))
            img2.save(os.path.join(save_path,class_name,line,'view2.png'))
            img3.save(os.path.join(save_path,class_name,line,'view3.png')) 




data_path = "/media/SSD/3d/clasp_pointnet/"

save_path = os.path.join(data_path + "mn40_depth_views")

if not os.path.isdir(save_path):
    os.mkdir(save_path)

data_list_file = os.path.join(data_path, 'modelnet40_normal_resampled/modelnet40_train.txt')
test_data_list_file = os.path.join(data_path, 'modelnet40_normal_resampled/modelnet40_test.txt')

with open(data_list_file, 'r') as f:
        lines = f.readlines()

num_processes = 1
start_id = 0
num_sub_files = int(len(lines)/num_processes)
file_list = lines
jobs = []
for i in range(num_processes):
	file_list_sub = file_list[start_id: start_id+num_sub_files] 
	dict_input = {'file_list':file_list_sub, 'thread_id' :i}
	p = multiprocessing.Process(target=do_augment, args = (dict_input,))
	jobs.append(p)
	p.start()
	start_id += num_sub_files



# file_list_sub = file_list[start_id:]
# do_augment(lines,rate)


