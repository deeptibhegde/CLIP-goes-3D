###  Generate Various Common Corruptions ###
from operator import index
import os
import h5py
import json
import numpy as np
from numpy import random
from convert import *
import distortion
# from occlusion import *
from util import *
np.random.seed(2021)


### Transformation ###
'''
Rotate the point cloud
'''
def rotation(pointcloud,severity):
    N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity-1]
    theta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
    gamma = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.
    beta = np.random.uniform(c-2.5,c+2.5) * np.random.choice([-1,1]) * np.pi / 180.

    matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
    
    new_pc = np.matmul(pointcloud,matrix_1)
    new_pc = np.matmul(new_pc,matrix_2)
    new_pc = np.matmul(new_pc,matrix_3).astype('float32')

    return normalize(new_pc)

'''
Shear the point cloud
'''
def shear(pointcloud,severity):
    N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity-1]
    a = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    b = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    d = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    e = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    f = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])
    g = np.random.uniform(c-0.05,c+0.05) * np.random.choice([-1,1])

    matrix = np.array([[1,0,b],[d,1,e],[f,0,1]])
    new_pc = np.matmul(pointcloud,matrix).astype('float32')
    return normalize(new_pc)

'''
Scale the point cloud
'''
def scale(pointcloud,severity):
    #TODO
    N, C = pointcloud.shape
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
    a=b=d=1
    r = np.random.randint(0,3)
    t = np.random.choice([-1,1])
    if r == 0:
        a += c * t
        b += c * (-t)
    elif r == 1:
        b += c * t
        d += c * (-t)
    elif r == 2:
        a += c * t
        d += c * (-t)

    matrix = np.array([[a,0,0],[0,b,0],[0,0,d]])
    new_pc = np.matmul(pointcloud,matrix).astype('float32')
    return normalize(new_pc)


### Noise ###
'''
Add Uniform noise to point cloud 
'''
def uniform_noise(pointcloud, severity):
    #TODO
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity-1]
    jitter = np.random.uniform(-c,c,(N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return normalize(new_pc)

'''
Add Gaussian noise to point cloud 
'''
def gaussian_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity-1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    new_pc = np.clip(new_pc,-1,1)
    return new_pc

'''
Add noise to the edge-length-2 cude
'''
def background_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//45, N//40, N//35, N//30, N//20][severity-1]
    jitter = np.random.uniform(-1,1,(c, C))
    new_pc = np.concatenate((pointcloud,jitter),axis=0).astype('float32')
    return normalize(new_pc)

'''
Upsampling
'''
def upsampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//5, N//4, N//3, N//2, N][severity-1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05,0.05,(c, C))
    new_pc = np.concatenate((pointcloud,add),axis=0).astype('float32')
    return normalize(new_pc)
    
'''
Add impulse noise
'''
def impulse_noise(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//30, N//25, N//20, N//15, N//10][severity-1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    pointcloud[index] += np.random.choice([-1,1], size=(c,C)) * 0.1
    return normalize(pointcloud)
    

### Point Number Modification ###
'''
Cutout several part in the point cloud
'''
def cutout(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(2,30), (3,30), (5,30), (7,30), (10,30)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # pointcloud[idx.squeeze()] = 0
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    # print(pointcloud.shape)
    return pointcloud

'''
Uniformly sampling the point cloud
'''
def uniform_sampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N//15, N//10, N//8, N//6, N//2, 3 * N//4][severity-1]
    index = np.random.choice(ORIG_NUM, ORIG_NUM - c, replace=False)
    return pointcloud[index]

'''
Density-based up-sampling the point cloud
'''
def density_inc(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    # idx = np.random.choice(N,c[0])
    temp = []
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        # idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
        # idx = idx[idx_2]
        temp.append(pointcloud[idx.squeeze()])
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
    
    idx = np.random.choice(pointcloud.shape[0],1024 - c[0] * c[1])
    temp.append(pointcloud[idx.squeeze()])

    pointcloud = np.concatenate(temp)
    # print(pointcloud.shape)
    return pointcloud

'''
Density-based sampling the point cloud
'''
def density(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(1,100), (2,100), (3,100), (4,100), (5,100)][severity-1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0],1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked)**2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        idx_2 = np.random.choice(c[1],int((3/4) * c[1]),replace=False)
        idx = idx[idx_2]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # pointcloud[idx.squeeze()] = 0
    # print(pointcloud.shape)
    return pointcloud

def occlusion(severity):
    ## severity here does not stand for real severity ##
    pointcloud = []
    f_0 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json")
    f_1 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_1_id2file.json")
    lsit_0 = json.load(f_0)
    lsit_1 = json.load(f_1)
    f_0.close()
    f_1.close()

    for item in lsit_0 + lsit_1:
        folder = item.split('/')[0]
        mesh = item.split('/')[1][:-3] + 'off'
        # print(mesh)
        original_data = load_mesh("./data/ModelNet40/" + folder + "/test/" + mesh)
        new_pc = occlusion_1(original_data,'occlusion',severity,n_points=1024)

        theta =  -np.pi / 2.
        gamma =  0
        beta = np.pi

        matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
        
        new_pc = np.matmul(new_pc,matrix_1)
        new_pc = np.matmul(new_pc,matrix_2)
        new_pc = normalize(np.matmul(new_pc,matrix_3).astype('float32'))

        pointcloud.append(new_pc)

    pointcloud = np.stack(pointcloud,axis=0)

    np.save("./data/modelnet40_c/data_occlusion_" + str(severity) + ".npy", pointcloud)
    return

def simulate_lidar(pointcloud,pose,severity):
    pose = pose.transpose()
    #####################################
    # simplify the rotation to I matrix #
    pose[:3,:3] = 0
    pose[0,0] = pose[1,1] = pose[2,2] = 1 
    # Translate the point cloud #
    pose[3,[0,1,2]] = -pose[3,[0,1,2]] 
    #####################################

    pointcloud_new = np.concatenate([pointcloud,np.ones((pointcloud.shape[0],1))],axis=1)
    pointcloud_new = np.dot(pointcloud_new,pose)

    pointcloud_new = appendSpherical_np(pointcloud_new[:,:3])
    delta = 1. * np.pi / 180.
    cur = np.min(pointcloud_new[:,4])

    new_pc = []
    
    while cur + delta < np.max(pointcloud_new[:4]):
        pointcloud_new[(pointcloud_new[:,4] >= cur+delta/4) & (pointcloud_new[:,4] < cur + delta*3/4),4] = cur + delta / 2.
        new_pc.append(pointcloud_new[(pointcloud_new[:,4] >= cur+delta/4) & (pointcloud_new[:,4] < cur + delta*3/4)])
        cur += delta
    new_pc = np.concatenate(new_pc,axis=0)
    # pointcloud = np.dot(pointcloud,np.linalg.inv(pose))
    new_pc = appendCart_np(new_pc[:,3:])
    new_pc = np.concatenate([new_pc[:,3:],np.ones((new_pc.shape[0],1))],axis=1)
    new_pc = np.dot(new_pc,np.linalg.inv(pose))
    index = np.random.choice(new_pc.shape[0],768)
    new_pc = new_pc[index]
    return new_pc[:,:3]

def lidar(severity):
    ## severity here does not stand for real severity ##
    pointcloud = []
    f_0 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json")
    f_1 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_1_id2file.json")
    lsit_0 = json.load(f_0)
    lsit_1 = json.load(f_1)
    f_0.close()
    f_1.close()

    for item in lsit_0 + lsit_1:
        folder = item.split('/')[0]
        mesh = item.split('/')[1][:-3] + 'off'
        original_data = load_mesh("./data/ModelNet40/" + folder + "/test/" + mesh)
        new_pc,pose = occlusion_1(original_data,'lidar',severity,n_points=1024)

        new_pc = simulate_lidar(new_pc,pose,severity)

        theta =  -np.pi / 2.
        gamma =  0
        beta = np.pi

        matrix_1 = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma),0,np.sin(gamma)],[0,1,0],[-np.sin(gamma),0,np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta),-np.sin(beta),0],[np.sin(beta),np.cos(beta),0],[0,0,1]])
        
        new_pc = np.matmul(new_pc,matrix_1)
        new_pc = np.matmul(new_pc,matrix_2)
        new_pc = np.matmul(new_pc,matrix_3).astype('float32')


        pointcloud.append(new_pc)

    pointcloud = np.stack(pointcloud,axis=0)

    np.save("./data/modelnet40_c/data_lidar_" + str(severity) + ".npy", pointcloud)
    return


def ffd_distortion(pointcloud, severity):
    N, C = pointcloud.shape
    c = [0.1,0.2,0.3,0.4,0.5][severity-1]
    new_pc = distortion.distortion(pointcloud,severity=c)
    return normalize(new_pc)

def rbf_distortion(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')

def rbf_distortion_inv(pointcloud, severity):
    N, C = pointcloud.shape
    c = [(0.025,5),(0.05,5),(0.075,5),(0.1,5),(0.125,5)][severity-1]
    new_pc = distortion.distortion_2(pointcloud,severity=c,func='inv_multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32')



def load_data():
    os.makedirs("./data/modelnet40_c",exist_ok = True)
    modelnet40_dir = "./data/modelnet40_ply_hdf5_2048/"
    modelnet40_test_file = os.path.join(modelnet40_dir, "test_files.txt")
    with open(modelnet40_test_file, "r") as f:
        modelnet40_test_paths = [l.strip() for l in f.readlines()]

    data   = []
    labels = []
    for modelnet40_test_path in modelnet40_test_paths:
        test_h5 = h5py.File(modelnet40_test_path, "r")

        data.append(test_h5["data"][:])
        labels.append(test_h5["label"][:])

    data   = np.concatenate(data)
    labels = np.concatenate(labels)

    np.save("./data/modelnet40_c/label.npy", labels)

    return data, labels


def save_data(data,corruption,severity):

    if not MAP[corruption]:
        np.save("./data/modelnet40_c/data_" + corruption + ".npy", data)
        return
        
    new_data = []
    for i in range(data.shape[0]):
        if corruption in ['occlusion', 'lidar']:
            new_data.append(MAP[corruption](severity))
        else:
            new_data.append(MAP[corruption](data[i],severity))
    new_data = np.stack(new_data,axis=0)
    np.save("./data/modelnet40_c/data_" + corruption + "_" + str(severity) + ".npy", new_data)


# MAP = {'uniform': uniform_noise,
#        'gaussian': gaussian_noise,
#        'background': background_noise,
#        'impulse': impulse_noise,
#        'scale': scale,
#        'upsampling': upsampling,
#        'shear': shear,
#        'rotation': rotation,
#        'cutout': cutout,
#        'density': density,
#        'density_inc': density_inc,
#        'distortion': ffd_distortion,
#        'distortion_rbf': rbf_distortion,
#        'distortion_rbf_inv': rbf_distortion_inv,
#     #    'occlusion': occlusion,
#     #    'lidar': lidar,
#     #    'original': None,
# }

ORIG_NUM = 8192

if __name__ == "__main__":
    data, labels = load_data()
    for cor in MAP.keys():
        # if cor in ['occlusion', 'lidar']:
        #     continue
        for sev in [1,2,3,4,5]:
            if cor == 'density_inc':
                ORIG_NUM = 2048
            else:
                ORIG_NUM = 1024
            index = np.random.choice(data.shape[1],ORIG_NUM,replace=False)
            save_data(data[:,index,:], cor, sev)
            print("Done with Corruption: {} with Severity: {}".format(cor,sev))

