import open3d as o3d
import numpy as np
import copy


def get_points(data):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        return np.asarray(data.vertices)
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        return np.asarray(data.points)
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def set_points(data, points):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        data.vertices = o3d.utility.Vector3dVector(points)
        return data
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        data.points = o3d.utility.Vector3dVector(points)
        return data
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def normalize(new_pc):
    new_pc[:,0] -= (np.max(new_pc[:,0]) + np.min(new_pc[:,0])) / 2
    new_pc[:,1] -= (np.max(new_pc[:,1]) + np.min(new_pc[:,1])) / 2
    new_pc[:,2] -= (np.max(new_pc[:,2]) + np.min(new_pc[:,2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:,0]) - np.min(new_pc[:,0]), np.max(new_pc[:,1]) - np.min(new_pc[:,1]), np.max(new_pc[:,2]) - np.min(new_pc[:,2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc


def denomalize(points, scale, offset, hard_copy=False):
    if hard_copy:
        new_points = copy.deepcopy(points)
    else:
        new_points = points

    n_points = new_points.shape[0]
    new_points = new_points * np.tile(scale, (n_points,1)) + np.tile(offset, (n_points,1))
    return new_points

def shuffle_data(data):

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...]


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew

def appendCart_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    ptsnew[:,3] = ptsnew[:,0] * np.sin(ptsnew[:,1]) * np.cos(ptsnew[:,2])
    ptsnew[:,4] = ptsnew[:,0] * np.sin(ptsnew[:,1]) * np.sin(ptsnew[:,2])
    ptsnew[:,5] = ptsnew[:,0] * np.cos(ptsnew[:,1]) 
    return ptsnew

    