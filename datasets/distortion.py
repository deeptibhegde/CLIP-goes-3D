import pygem
from pygem import FFD, RBF, IDW
import open3d as o3d
import copy
import numpy as np
np.random.seed(2021)


def core_distortion(points, n_control_points=[2,2,2], displacement=None):
    """
        Ref: http://mathlab.github.io/PyGeM/tutorial-1-ffd.html
    """
    # the size of displacement matrix: 3 * control_points.shape
    if displacement is None:
        displacement = np.zeros((3,*n_control_points))

    ffd = FFD(n_control_points=n_control_points)
    ffd.box_length = [2.,2.,2.]
    ffd.box_origin = [-1., -1., -1.]
    ffd.array_mu_x = displacement[0,:,:,:]
    ffd.array_mu_y = displacement[1,:,:,:]
    ffd.array_mu_z = displacement[2,:,:,:]
    new_points = ffd(points)

    return new_points


def distortion(points, direction_mask=np.array([1,1,1]), point_mask=np.ones((5,5,5)), severity=0.5):

    
    n_control_points=[5,5,5]
    # random
    displacement = np.random.rand(3,*n_control_points) * 2 * severity - np.ones((3,*n_control_points)) * severity
    displacement *= np.transpose(np.tile(direction_mask, (5, 5, 5, 1)), (3, 0, 1, 2))
    displacement *= np.tile(point_mask, (3, 1, 1, 1))
    
    points = core_distortion(points, n_control_points=n_control_points, displacement=displacement)
    
    # points = denomalize(points, scale, offset)
    # set_points(data, points)
    return points


def distortion_2(points, severity=(0.4,3), func = 'gaussian_spline'):

    rbf = RBF(func=func)
    xv = np.linspace(-1, 1, severity[1])
    yv = np.linspace(-1, 1, severity[1])
    zv = np.linspace(-1, 1, severity[1])
    z, y, x = np.meshgrid(zv, yv, xv)
    mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    rbf.original_control_points = mesh
    alpha = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    gamma = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    distance = np.ones(mesh.shape[0]) * severity[0]
    displacement_x = distance * np.cos(alpha) * np.sin(gamma)
    displacement_y = distance * np.sin(alpha) * np.sin(gamma)
    displacement_z = distance * np.cos(gamma)
    displacement = np.array([displacement_x,displacement_y,displacement_z]).T
    rbf.deformed_control_points = mesh + displacement
    new_points = rbf(points)
    return new_points


def distortion_3(points, severity=(0.4,3)):

    idw = IDW()
    xv = np.linspace(-1, 1, severity[1])
    yv = np.linspace(-1, 1, severity[1])
    zv = np.linspace(-1, 1, severity[1])
    z, y, x = np.meshgrid(zv, yv, xv)
    mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    idw.original_control_points = mesh
    alpha = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    gamma = np.random.uniform(-np.pi,np.pi,mesh.shape[0])
    distance = np.ones(mesh.shape[0]) * severity[0]
    displacement_x = distance * np.cos(alpha) * np.sin(gamma)
    displacement_y = distance * np.sin(alpha) * np.sin(gamma)
    displacement_z = distance * np.cos(gamma)
    displacement = np.array([displacement_x,displacement_y,displacement_z]).T
    idw.deformed_control_points = mesh + displacement
    new_points = idw(points)
    return new_points

