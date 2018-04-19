'''
Script to read and write .binvox format using numpy.

Currently supports read and visualization operations.

Utilizes code found at: https://github.com/dimatura/binvox-rw-py
and CS 231A HW3 code.

TODO: support voxel writing for grasp metric heatmap generation. Refactor the code
to be more useful and modular

'''

import binvox_rw_py.binvox_rw as bv
import numpy as np
import argparse
from space_carving.plotting import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'binvox_path',
        type=str,
        help='Path to the .binvox file')

    parser.add_argument(
        '--sparse',
        dest='use_sparse',
        type=bool,
        help='whether to read .binvox using sparse representation',
        default=True
        )

    args = parser.parse_args()
    return args


def plot_sparse(binvox_path):
    '''
    Plots the voxel map using bv.read_as_coord_array(), 
    which returns a sparse voxel map as an 3 x N np array.

    This format is closest to the input of plot_surface(),
    so by default we use this method.
    '''
    with open(binvox_path, 'rb') as fin:
        model = bv.read_as_coord_array(fin) 
        voxels = model.data.T 
        plot_surface(voxels) 


def diff(x):
    return x[1] - x[0]


def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    '''
    Form initial voxels given constraints.

    Copied directly from CS 231A HW3, so may be slightly overkill,
    since we just want to mirror the dimensions of the voxel map,
    i.e. side_length = 1 and num_voxels = (n_x, n_y, n_z) of input.
    '''

    # Total volume / approx number of voxels = voxel volume
    # Then get the side length for one voxel (cube)
    total_volume = diff(xlim) * diff(ylim) * diff(zlim)
    voxel_volume = float(total_volume) / float(num_voxels)
    side_length = np.cbrt(voxel_volume)

    # Enumerate each dimension with limits and step size
    x = np.arange(xlim[0], xlim[1], side_length)
    y = np.arange(ylim[0], ylim[1], side_length)
    z = np.arange(zlim[0], zlim[1], side_length)

    # Repeat and tile as necessary to form 3D grid
    x_col = np.repeat(x, len(y) * len(z))
    y_col = np.tile(np.repeat(y, len(z)), len(x))
    z_col = np.tile(z, len(x) * len(y))

    # Stack and ensure right dimensions
    voxels = np.vstack((x_col, y_col, z_col))
    voxels = voxels.T

    return voxels, side_length


def carve(voxels, voxel_bool):
    '''
    Carves initial voxels according to voxel_bool
    '''
    new_voxels = []

    # Keep only the voxels that are marked True in voxel_bool
    for voxel in voxels:
        x,y,z = np.array(voxel, dtype=int)
        if voxel_bool[x,y,z]:
            new_voxels.append(voxel)

    new_voxels = np.array(new_voxels)
    return new_voxels


def plot_dense(binvox_path):
    '''
    Plots the voxel map using bv.read_as_3d_array(),
    which returns a dense voxel map as a (n_x, n_y, n_z)
    np array of boolean values, denoting whether there exists
    a voxel at the given coordinate.

    Requires more maneuvering to plot but may be useful in the 
    future if we want to directly work with 3D coordinates. 
    '''
    with open(binvox_path, 'rb') as fin:
        model = bv.read_as_3d_array(fin)
        voxel_bool = model.data # (n_x, n_y, n_z) boolean np array

        # Form initial voxels, mirroring the dimensions of the .binvox input
        n_x, n_y, n_z = model.dims
        num_voxels = n_x * n_y * n_z
        xlim = (0, n_x) 
        ylim = (0, n_y)
        zlim = (0, n_z)
        voxels, side_length = form_initial_voxels(xlim, ylim, zlim, num_voxels)
        
        # Carve the initial voxels according to the .binvox input
        voxels = carve(voxels, voxel_bool)
        plot_surface(voxels)

def main():
    args = parse_args()
    binvox_path = args.binvox_path
    use_sparse = args.use_sparse    

    if use_sparse:
        plot_sparse(binvox_path)
    else:
        plot_dense(binvox_path)

    print("Exit gracefully")


if __name__ == '__main__':
    main()
    


        # print(type(model.data))
        # plot_surface(model.data)



        # with open('test.binvox', 'wb') as fout:
        #     model.write(fout)

        # with open('test.binvox', 'rb') as fin2:
        #     model2 = bv.read_as_3d_array(fin2)
        #     print(model2.dims)
        #     print(model2.scale)
        #     print(model2.translate)

        #     print(np.all(model.data == model2.data))




# def carve(voxels, camera):
#     new_voxels = []

#     # For each voxel, if projection into image is in silhouette, keep the voxel 
#     for voxel in voxels:
#         # print voxel
#         pixel = np.dot(camera.P, homogenous(voxel))
#         assert pixel.shape == (3,)
#         # Calculate voxel's corresponding image pixels
#         x = int(pixel[0]/pixel[2])
#         y = int(pixel[1]/pixel[2])
#         # Skip voxel if reprojected pixels are outside the image
#         if x >= camera.silhouette.shape[1] or y >= camera.silhouette.shape[0]\
#             or x < 0 or y < 0: continue
#         # Add voxel if reprojected pixels are in the silhouette
#         if camera.silhouette[y, x] == 1:
#             new_voxels.append(voxel)
#     new_voxels = np.array(new_voxels)
#     return new_voxels