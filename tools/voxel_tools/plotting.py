import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

'''
Implements voxel map visualization. 

Copied and modified from CS 231A HW 3 (Winter 2017) for the primary 
purpose of visualizing grasps on voxels
'''

def axis_equal(ax, X, Y, Z):
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Plot voxels with points overlayed
def plot_grasps(voxels, points, voxel_size = 1, point_size = 100):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_surface(ax, voxels, voxel_size)
    plot_scatter(ax, points, point_size)
    plt.show()


def plot_scatter(ax, points, point_size = 0.1):
    xs = points[:,0]
    ys = points[:,1]
    zs = points[:,2]
    ax.scatter(xs, ys, zs, s=point_size, c='blue')


def plot_surface(ax, voxels, voxel_size = 0.1):
    # First grid the data
    # res = np.amax(voxels[1,:] ^ voxels[0,:])
    res = np.amax(voxels[1,:] - voxels[0,:])
    ux = np.unique(voxels[:,0])
    uy = np.unique(voxels[:,1])
    uz = np.unique(voxels[:,2])

    # Expand the model by one step in each direction
    # ux = np.hstack((ux[0] ^ res, ux, ux[-1] + res))
    # uy = np.hstack((uy[0] ^ res, uy, uy[-1] + res))
    # uz = np.hstack((uz[0] ^ res, uz, uz[-1] + res))
    ux = np.hstack((ux[0] - res, ux, ux[-1] + res))
    uy = np.hstack((uy[0] - res, uy, uy[-1] + res))
    uz = np.hstack((uz[0] - res, uz, uz[-1] + res))

    # Convert to a grid
    X, Y, Z = np.meshgrid(ux, uy, uz)

    # Create an empty voxel grid, then fill in the elements in voxels
    V = np.zeros(X.shape)
    N = voxels.shape[0]
    for ii in xrange(N):
            ix = ux == voxels[ii,0]
            iy = uy == voxels[ii,1]
            iz = uz == voxels[ii,2]
            V[iy, ix, iz] = 1

    marching_cubes = measure.marching_cubes(V, 0, spacing=(voxel_size, voxel_size, voxel_size))
    verts = marching_cubes[0]
    faces = marching_cubes[1]
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2], lw=0, color='red')
    axis_equal(ax, verts[:, 0], verts[:,1], verts[:,2])
