# from data_prep_util import *
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)
'''

# Parse args
parser = argparse.ArgumentParser(description='Validate generated training data through visualization')
parser.add_argument('h5_filename', type=str, default=None, help='path to h5 data file')
parser.add_argument('--data_start', type=int, default=0, help='data point to begin visualization')
args = parser.parse_args()
h5_filename = args.h5_filename
data_start = args.data_start

# Read in h5 file
f = h5py.File(h5_filename)
point_cloud_data = f['data'][:]
grasp_labels = f['label'][:]

# Debug
# assert len(data) == len(label)
# print(len(data))
# print(len(label))

# Visualize each training example
for i in range(data_start, len(point_cloud_data)):
    '''
    what's read should match what's written
    point_cloud_data = np.reshape(point_cloud_data, [-1, num_points, 3]) # (num_eg, num_points, 3)
    grasp_labels = np.reshape(grasp_labels, [-1, num_points, 7]) # (num_eg, num_points, 7)
    '''
    point_cloud = point_cloud_data[i] # (num_points, 3)
    point_labels = grasp_labels[i] # (num_points, 7)

    # Check dimensions
    assert point_cloud.shape[0] == point_labels.shape[0]
    assert point_cloud.shape[1] == 3
    assert point_labels.shape[1] == 7

    # Check label of each point
    for i, point_label in enumerate(point_labels):
        # Plot point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pc_xs = point_cloud[:,0]
        pc_ys = point_cloud[:,1]
        pc_zs = point_cloud[:,2]
        ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud

        # Plot point
        point = point_cloud[i]
        ax.scatter(point[0], point[1], point[2], s=50, c='purple') # Object point cloud

        # Plot nearest grasp according to label
        rg_x = point_label[1]
        rg_y = point_label[2]
        rg_z = point_label[3]

        exists_near_robust = point_label[0]
        if exists_near_robust:
            ax.scatter(rg_x, rg_y, rg_z, s=50, c='green') # Nearby robust grasp positions
        else: 
            ax.scatter(rg_x, rg_y, rg_z, s=50, c='red') # Nearest but still too far robust grasp positions

        plt.show()
