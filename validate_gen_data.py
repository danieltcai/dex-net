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
data = f['data'][:]
label = f['label'][:]

# Debug
# assert len(data) == len(label)
# print(len(data))
# print(len(label))

# Visualize each datapoint
for i in range(data_start, len(data)):
    '''
    what's read should match what's written
    point_cloud_data = np.reshape(point_cloud_data, [-1, 2048, 3]) # (num_point_clouds x points_per_cloud x 3)
    truncated_grasp_labels = np.reshape(truncated_grasp_labels, [-1, num_grasps_to_keep, 7]) # (num_point_clouds x num_grasps_per_cloud x 7) CONFIRM THIS DOES WHAT YOU WANT
    '''
    point_cloud = data[i]
    grasps = label[i]

    non_robust_grasps = grasps[np.where(grasps[:,0] <= 0)]
    robust_grasps = grasps[np.where(grasps[:,0] > 0)]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc_xs = point_cloud[:,0]
    pc_ys = point_cloud[:,1]
    pc_zs = point_cloud[:,2]

    ng_xs = non_robust_grasps[:,1]
    ng_ys = non_robust_grasps[:,2]
    ng_zs = non_robust_grasps[:,3]

    rg_xs = robust_grasps[:,1]
    rg_ys = robust_grasps[:,2]
    rg_zs = robust_grasps[:,3]

    ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud
    ax.scatter(ng_xs, ng_ys, ng_zs, s=25, c='red') # Non robust grasp positions
    ax.scatter(rg_xs, rg_ys, rg_zs, s=50, c='green') # Robust grasp positions
    plt.show()
