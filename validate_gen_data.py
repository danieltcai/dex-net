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
def rotation_matrix(alpha, beta, gamma):
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(alpha), -np.sin(alpha)],
                   [0,  np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # R = Rz * Ry * Rx
    R = np.dot(np.dot(Rz, Ry), Rx)
    # R = np.dot(Rz, np.dot(Ry, Rx))
    return R


# Parse args
parser = argparse.ArgumentParser(description='Validate generated training data through visualization')
parser.add_argument('h5_filename', type=str, default=None, help='path to h5 data file')
parser.add_argument('--data_start', type=int, default=0, help='data point to begin visualization')
args = parser.parse_args()
h5_filename = args.h5_filename
data_start = args.data_start

# All training examples in file
f = h5py.File(h5_filename)
point_cloud_data = f['data'][:]
grasp_labels = f['label'][:]

assert len(point_cloud_data) == len(grasp_labels)
print("Num training examples in file: {}".format(len(point_cloud_data)))

# For each training example
for eg_idx in range(data_start, len(point_cloud_data)):
    '''
    what's read should match what's written
    point_cloud_data = np.reshape(point_cloud_data, [-1, num_points, 3]) # (num_eg, num_points, 3)
    grasp_labels = np.reshape(grasp_labels, [-1, num_points, 7]) # (num_eg, num_points, 7)
    '''
    point_cloud = point_cloud_data[eg_idx] # (num_points, 3)
    point_labels = grasp_labels[eg_idx] # (num_points, 7)

    print("Label for example number: {}".format(eg_idx))
    print(point_labels)

    # Check dimensions
    assert point_cloud.shape[0] == point_labels.shape[0]
    assert point_cloud.shape[1] == 3
    assert point_labels.shape[1] == 7

    # Check label of each point
    for pt_idx, point_label in enumerate(point_labels):
        print("Label for point: {}".format(pt_idx))
        print(point_labels[pt_idx])

        # Plot point cloud
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pc_xs = point_cloud[:,0]
        pc_ys = point_cloud[:,1]
        pc_zs = point_cloud[:,2]
        ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud

        # Plot grasp position
        point = point_cloud[pt_idx]
        ax.scatter(point[0], point[1], point[2], s=50, c='purple') # Object point cloud

        # Plot nearest grasp according to label. Dont forget pos is offset from point
        grasp_x = point_label[1] + point[0] 
        grasp_y = point_label[2] + point[1]
        grasp_z = point_label[3] + point[2]
        grasp_pos = np.reshape([grasp_x, grasp_y, grasp_z], [3,1])

        # Plot grasp axes orientation
        alpha = point_label[4]
        beta = point_label[5]
        gamma = point_label[6]
        R = rotation_matrix(alpha, beta, gamma)

        x_axis = np.reshape([0.01,0,0], [3,1])
        x_axis = np.dot(R, x_axis) + grasp_pos
        y_axis = np.reshape([0,0.01,0], [3,1])
        y_axis = np.dot(R, y_axis) + grasp_pos
        z_axis = np.reshape([0,0,0.01], [3,1])
        z_axis = np.dot(R, z_axis) + grasp_pos



        exists_near_robust = point_label[0]
        if exists_near_robust:
            ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='green') # Nearby robust grasp positions
        else: 
            ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='red') # Nearest but still too far robust grasp positions

        ax.scatter(x_axis[0], x_axis[1], x_axis[2], s=50, c='red') # Nearest but still too far robust grasp positions
        ax.scatter(y_axis[0], y_axis[1], y_axis[2], s=50, c='green') # Nearest but still too far robust grasp positions
        ax.scatter(z_axis[0], z_axis[1], z_axis[2], s=50, c='blue') # Nearest but still too far robust grasp positions
        plt.show()
