# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Generates datasets of synthetic point clouds, grasps, and grasp robustness metrics from a Dex-Net HDF5 database for GQ-CNN training.

Author
------
Jeff Mahler

YAML Configuration File Parameters
----------------------------------
database_name : str
    full path to a Dex-Net HDF5 database
target_object_keys : :obj:`OrderedDict`
    dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
env_rv_params : :obj:`OrderedDict`
    parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
gripper_name : str
    name of the gripper to use

--------------------------------------------------------------------------------

THIS FILE HAS BEEN MODIFIED FROM THE ORIGINAL generate_gqcnn_dataset.py TO:
- Write file containing corresponding object mesh names for future voxel heatmap generation


Below is an excerpt from the DexNet 2.0 training dataset readme
   1) depth_ims_tf_table:
       Description: depth images transformed to align the grasp center with the image center and the grasp axis with the middle row of pixels
       File dimension: 1000x32x32x1 (except the last file)
       Organization: {num_datapoints} x {image_height} x {image_width} x {num_channels}
       Notes: Rendered with OSMesa using the parameters of a Primesense Carmine 1.08

  2) hand_poses:
       Description: configuration of the robot gripper corresponding to the grasp
       File dimension: 1000x7 (except the last file)
       Organization: {num_datapoints} x {hand_configuration}, where columns are
         0: row index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         1: column index, in pixels, of grasp center projected into a depth image centered on the object (pre-rotation-and-translation)
         2: depth, in meters, of gripper center from the camera that took the corresponding depth image
     3: angle, in radians, of the grasp axis from the image x-axis (middle row of pixels, pointing right in image space)
     4: row index, in pixels, of the object center projected into a depth image centered on the world origin
     5: column index, in pixels, of the object center projected into a depth image centered on the world origin
     6: width, in pixels, of the gripper projected into the depth image
       Notes: To replicate the Dex-Net 2.0 results, you only need column 2.
         The gripper width was 5cm, corresponding to the width of our custom ABB YuMi grippers

  3) robust_ferrari_canny:
       Description: value of the robust epsilon metric computed according to the Dex-Net 2.0 graphical model
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Threshold the values in this value by 0.002 to generate the 0-1 labels of Dex-Net 2.0

  4) ferrari_canny:
       Description: value of the epsilon metric, without measuring robustness to perturbations in object pose, gripper pose, and friction
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Not used in the Dex-Net 2.0 paper. Included for comparison purposes

  5) force_closure:
       Description: value of force closure, without measuring robustness to perturbations in object pose, gripper pose, and friction
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
       Notes: Not used in the Dex-net 2.0 paper. Included for comparison purposes


Below is a proposed additional file to be written during the data generation phase:
  6) mesh_filename:
       Description: name of mesh file used to generate training data. Used when creating a voxelized heatmap of grasp success
       File dimension: 1000 (except the last file)
       Organization: {num_datapoints}
"""
import argparse
import collections
import cPickle as pkl
import gc
import IPython
import json
import logging
import numpy as np
import os
import random
import shutil
import sys
import time

from autolab_core import Point, RigidTransform, YamlConfig
import autolab_core.utils as utils
from gqcnn import Grasp2D
from gqcnn import Visualizer as vis2d
from meshpy import ObjFile, RenderMode, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from perception import CameraIntrinsics, BinaryImage, DepthImage

from dexnet.constants import READ_ONLY_ACCESS
from dexnet.database import Hdf5Database
from dexnet.grasping import GraspCollisionChecker, RobotGripper
from dexnet.learning import TensorDataset

try:
    from dexnet.visualization import DexNetVisualizer3D as vis
except:
    logging.warning('Failed to import DexNetVisualizer3D, visualization methods will be unavailable')


# Voxel tools
import voxel_tools.plotting as vplot
import voxel_tools.binvox_rw_py.binvox_rw as bv
import subprocess
import matplotlib.pyplot as plt

from depth_image_renderer import Depth_Image_Renderer


# h5py data saver
import h5py

logging.root.name = 'dex-net'

# seed for deterministic behavior when debugging
SEED = 197561

# name of the grasp cache file
CACHE_FILENAME = 'grasp_cache.pkl'

class GraspInfo(object):
    """ Struct to hold precomputed grasp attributes.
    For speeding up dataset generation.
    """
    def __init__(self, grasp, collision_free, phi=0.0):
        self.grasp = grasp
        self.collision_free = collision_free
        self.phi = phi


# Write numpy array data and label to h5_filename
# Copied from pointnet data_prep_util.py
def save_h5(h5_filename, data, label, data_dtype='float32', label_dtype='float32'):
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


def generate_gqcnn_dataset(dataset_path,
                           hdf5_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config):
    """
    Generates a GQ-CNN TensorDataset for training models with new grippers, quality metrics, objects, and cameras.

    Parameters
    ----------
    dataset_path : str
        path to save the dataset to
    database : :obj:`Hdf5Database`
        Dex-Net database containing the 3D meshes, grasps, and grasp metrics
    target_object_keys : :obj:`OrderedDict`
        dictionary mapping dataset names to target objects (either 'all' or a list of specific object keys)
    env_rv_params : :obj:`OrderedDict`
        parameters of the camera and object random variables used in sampling (see meshpy.UniformPlanarWorksurfaceImageRandomVariable for more info)
    gripper_name : str
        name of the gripper to use
    config : :obj:`autolab_core.YamlConfig`
        other parameters for dataset generation

    Notes
    -----
    Required parameters of config are specified in Other Parameters

    Other Parameters
    ----------------    
    images_per_stable_pose : int
        number of object and camera poses to sample for each stable pose
    stable_pose_min_p : float
        minimum probability of occurrence for a stable pose to be used in data generation (used to prune bad stable poses
    
    gqcnn/crop_width : int
        width, in pixels, of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/crop_height : int
        height, in pixels,  of crop region around each grasp center, before resize (changes the size of the region seen by the GQ-CNN)
    gqcnn/final_width : int
        width, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)
    gqcnn/final_height : int
        height, in pixels,  of final transformed grasp image for input to the GQ-CNN (defaults to 32)

    table_alignment/max_approach_table_angle : float
        max angle between the grasp axis and the table normal when the grasp approach is maximally aligned with the table normal
    table_alignment/max_approach_offset : float
        max deviation from perpendicular approach direction to use in grasp collision checking
    table_alignment/num_approach_offset_samples : int
        number of approach samples to use in collision checking

    collision_checking/table_offset : float
        max allowable interpenetration between the gripper and table to be considered collision free
    collision_checking/table_mesh_filename : str
        path to a table mesh for collision checking (default data/meshes/table.obj)
    collision_checking/approach_dist : float
        distance, in meters, between the approach pose and final grasp pose along the grasp axis
    collision_checking/delta_approach : float
        amount, in meters, to discretize the straight-line path from the gripper approach pose to the final grasp pose

    tensors/datapoints_per_file : int
        number of datapoints to store in each unique tensor file on disk
    tensors/fields : :obj:`dict`
        dictionary mapping field names to dictionaries specifying the data type, height, width, and number of channels for each tensor

    debug : bool
        True (or 1) if the random seed should be set to enforce deterministic behavior, False (0) otherwise
    vis/candidate_grasps : bool
        True (or 1) if the collision free candidate grasps should be displayed in 3D (for debugging)
    vis/rendered_images : bool
        True (or 1) if the rendered images for each stable pose should be displayed (for debugging)
    vis/grasp_images : bool
        True (or 1) if the transformed grasp images should be displayed (for debugging)
    """

    if not os.path.isabs(hdf5_path):
        hdf5_path = os.path.join(os.getcwd(), hdf5_path)
    if os.path.exists(hdf5_path):
        logging.info('Save file already exists. Shutting down.')
        exit(0)



    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    # read data gen params
    output_dir = dataset_path
    gripper = RobotGripper.load(gripper_name)
    image_samples_per_stable_pose = config['images_per_stable_pose']
    stable_pose_min_p = config['stable_pose_min_p']
    
    # read gqcnn params
    gqcnn_params = config['gqcnn']
    im_crop_height = gqcnn_params['crop_height']
    im_crop_width = gqcnn_params['crop_width']
    im_final_height = gqcnn_params['final_height']
    im_final_width = gqcnn_params['final_width']
    cx_crop = float(im_crop_width) / 2
    cy_crop = float(im_crop_height) / 2

    # open database
    dataset_names = target_object_keys.keys()
    datasets = [database.dataset(dn) for dn in dataset_names]

    # set target objects
    for dataset in datasets:
        if target_object_keys[dataset.name] == 'all':
            target_object_keys[dataset.name] = dataset.object_keys

    # setup grasp params
    table_alignment_params = config['table_alignment']
    min_grasp_approach_offset = -np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_offset = np.deg2rad(table_alignment_params['max_approach_offset'])
    max_grasp_approach_table_angle = np.deg2rad(table_alignment_params['max_approach_table_angle'])
    num_grasp_approach_samples = table_alignment_params['num_approach_offset_samples']

    phi_offsets = []
    if max_grasp_approach_offset == min_grasp_approach_offset:
        phi_inc = 1
    elif num_grasp_approach_samples == 1:
        phi_inc = max_grasp_approach_offset - min_grasp_approach_offset + 1
    else:
        phi_inc = (max_grasp_approach_offset - min_grasp_approach_offset) / (num_grasp_approach_samples - 1)
                                                            
    phi = min_grasp_approach_offset
    while phi <= max_grasp_approach_offset:
        phi_offsets.append(phi)
        phi += phi_inc

    # setup collision checking
    coll_check_params = config['collision_checking']
    approach_dist = coll_check_params['approach_dist']
    delta_approach = coll_check_params['delta_approach']
    table_offset = coll_check_params['table_offset']

    table_mesh_filename = coll_check_params['table_mesh_filename']
    if not os.path.isabs(table_mesh_filename):
        table_mesh_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', table_mesh_filename)
    table_mesh = ObjFile(table_mesh_filename).read()
    
    # set tensor dataset config
    tensor_config = config['tensors']
    tensor_config['fields']['depth_ims_tf_table']['height'] = im_final_height
    tensor_config['fields']['depth_ims_tf_table']['width'] = im_final_width
    tensor_config['fields']['obj_masks']['height'] = im_final_height
    tensor_config['fields']['obj_masks']['width'] = im_final_width

    # add available metrics (assuming same are computed for all objects)
    metric_names = []
    dataset = datasets[0]
    obj_keys = dataset.object_keys
    if len(obj_keys) == 0:
        raise ValueError('No valid objects in dataset %s' %(dataset.name))
    
    obj = dataset[obj_keys[0]]
    # print ("PRE GRIP") # TODO: DELETE ME
    grasps = dataset.grasps(obj.key, gripper=gripper.name)
    # print ("PRE GRIP")# TODO: DELETE ME
    grasp_metrics = dataset.grasp_metrics(obj.key, grasps, gripper=gripper.name)
    metric_names = grasp_metrics[grasp_metrics.keys()[0]].keys()
    for metric_name in metric_names:
        tensor_config['fields'][metric_name] = {}
        tensor_config['fields'][metric_name]['dtype'] = 'float32'

    # init tensor dataset
    tensor_dataset = TensorDataset(output_dir, tensor_config)
    tensor_datapoint = tensor_dataset.datapoint_template

    # setup log file
    experiment_log_filename = os.path.join(output_dir, 'dataset_generation.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    hdlr = logging.FileHandler(experiment_log_filename)
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr)
    root_logger = logging.getLogger()

    # copy config
    out_config_filename = os.path.join(output_dir, 'dataset_generation.json')
    ordered_dict_config = collections.OrderedDict()
    for key in config.keys():
        ordered_dict_config[key] = config[key]
    with open(out_config_filename, 'w') as outfile:
        json.dump(ordered_dict_config, outfile)

    # 1. Precompute the set of valid grasps for each stable pose:
    #    i) Perpendicular to the table
    #   ii) Collision-free along the approach direction

    """
    Computes a dict of candidate grasps.

    Indexed as candidate_grasps_dict[obj.key][stable_pose.id] = list of 
    collision free grasps for a given object and stable pose

    For each dataset:
        For each object:
            Compute all stable poses
            For all stable poses:
                If stable pose is valid:
                    Compute list of grasps (ParallelJawPtGrasp3D objs,
                    see dex-net/grasping/grasp.py for definition)
                    Select only aligned grasps (those perpendicular to table)
                    For each aligned grasp:
                        Append to candidate_grasps_dict[obj.key][stable_pose.id]
                        a list of collision free grasps (list of GraspInfo objects)
                        with different rotation angles (phi)               
    """ 

    # load grasps if they already exist
    grasp_cache_filename = os.path.join(output_dir, CACHE_FILENAME)
    if os.path.exists(grasp_cache_filename):
        logging.info('Loading grasp candidates from file')
        candidate_grasps_dict = pkl.load(open(grasp_cache_filename, 'rb'))
    # otherwise re-compute by reading from the database and enforcing constraints
    else:        
        # create grasps dict
        candidate_grasps_dict = {}
        
        # loop through datasets and objects
        for dataset in datasets:
            logging.info('Reading dataset %s' %(dataset.name))
            for obj in dataset:
                if obj.key not in target_object_keys[dataset.name]:
                    continue

                # init candidate grasp storage
                candidate_grasps_dict[obj.key] = {}

                # setup collision checker
                collision_checker = GraspCollisionChecker(gripper)
                collision_checker.set_graspable_object(obj)

                # read in the stable poses of the mesh
                stable_poses = dataset.stable_poses(obj.key)
                for i, stable_pose in enumerate(stable_poses):
                    # render images if stable pose is valid
                    if stable_pose.p > stable_pose_min_p:
                        candidate_grasps_dict[obj.key][stable_pose.id] = []

                        # setup table in collision checker
                        T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                        T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=table_offset).as_frames('obj', 'table')
                        T_table_obj = T_obj_table.inverse()
                        collision_checker.set_table(table_mesh_filename, T_table_obj)

                        # read grasp and metrics
                        grasps = dataset.grasps(obj.key, gripper=gripper.name)
                        logging.info('Aligning %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))

                        # align grasps with the table
                        aligned_grasps = [grasp.perpendicular_table(stable_pose) for grasp in grasps]

                        # check grasp validity
                        logging.info('Checking collisions for %d grasps for object %s in stable %s' %(len(grasps), obj.key, stable_pose.id))
                        for aligned_grasp in aligned_grasps:
                            # check angle with table plane and skip unaligned grasps
                            _, grasp_approach_table_angle, _ = aligned_grasp.grasp_angles_from_stp_z(stable_pose)
                            perpendicular_table = (np.abs(grasp_approach_table_angle) < max_grasp_approach_table_angle)
                            if not perpendicular_table: 
                                continue

                            # check whether any valid approach directions are collision free
                            collision_free = False
                            for phi_offset in phi_offsets:
                                rotated_grasp = aligned_grasp.grasp_y_axis_offset(phi_offset)
                                collides = collision_checker.collides_along_approach(rotated_grasp, approach_dist, delta_approach)
                                if not collides:
                                    collision_free = True
                                    break
                    
                            # NEW
                            # Store if robust
                            candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp, collision_free))


                            # # store if aligned to table
                            # candidate_grasps_dict[obj.key][stable_pose.id].append(GraspInfo(aligned_grasp, collision_free))

                            # visualize if specified
                            if collision_free and config['vis']['candidate_grasps']:
                                logging.info('Grasp %d' %(aligned_grasp.id))
                                vis.figure()
                                vis.gripper_on_object(gripper, aligned_grasp, obj, stable_pose.T_obj_world)
                                vis.show()                  
        # save to file
        logging.info('Saving to file')
        pkl.dump(candidate_grasps_dict, open(grasp_cache_filename, 'wb'))

    # 2. Render a dataset of images and associate the gripper pose with image coordinates for each grasp in the Dex-Net database
    '''
    For each dataset:
        For each object key:
            For each stable pose:
                If stable pose is valid:
                    Read in candidate grasps and metrics
                    Sample images from the camera
                    For each sampled image:
                        Read image and calculate camera intrinsics
                        


    '''
    # setup variables
    obj_category_map = {}
    pose_category_map = {}

    cur_pose_label = 0
    cur_obj_label = 0
    cur_image_label = 0


    # setup data recording to HDF5 format NEW
    point_cloud_data = []
    grasp_labels = []

                
    # render images for each stable pose of each object in the dataset
    render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH_SCENE]
    for dataset in datasets:
        logging.info('Generating data for dataset %s' %(dataset.name))
        
        # iterate through all object keys
        object_keys = dataset.object_keys
        for obj_key in object_keys:
            obj = dataset[obj_key]
            if obj.key not in target_object_keys[dataset.name]:
                continue

            # read in the stable poses of the mesh
            stable_poses = dataset.stable_poses(obj.key)
            for i, stable_pose in enumerate(stable_poses):

                # render images if stable pose is valid
                if stable_pose.p > stable_pose_min_p:
                    # log progress
                    logging.info('Rendering images for object %s in %s' %(obj.key, stable_pose.id))

                    # add to category maps
                    if obj.key not in obj_category_map.keys():
                        obj_category_map[obj.key] = cur_obj_label
                    pose_category_map['%s_%s' %(obj.key, stable_pose.id)] = cur_pose_label

                    '''
                    Read in candidate grasps and metrics
                    candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                    candidate_grasp_info is list of <GraspInfo> objects, each with .grasp 
                    and .collision_free properties
                    '''
                    candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                    candidate_grasps = [g.grasp for g in candidate_grasp_info]
                    grasp_metrics = dataset.grasp_metrics(obj.key, candidate_grasps, gripper=gripper.name)

                    # compute object pose relative to the table
                    T_obj_stp = stable_pose.T_obj_table.as_frames('obj', 'stp')
                    T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

                    # sample images from random variable
                    T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                    scene_objs = {'table': SceneObject(table_mesh, T_table_obj)}
                    urv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                                      render_modes,
                                                                      'camera',
                                                                      env_rv_params,
                                                                      stable_pose=stable_pose,
                                                                      scene_objs=scene_objs)
                    
                    render_start = time.time()
                    render_samples = urv.rvs(size=image_samples_per_stable_pose)
                    render_stop = time.time()
                    logging.info('Rendering images took %.3f sec' %(render_stop - render_start))

                    # tally total amount of data
                    num_grasps = len(candidate_grasps)
                    num_images = image_samples_per_stable_pose 
                    num_save = num_images * num_grasps
                    logging.info('Saving %d datapoints' %(num_save))

                    # for each candidate grasp on the object compute the projection
                    # of the grasp into image space
                    for render_sample in render_samples:
                        # read images
                        binary_im = render_sample.renders[RenderMode.SEGMASK].image
                        depth_im_table = render_sample.renders[RenderMode.DEPTH_SCENE].image # Rendered depth image
                        # read camera params
                        T_stp_camera = render_sample.camera.object_to_camera_pose
                        shifted_camera_intr = render_sample.camera.camera_intr

                        # read pixel offsets
                        cx = depth_im_table.center[1]
                        cy = depth_im_table.center[0]

                        # compute intrinsics for virtual camera of the final
                        # cropped and rescaled images
                        camera_intr_scale = float(im_final_height) / float(im_crop_height)
                        cropped_camera_intr = shifted_camera_intr.crop(im_crop_height, im_crop_width, cy, cx)
                        final_camera_intr = cropped_camera_intr.resize(camera_intr_scale)

                        # NEW 
                        # Randomly pick N points around a central H x W window to be reprojected, place into (N x 2) array
                        im_height = env_rv_params['im_height']
                        im_width = env_rv_params['im_width']
                        window_height = config['point_cloud']['window_height'] 
                        window_width = config['point_cloud']['window_width'] 
                        num_points = config['point_cloud']['num_points'] 
                        height_offset = (im_height - window_height)/2 
                        width_offset = (im_width - window_width)/2 



                        
                        # First approach: uniformly sample points from a central window
                        flattened_idxs = np.arange(window_height * window_width)
                        # num_points = 100 * 100 # Test out extreme case where we keep all points
                        sampled_point_idx = np.random.choice(flattened_idxs, num_points, replace=False)
                        row_idxs = (sampled_point_idx / window_height) + height_offset
                        col_idxs = (sampled_point_idx % window_width) + width_offset
                        sampled_point_coords = np.hstack((col_idxs.reshape(-1, 1),  # Since pixels are in (u, v)
                                                          row_idxs.reshape(-1, 1)))

                        # Reproject point cloud from depth image
                        if len(sampled_point_coords.shape) == 1 and sampled_point_coords.shape[0] == 2:
                            sampled_point_coords = np.reshape(sampled_point_coords, [1, 2])
                        pc_from_depth = np.ones((sampled_point_coords.shape[0], 3), dtype=float)
                        intr_cx = shifted_camera_intr.cx
                        intr_cy = shifted_camera_intr.cy
                        fx = shifted_camera_intr.fx
                        fy = shifted_camera_intr.fy
                        pc_from_depth[:, 0] = ((sampled_point_coords.astype(float)[:, 0]) - intr_cx) / fx
                        pc_from_depth[:, 1] = ((sampled_point_coords.astype(float)[:, 1]) - intr_cy) / fy

                        depth = np.zeros((num_points))
                        for i in range(num_points):
                            x, y = sampled_point_coords[i]
                            depth[i] = depth_im_table.raw_data[y, x] # NOTE: image arrays indexed by h, w

                        if depth is not None:
                            # homogenize
                            if len(depth.shape) == 0:
                                depth = depth * np.ones([len(sampled_point_coords)])
                            assert(depth.shape[0] == sampled_point_coords.shape[0])
                            pc_from_depth *= depth[:, None]
                        # transform to world coordinate system
                        # if to_world:
                        #     pc_from_depth = transform_points_3d(pc_from_depth, self.world_from_camera)
                        
                        if config['vis']['pc_from_depth']:
                            # Visualize object point cloud for rendered_sample alone
                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            xs = pc_from_depth[:,0]
                            ys = pc_from_depth[:,1]
                            zs = pc_from_depth[:,2]
                            ax.scatter(xs, ys, zs, s=1, c='blue')    
                            plt.show()
                        # END FIRST APPROACH
                        

                        # # Second approach, remove the table so we just sample the object surface
                        # # get the approx depth of the table corners, take the min of that to find the highest corner
                        # # take coordinates where depth is greater than highest corner

                        # # Get depth image and table corners
                        # depth_im = np.squeeze(depth_im_table.raw_data)
                        # corners = [(0,0), (im_height - 1, 0), (0, im_width - 1), (im_height - 1, im_width - 1)]

                        # # Get highest corner of table
                        # highest_corner_depth = np.min([depth_im[corner] for corner in corners])

                        # # Keep only object pixels
                        # obj_pixel_idxs = np.where(depth_im < highest_corner_depth) 
                        # obj_pixel_idxs = np.vstack((obj_pixel_idxs[1], obj_pixel_idxs[0])) # now first row is x coords, second row is y coords
                        # assert obj_pixel_idxs.shape[0] == 2
                        # obj_pixel_idxs = np.reshape(obj_pixel_idxs, [2, -1]) 
                        # num_pixels_on_obj = obj_pixel_idxs.shape[1]

                        # if num_pixels_on_obj < num_points:
                        #     logging.info("Not enough object points to sample ({} of {} required)".format(
                        #         num_pixels_on_obj, num_points))
                        #     continue

                        # # Uniformly sample from object pixels
                        # flattened_idxs = np.arange(num_pixels_on_obj)
                        # sampled_point_idx = np.random.choice(flattened_idxs, num_points, replace=False)
                        # sampled_point_coords = obj_pixel_idxs[:, sampled_point_idx]
                        # sampled_point_coords = sampled_point_coords.T # (N x 2)
                        
                        # # Reproject point cloud from depth image
                        # if len(sampled_point_coords.shape) == 1 and sampled_point_coords.shape[0] == 2:
                        #     sampled_point_coords = np.reshape(sampled_point_coords, [1, 2])
                        # pc_from_depth = np.ones((sampled_point_coords.shape[0], 3), dtype=float)
                        # intr_cx = shifted_camera_intr.cx
                        # intr_cy = shifted_camera_intr.cy
                        # fx = shifted_camera_intr.fx
                        # fy = shifted_camera_intr.fy
                        # pc_from_depth[:, 0] = ((sampled_point_coords.astype(float)[:, 0]) - intr_cx) / fx
                        # pc_from_depth[:, 1] = ((sampled_point_coords.astype(float)[:, 1]) - intr_cy) / fy

                        # depth = np.zeros((num_points))
                        # for i in range(num_points):
                        #     x, y = sampled_point_coords[i]
                        #     depth[i] = depth_im_table.raw_data[y, x] # NOTE: image arrays indexed by h, w

                        # if depth is not None:
                        #     # homogenize
                        #     if len(depth.shape) == 0:
                        #         depth = depth * np.ones([len(sampled_point_coords)])
                        #     assert(depth.shape[0] == sampled_point_coords.shape[0])
                        #     pc_from_depth *= depth[:, None]
                        # # transform to world coordinate system
                        # # if to_world:
                        # #     pc_from_depth = transform_points_3d(pc_from_depth, self.world_from_camera)
                        
                        # if config['vis']['pc_from_depth']:
                        #     # Visualize object point cloud for rendered_sample alone
                        #     fig = plt.figure()
                        #     ax = fig.add_subplot(111, projection='3d')
                        #     xs = pc_from_depth[:,0]
                        #     ys = pc_from_depth[:,1]
                        #     zs = pc_from_depth[:,2]
                        #     ax.scatter(xs, ys, zs, s=1, c='blue')    

                        #     fig2 = plt.figure(2)
                        #     plt.imshow(depth_im)
                        #     plt.show()



                        # Keep track of all the robust sampled grasps per rendered image NEW
                        grasp_pos_rendered_sample = []
                        grasp_ori_rendered_sample = []
                        grasp_quality_rendered_sample = []

                        # robust_grasps = []
                        # non_robust_grasps = []



                        # create a thumbnail for each grasp
                        for grasp_info in candidate_grasp_info:
                            '''
                            candidate_grasp_info = candidate_grasps_dict[obj.key][stable_pose.id]
                            candidate_grasp_info is list of <GraspInfo> objects, each with .grasp 
                            and .collision_free properties
                            '''

                            # read info
                            grasp = grasp_info.grasp
                            collision_free = grasp_info.collision_free
                            
                            # get the gripper pose
                            T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', T_stp_camera.from_frame)

                            '''
                            grasp_2d = grasp.project_camera(T_obj_camera, shifted_camera_intr)

                            # center images on the grasp, rotate to image x axis
                            dx = cx - grasp_2d.center.x
                            dy = cy - grasp_2d.center.y
                            translation = np.array([dy, dx])

                            binary_im_tf = binary_im.transform(translation, grasp_2d.angle)
                            depth_im_tf_table = depth_im_table.transform(translation, grasp_2d.angle)

                            # crop to image size
                            binary_im_tf = binary_im_tf.crop(im_crop_height, im_crop_width)
                            depth_im_tf_table = depth_im_tf_table.crop(im_crop_height, im_crop_width)

                            # resize to image size
                            binary_im_tf = binary_im_tf.resize((im_final_height, im_final_width), interp='nearest')
                            depth_im_tf_table = depth_im_tf_table.resize((im_final_height, im_final_width))
                            '''                        
    
                            # NEW: Added function to grasp class to returnreturn grasp position and orientation in camera frame
                            grasp_pos_camera_frame, grasp_eul_camera_frame = grasp.grasp_camera_T(T_obj_camera)
                            grasp_quality_col_free = (1 * collision_free) * grasp_metrics[grasp.id]['robust_ferrari_canny']



                            # non_robust_grasps = grasp_pos_rendered_sample[np.where(grasp_quality_rendered_sample[:,0] <= 0)] # Since it is (num_grasps, 1)
                            # robust_grasps = grasp_pos_rendered_sample[np.where(grasp_quality_rendered_sample[:,0] > 0)]


                            grasp_pos_rendered_sample.append(grasp_pos_camera_frame)
                            grasp_ori_rendered_sample.append(grasp_eul_camera_frame)
                            grasp_quality_rendered_sample.append(grasp_quality_col_free)


                            if config['vis']['grasp_on_pc']:
                                # Visualize a single grasp on the rendered point cloud
                                # print("grasp depth from grasp3d: {}".format(grasp_pos_camera_frame))
                                # print("grasp theta from grasp3d: {}".format(np.rad2deg(grasp_eul_camera_frame)))
                                # print("grasp depth from grasp2d: {}".format(grasp_2d.depth))
                                # print("grasp theta from grasp2d: {}".format(grasp_2d.angle))

                                # TODO: Confirm the orientation is correct by visualizing it

                                fig = plt.figure(1)
                                ax = fig.add_subplot(111, projection='3d')
                                xs = pc_from_depth[:,0]
                                ys = pc_from_depth[:,1]
                                zs = pc_from_depth[:,2]
                                ax.scatter(xs, ys, zs, s=1, c='blue') # Object point cloud    
                                ax.scatter(grasp_pos_camera_frame[0], grasp_pos_camera_frame[1], grasp_pos_camera_frame[2], s=50, c='red') # Grasp position

                                fig_obj_pc = plt.figure(2)
                                plt.imshow(np.squeeze(depth_im_tf_table.raw_data))
                                plt.show()
    
                            # form hand pose array
                            # hand_pose = np.r_[grasp_2d.center.y,
                            #                   grasp_2d.center.x,
                            #                   grasp_2d.depth,
                            #                   grasp_2d.angle,
                            #                   grasp_2d.center.y - shifted_camera_intr.cy,
                            #                   grasp_2d.center.x - shifted_camera_intr.cx,
                            #                   grasp_2d.width_px]
         

                            # store to data buffers
                            # tensor_datapoint['depth_ims_tf_table'] = depth_im_tf_table.raw_data
                            # tensor_datapoint['obj_masks'] = binary_im_tf.raw_data
                            # tensor_datapoint['hand_poses'] = hand_pose
                            # tensor_datapoint['collision_free'] = collision_free
                            # tensor_datapoint['obj_labels'] = cur_obj_label
                            # tensor_datapoint['pose_labels'] = cur_pose_label
                            # tensor_datapoint['image_labels'] = cur_image_label

                            # for metric_name, metric_val in grasp_metrics[grasp.id].iteritems():
                            #     coll_free_metric = (1 * collision_free) * metric_val
                            #     tensor_datapoint[metric_name] = coll_free_metric
                            # tensor_dataset.add(tensor_datapoint)

                        # visualize all the grasp positions on the rendered point cloud
                        rendered_img_num = (cur_image_label % num_images) + 1
                        logging.info("Located {} grasps on rendered image {} (of {}) in stable pose {} of object {}".format(
                            num_grasps, rendered_img_num, num_images, stable_pose.id, obj.key))                        

                        # Reshape all of the collected grasp information                         
                        grasp_pos_rendered_sample = np.reshape(grasp_pos_rendered_sample, (-1, 3))
                        grasp_ori_rendered_sample = np.reshape(grasp_ori_rendered_sample, (-1, 3))
                        grasp_quality_rendered_sample = np.reshape(grasp_quality_rendered_sample, (-1, 1))

                        assert num_grasps == grasp_pos_rendered_sample.shape[0]
                        assert num_grasps == grasp_ori_rendered_sample.shape[0]
                        assert num_grasps == grasp_quality_rendered_sample.shape[0]

                        if config['vis']['data_point']:
                            # fig = plt.figure(1)
                            # ax = fig.add_subplot(111, projection='3d')
                            # pc_xs = pc_from_depth[:,0]
                            # pc_ys = pc_from_depth[:,1]
                            # pc_zs = pc_from_depth[:,2]

                            # g_xs = grasp_pos_rendered_sample[:,0]
                            # g_ys = grasp_pos_rendered_sample[:,1]
                            # g_zs = grasp_pos_rendered_sample[:,2]

                            # ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud
                            # ax.scatter(g_xs, g_ys, g_zs, s=50, c='red') # Grasp positions
                            # plt.show()


                            non_robust_grasps = grasp_pos_rendered_sample[np.where(grasp_quality_rendered_sample[:,0] <= 0)] # Since it is (num_grasps, 1)
                            robust_grasps = grasp_pos_rendered_sample[np.where(grasp_quality_rendered_sample[:,0] > 0)]

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection='3d')
                            pc_xs = pc_from_depth[:,0]
                            pc_ys = pc_from_depth[:,1]
                            pc_zs = pc_from_depth[:,2]

                            ng_xs = non_robust_grasps[:,0]
                            ng_ys = non_robust_grasps[:,1]
                            ng_zs = non_robust_grasps[:,2]

                            rg_xs = robust_grasps[:,0]
                            rg_ys = robust_grasps[:,1]
                            rg_zs = robust_grasps[:,2]

                            ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud
                            ax.scatter(ng_xs, ng_ys, ng_zs, s=25, c='red') # Non robust grasp positions
                            ax.scatter(rg_xs, rg_ys, rg_zs, s=50, c='green') # Robust grasp positions
                            plt.show()


                        # Prepare labels: older stuff where each point cloud has K grasps
                        # label_rendered_sample = np.hstack([
                        #     grasp_quality_rendered_sample,
                        #     grasp_pos_rendered_sample,
                        #     grasp_ori_rendered_sample])
                        # assert label_rendered_sample.shape == (num_grasps, 7) 

                        # Sort rows (grasps) by grasp quality
                        # grasp_ordering = np.argsort(np.squeeze(grasp_quality_rendered_sample))
                        # sorted_labels = np.flip(label_rendered_sample[grasp_ordering], 0) # Since we want in decreasing order

                        '''
                        NEW: assign grasp to each point
                        for each point in point cloud:
                            nearest_grasp =  min distance grasp
                            if distance(nearest_grasp, point) is less than threshold away:
                                associate nearest_grasp with point
                            else:
                                mark point as having no robust grasp
                        '''
                        label_rendered_sample = np.zeros([num_points, 7]) # (num_points x 7) THE final label
                        unassigned_grasp_labels = np.hstack([ # (num_grasps x 7) Temporary stack the label together
                            grasp_quality_rendered_sample,
                            grasp_pos_rendered_sample,
                            grasp_ori_rendered_sample])


                        # n_bins = 10
                        # bin_counts = np.zeros([n_bins, ], dtype=int) # To get a sense of how far the nearest grasp is for each point

                        '''
                        Generate a label for each point
                        label = (exists_nearby_grasp, robustness_class, x, y, z, alpha, beta, gamma)

                        hyperparameters:
                            threshold: 
                            robustness_class_boundaries:

                        '''
                        threshold = 0.02
                        # robustness_class_boundaries = 
                        for i, point in enumerate(pc_from_depth):
                            # Get index of nearest grasp
                            distance_to_grasps = np.linalg.norm(point - grasp_pos_rendered_sample, axis=-1) # (num_grasps, )
                            nearest_grasp_idx = np.argmin(distance_to_grasps)
                            
                            # TODO: Thresholding
                            nearest_grasp_distance = distance_to_grasps[nearest_grasp_idx]
                            # bin_counts[int(nearest_grasp_distance/0.01)] += 1

                            label_rendered_sample[i] = unassigned_grasp_labels[nearest_grasp_idx] # (7, )

                            # print("Distance to nearest grasp: {}".format(nearest_grasp_distance))
                            # print("Point: {}".format(point))
                            # print("Label: {}".format(label_rendered_sample[i]))
                            
                            # # Visualize point, nearest grasp, and other grasps
                            # fig = plt.figure(1)
                            # ax = fig.add_subplot(111, projection='3d')
                            # xs = pc_from_depth[:,0]
                            # ys = pc_from_depth[:,1]
                            # zs = pc_from_depth[:,2]

                            # g_xs = grasp_pos_rendered_sample[:,0]
                            # g_ys = grasp_pos_rendered_sample[:,1]
                            # g_zs = grasp_pos_rendered_sample[:,2]

                            # ax.scatter(xs, ys, zs, s=1, c='blue') # Object point cloud    
                            # ax.scatter(label_rendered_sample[i, 1], label_rendered_sample[i, 2], label_rendered_sample[i, 3], s=50, c='red') # Nearest grasp
                            # ax.scatter(g_xs, g_ys, g_zs, s=30, c='purple') # Other grasps
                            # # ax.scatter(label_rendered_sample[i, 1], label_rendered_sample[i, 2], label_rendered_sample[i, 3], s=30, c='yellow') # Other grasps
                            # ax.scatter(point[0], point[1], point[2], s=50, c='green') # Point

                            # plt.show()

                        # To get a sense of how far the nearest grasp is for each point
                        # bin_ranges = [0.01 * i for i in range(n_bins)]
                        # print("Total points: {}".format(num_points))
                        # plt.plot(bin_ranges, bin_counts)
                        # plt.show()



                        # print("labels {}".format(label_rendered_sample.shape))
                        point_cloud_data.append(pc_from_depth)
                        # grasp_labels.append(sorted_labels)
                        grasp_labels.append(label_rendered_sample)


                        # update image label
                        cur_image_label += 1

                    # update pose label
                    cur_pose_label += 1

                    # force clean up
                    gc.collect()

            # update object label
            cur_obj_label += 1

            # force clean up
            gc.collect()

    # save last file
    tensor_dataset.flush()

    # Debugging
    # mngpc = float('inf')
    # for vec in grasp_labels:
    #     d = vec.shape[0]
    #     if d < mngpc:
    #         mngpc = d
    # print("Minimum num_grasps_per_cloud: {}".format(mngpc))

    # Only keep the best K grasps
    # num_grasps_to_keep = 20
    # # assert num_grasps_to_keep < mngpc
    # truncated_grasp_labels = [label[:num_grasps_to_keep] for label in grasp_labels]

    # point_cloud_data = np.reshape(point_cloud_data, [-1, 2048, 3]) # (num_point_clouds x points_per_cloud x 3)
    # truncated_grasp_labels = np.reshape(truncated_grasp_labels, [-1, num_grasps_to_keep, 7]) # (num_point_clouds x num_grasps_per_cloud x 7) CONFIRM THIS DOES WHAT YOU WANT
    # assert point_cloud_data.shape[0] == truncated_grasp_labels.shape[0]

    # grasp_labels = np.reshape(grasp_labels, [-1, num_grasps_to_keep, 7]) # (num_point_clouds x num_grasps_per_cloud x 7) CONFIRM THIS DOES WHAT YOU WANT
    # filename = 'test_data.h5'
    # print(point_cloud_data)
    # print(truncated_grasp_labels)
    save_h5(hdf5_path, point_cloud_data, grasp_labels)
    logging.info("Saved hdf5 file. Program exiting.")


    # save category mappings
    obj_cat_filename = os.path.join(output_dir, 'object_category_map.json')
    json.dump(obj_category_map, open(obj_cat_filename, 'w'))
    pose_cat_filename = os.path.join(output_dir, 'pose_category_map.json')
    json.dump(pose_category_map, open(pose_cat_filename, 'w'))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Create a GQ-CNN training dataset from a dataset of 3D object models and grasps in a Dex-Net database')
    parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    parser.add_argument('hdf5_path', type=str, default=None, help='name of file to save the training dataset in')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    hdf5_path = args.hdf5_path
    config_filename = args.config_filename

    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/generate_pointcloud_dataset.yaml') # NEW

    # turn relative paths absolute
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(os.getcwd(), dataset_path)
    if not os.path.isabs(config_filename):
        config_filename = os.path.join(os.getcwd(), config_filename)

    # parse config
    config = YamlConfig(config_filename)

    # set seed
    debug = config['debug']
    if debug:
        random.seed(SEED)
        np.random.seed(SEED)
        
    # open database
    database = Hdf5Database(config['database_name'],
                            access_level=READ_ONLY_ACCESS)

    # read params
    target_object_keys = config['target_objects']
    env_rv_params = config['env_rv_params']
    gripper_name = config['gripper']

    # generate the dataset
    generate_gqcnn_dataset(dataset_path,
                           hdf5_path,
                           database,
                           target_object_keys,
                           env_rv_params,
                           gripper_name,
                           config)
