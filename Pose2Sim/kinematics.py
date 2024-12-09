#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## KINEMATICS PROCESSING                                                 ##
###########################################################################

    Runs OpenSim scaling and inverse kinematics
    
    Scaling:
    - No need for a static trial: scaling is done on the triangulated coordinates (trc file)
    - Remove 10% fastest frames (potential outliers)
    - Remove frames where coordinate speed is null (person probably out of frame)
    - Remove 40% most extreme calculated segment values (potential outliers)
    - For each segment, scale on the mean of the remaining segment values
    
    Inverse Kinematics:
    - Run on the scaled model with the same trc file
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - config_dict (dict): Generated from a .toml calibration file

    OUTPUTS:
    - A scaled .osim model for each person
    - Joint angle data files (.mot) for each person
    - Optionally, OpenSim scaling and IK setup files saved to the kinematics directory
    - Pose2Sim and OpenSim logs saved to files
'''


## INIT
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree
import logging
from anytree import PreOrderIter

import opensim

from Pose2Sim.common import natural_sort_key, euclidean_distance, trimmed_mean, points_to_angles
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "Ivan Sun, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["Ivan Sun", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.10.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## CONSTANTS
angle_dict = { # lowercase!
    # joint angles
    'right ankle': [['RKnee', 'RAnkle', 'RBigToe', 'RHeel'], 'dorsiflexion', 90, 1],
    'left ankle': [['LKnee', 'LAnkle', 'LBigToe', 'LHeel'], 'dorsiflexion', 90, 1],
    'right knee': [['RAnkle', 'RKnee', 'RHip'], 'flexion', -180, 1],
    'left knee': [['LAnkle', 'LKnee', 'LHip'], 'flexion', -180, 1],
    'right hip': [['RKnee', 'RHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left hip': [['LKnee', 'LHip', 'Hip', 'Neck'], 'flexion', 0, -1],
    # 'lumbar': [['Neck', 'Hip', 'RHip', 'LHip'], 'flexion', -180, -1],
    # 'neck': [['Head', 'Neck', 'RShoulder', 'LShoulder'], 'flexion', -180, -1],
    'right shoulder': [['RElbow', 'RShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'left shoulder': [['LElbow', 'LShoulder', 'Hip', 'Neck'], 'flexion', 0, -1],
    'right elbow': [['RWrist', 'RElbow', 'RShoulder'], 'flexion', 180, -1],
    'left elbow': [['LWrist', 'LElbow', 'LShoulder'], 'flexion', 180, -1],
    'right wrist': [['RElbow', 'RWrist', 'RIndex'], 'flexion', -180, 1],
    'left wrist': [['LElbow', 'LIndex', 'LWrist'], 'flexion', -180, 1],

    # segment angles
    'right foot': [['RBigToe', 'RHeel'], 'horizontal', 0, -1],
    'left foot': [['LBigToe', 'LHeel'], 'horizontal', 0, -1],
    'right shank': [['RAnkle', 'RKnee'], 'horizontal', 0, -1],
    'left shank': [['LAnkle', 'LKnee'], 'horizontal', 0, -1],
    'right thigh': [['RKnee', 'RHip'], 'horizontal', 0, -1],
    'left thigh': [['LKnee', 'LHip'], 'horizontal', 0, -1],
    'pelvis': [['LHip', 'RHip'], 'horizontal', 0, -1],
    'trunk': [['Neck', 'Hip'], 'horizontal', 0, -1],
    'shoulders': [['LShoulder', 'RShoulder'], 'horizontal', 0, -1],
    'head': [['Head', 'Neck'], 'horizontal', 0, -1],
    'right arm': [['RElbow', 'RShoulder'], 'horizontal', 0, -1],
    'left arm': [['LElbow', 'LShoulder'], 'horizontal', 0, -1],
    'right forearm': [['RWrist', 'RElbow'], 'horizontal', 0, -1],
    'left forearm': [['LWrist', 'LElbow'], 'horizontal', 0, -1],
    'right hand': [['RIndex', 'RWrist'], 'horizontal', 0, -1],
    'left hand': [['LIndex', 'LWrist'], 'horizontal', 0, -1]
    }


## FUNCTIONS
def read_trc(trc_path):
    '''
    Read a TRC file and extract its contents.

    INPUTS:
    - trc_path (str): The path to the TRC file.

    OUTPUTS:
    - tuple: A tuple containing the Q coordinates, frames column, time column, marker names, and header.
    '''

    try:
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split('\t')[2::3]
        
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

        return Q_coords, frames_col, time_col, markers, header
    
    except Exception as e:
        raise ValueError(f"Error reading TRC file at {trc_path}: {e}")


def get_opensim_setup_dir():
    '''
    Locate the OpenSim setup directory within the Pose2Sim package.

    INPUTS:
    - None

    OUTPUTS:
    - Path: The path to the OpenSim setup directory.
    '''
    
    pose2sim_path = Path(sys.modules['Pose2Sim'].__file__).resolve().parent
    setup_dir = pose2sim_path / 'OpenSim_Setup'
    return setup_dir


def get_model_path(model_name, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim model file.

    INPUTS:
    - model_name (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - pose_model_path: (Path) Path to the OpenSim model file.
    '''

    if model_name == 'BODY_25B':
        pose_model_file = 'Model_Setup_Pose2Sim_Body25b.osim'
    elif model_name == 'BODY_25':
        pose_model_file = 'Model_Pose2Sim_Body25.osim'
    elif model_name == 'BODY_135':
        pose_model_file = 'Model_Pose2Sim_Body135.osim'
    elif model_name == 'BLAZEPOSE':
        pose_model_file = 'Model_Pose2Sim_Blazepose.osim'
    elif model_name == 'HALPE_26':
        pose_model_file = 'Model_Pose2Sim_Halpe26.osim'
    elif model_name == 'HALPE_68' or model_name == 'HALPE_136':
        pose_model_file = 'Model_Pose2Sim_Halpe68_136.osim'
    elif model_name == 'COCO_133':
        pose_model_file = 'Model_Pose2Sim_Coco133.osim'
    # elif model_name == 'COCO' or model_name == 'MPII':
    #     pose_model_file = 'Model_Pose2Sim_Coco.osim'
    elif model_name == 'COCO_17':
        pose_model_file = 'Model_Pose2Sim_Coco17.osim'
    elif model_name == 'LSTM':
        pose_model_file = 'Model_Pose2Sim_LSTM.osim'
    else:
        raise ValueError(f"Pose model '{model_name}' not found.")

    unscaled_model_path = osim_setup_dir / pose_model_file

    return unscaled_model_path


def get_scaling_setup(model_name, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim scaling setup file.

    INPUTS:
    - model_name (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - scaling_setup_path: (Path) Path to the OpenSim scaling setup file.
    '''

    if model_name == 'BODY_25B':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body25b.xml'
    elif model_name == 'BODY_25':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body25.xml'
    elif model_name == 'BODY_135':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body135.xml'
    elif model_name == 'BLAZEPOSE':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Blazepose.xml'
    elif model_name == 'HALPE_26':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Halpe26.xml'
    elif model_name == 'HALPE_68' or model_name == 'HALPE_136':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Halpe68_136.xml'
    elif model_name == 'COCO_133':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco133.xml'
    # elif model_name == 'COCO' or model_name == 'MPII':
    #     scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco.xml'
    elif model_name == 'COCO_17':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco17.xml'
    elif model_name == 'LSTM':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{model_name}' not found.")

    scaling_setup_path = osim_setup_dir / scaling_setup_file

    return scaling_setup_path


def get_IK_Setup(model_name, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim inverse kinematics setup file.

    INPUTS:
    - model_name (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - ik_setup_path: (Path) Path to the OpenSim IK setup file.
    '''
    
    if model_name == 'BODY_25B':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25b.xml'
    elif model_name == 'BODY_25':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25.xml'
    elif model_name == 'BODY_135':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body135.xml'
    elif model_name == 'BLAZEPOSE':
        ik_setup_file = 'IK_Setup_Pose2Sim_Blazepose.xml'
    elif model_name == 'HALPE_26':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe26.xml'
    elif model_name == 'HALPE_68' or model_name == 'HALPE_136':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe68_136.xml'
    elif model_name == 'COCO_133':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco133.xml'
    # elif model_name == 'COCO' or model_name == 'MPII':
    #     ik_setup_file = 'IK_Setup_Pose2Sim_Coco.xml'
    elif model_name == 'COCO_17':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco17.xml'
    elif model_name == 'LSTM':
        ik_setup_file = 'IK_Setup_Pose2Sim_withHands_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{model_name}' not found.")

    ik_setup_path = osim_setup_dir / ik_setup_file
    return ik_setup_path


def get_kpt_pairs_from_tree(root_node):
    '''
    Get marker pairs for all parent-child relationships in the tree.
    # Excludes the root node.
    # Not used in the current version.

    INPUTS:
    - root_node (Node): The root node of the tree.

    OUTPUTS:
    - list: A list of name pairs for all parent-child relationships in the tree.
    '''

    pairs = []
    for node in PreOrderIter(root_node):
        # if node.is_root:
        #     continue
        for child in node.children:
            pairs.append([node.name, child.name])

    return pairs


def get_kpt_pairs_from_scaling(scaling_root):
    '''
    Get all marker pairs from the scaling setup file.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - pairs: A list of marker pairs.
    '''

    pairs = [pair.find('markers').text.strip().split(' ') 
             for pair in scaling_root[0].findall(".//MarkerPair")]

    return pairs


def mean_angles(Q_coords, markers, ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip']):
    '''
    Compute the mean angle time series from 3D points for a given list of angles.

    INPUTS:
    - Q_coords (DataFrame): The triangulated coordinates of the markers.
    - markers (list): The list of marker names.
    - ang_to_consider (list): The list of angles to consider (requires angle_dict).

    OUTPUTS:
    - ang_mean: The mean angle time series.
    '''

    ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip']

    angs = []
    for ang_name in ang_to_consider:
        ang_params = angle_dict[ang_name]
        ang_mk = ang_params[0]
        
        pts_for_angles = []
        for pt in ang_mk:
            pts_for_angles.append(Q_coords.iloc[:,markers.index(pt)*3:markers.index(pt)*3+3])
        ang = points_to_angles(pts_for_angles)

        ang += ang_params[2]
        ang *= ang_params[3]
        ang = np.abs(ang)

        angs.append(ang)

    ang_mean = np.mean(angs, axis=0)

    return ang_mean


def best_coords_for_measurements(Q_coords, keypoints_names, fastest_frames_to_remove_percent=0.2, close_to_zero_speed=0.2, large_hip_knee_angles=45):
    '''
    Compute the best coordinates for measurements, after removing:
    - 20% fastest frames (may be outliers)
    - frames when speed is close to zero (person is out of frame): 0.2 m/frame, or 50 px/frame
    - frames when hip and knee angle below 45° (imprecise coordinates when person is crouching)
    
    INPUTS:
    - Q_coords: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float
    - close_to_zero_speed: float (sum for all keypoints: about 50 px/frame or 0.2 m/frame)
    - large_hip_knee_angles: int
    - trimmed_extrema_percent

    OUTPUT:
    - Q_coords_low_speeds_low_angles: pd.DataFrame. The best coordinates for measurements
    '''

    # Add Hip column if not present
    n_markers_init = len(keypoints_names)
    if 'Hip' not in keypoints_names:
        RHip_df = Q_coords.iloc[:,keypoints_names.index('RHip')*3:keypoints_names.index('RHip')*3+3]
        LHip_df = Q_coords.iloc[:,keypoints_names.index('LHip')*3:keypoints_names.index('RHip')*3+3]
        Hip_df = RHip_df.add(LHip_df, fill_value=0) /2
        Hip_df.columns = [col+ str(int(Q_coords.columns[-1][1:])+1) for col in ['X','Y','Z']]
        keypoints_names += ['Hip']
        Q_coords = pd.concat([Q_coords, Hip_df], axis=1)
    n_markers = len(keypoints_names)

    # Using 80% slowest frames
    sum_speeds = pd.Series(np.nansum([np.linalg.norm(Q_coords.iloc[:,kpt:kpt+3].diff(), axis=1) for kpt in range(n_markers)], axis=0))
    sum_speeds = sum_speeds[sum_speeds>close_to_zero_speed] # Removing when speeds close to zero (out of frame)
    min_speed_indices = sum_speeds.abs().nsmallest(int(len(sum_speeds) * (1-fastest_frames_to_remove_percent))).index
    Q_coords_low_speeds = Q_coords.iloc[min_speed_indices].reset_index(drop=True)    
    
    # Only keep frames with hip and knee flexion angles below 45% 
    # (if more than 50 of them, else take 50 smallest values)
    ang_mean = mean_angles(Q_coords_low_speeds, keypoints_names, ang_to_consider = ['right knee', 'left knee', 'right hip', 'left hip'])
    Q_coords_low_speeds_low_angles = Q_coords_low_speeds[ang_mean < large_hip_knee_angles]
    if len(Q_coords_low_speeds_low_angles) < 50:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds.iloc[pd.Series(ang_mean).nsmallest(50).index]

    if n_markers_init < n_markers:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds_low_angles.iloc[:,:-3]

    return Q_coords_low_speeds_low_angles


def compute_height(Q_coords, keypoints_names, fastest_frames_to_remove_percent=0.1, close_to_zero_speed=50, large_hip_knee_angles=45, trimmed_extrema_percent=0.5):
    '''
    Compute the height of the person from the trc data.

    INPUTS:
    - Q_coords: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float. Frames with high speed are considered as outliers
    - close_to_zero_speed: float. Sum for all keypoints: about 50 px/frame or 0.2 m/frame
    - large_hip_knee_angles5: float. Hip and knee angles below this value are considered as imprecise
    - trimmed_extrema_percent: float. Proportion of the most extreme segment values to remove before calculating their mean)
    
    OUTPUT:
    - height: float. The estimated height of the person
    '''
    
    # Retrieve most reliable coordinates
    Q_coords_low_speeds_low_angles = best_coords_for_measurements(Q_coords, keypoints_names, 
                                                                  fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, close_to_zero_speed=close_to_zero_speed, large_hip_knee_angles=large_hip_knee_angles)
    Q_coords_low_speeds_low_angles.columns = np.array([[m]*3 for m in keypoints_names]).flatten()

    # Add MidShoulder column
    df_MidShoulder = pd.DataFrame((Q_coords_low_speeds_low_angles['RShoulder'].values + Q_coords_low_speeds_low_angles['LShoulder'].values) /2)
    df_MidShoulder.columns = ['MidShoulder']*3
    Q_coords_low_speeds_low_angles = pd.concat((Q_coords_low_speeds_low_angles.reset_index(drop=True), df_MidShoulder), axis=1)

    # Automatically compute the height of the person
    pairs_up_to_shoulders = [['RHeel', 'RAnkle'], ['RAnkle', 'RKnee'], ['RKnee', 'RHip'], ['RHip', 'RShoulder'],
                            ['LHeel', 'LAnkle'], ['LAnkle', 'LKnee'], ['LKnee', 'LHip'], ['LHip', 'LShoulder']]
    try:
        rfoot, rshank, rfemur, rback, lfoot, lshank, lfemur, lback = [euclidean_distance(Q_coords_low_speeds_low_angles[pair[0]],Q_coords_low_speeds_low_angles[pair[1]]) for pair in pairs_up_to_shoulders]
    except:
        raise ValueError('At least one of the following markers is missing for computing the height of the person:\
                         RHeel, RAnkle, RKnee, RHip, RShoulder, LHeel, LAnkle, LKnee, LHip, LShoulder.\
                         Make sure that the person is entirely visible, or use a calibration file instead, or set "to_meters=false".')
    if 'Head' in keypoints_names:
        head = euclidean_distance(Q_coords_low_speeds_low_angles['MidShoulder'], Q_coords_low_speeds_low_angles['Head'])
    else:
        head = euclidean_distance(Q_coords_low_speeds_low_angles['MidShoulder'], Q_coords_low_speeds_low_angles['Nose'])*1.33
    heights = (rfoot + lfoot)/2 + (rshank + lshank)/2 + (rfemur + lfemur)/2 + (rback + lback)/2 + head
    
    # Remove the 20% most extreme values
    height = trimmed_mean(heights, trimmed_extrema_percent=trimmed_extrema_percent)

    return height


def dict_segment_marker_pairs(scaling_root, right_left_symmetry=True):
    '''
    Get a dictionary of segment names and their corresponding marker pairs.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.

    OUTPUTS:
    - segment_markers_dict: A dictionary of segment names and their corresponding marker pairs.
    '''

    segment_markers_dict = {}
    for measurement in scaling_root.findall(".//Measurement"):
        # Collect all marker pairs for this measurement
        marker_pairs = [pair.find('markers').text.strip().split() for pair in measurement.findall(".//MarkerPair")]

        # Collect all body scales for this measurement
        for body_scale in measurement.findall(".//BodyScale"):
            body_name = body_scale.get('name')
            axes = body_scale.find('axes').text.strip().split()
            for axis in axes:
                body_name_axis = f"{body_name}_{axis}"
                if right_left_symmetry:
                    segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)
                else:
                    if body_name.endswith('_r'):
                        marker_pairs_r = [pair for pair in marker_pairs if any([pair[0].upper().startswith('R'), pair[1].upper().startswith('R')])]
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_r)
                    elif body_name.endswith('_l'):
                        marker_pairs_l = [pair for pair in marker_pairs if any([pair[0].upper().startswith('L'), pair[1].upper().startswith('L')])]
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs_l)
                    else:
                        segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)

    return segment_markers_dict


def dict_segment_ratio(scaling_root, unscaled_model, Q_coords_scaling, markers, trimmed_extrema_percent=0.5, right_left_symmetry=True):
    '''
    Calculate the ratios between the size of the actual segment and the size of the model segment.
    X, Y, and Z ratios are calculated separately if the original scaling setup file asks for it.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - unscaled_model (Model): The original OpenSim model before scaling.
    - Q_coords_scaling (DataFrame): The triangulated coordinates of the markers.
    - markers (list): The list of marker names.
    - trimmed_extrema_percent (float): The proportion of the most extreme segment values to remove before calculating their mean.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.

    OUTPUTS:
    - segment_ratio_dict: A dictionary of segment names and their corresponding X, Y, and Z ratios.
    '''

    # segment_pairs = get_kpt_pairs_from_tree(eval(model_name))
    segment_pairs = get_kpt_pairs_from_scaling(scaling_root)

    # Get median segment lengths from Q_coords_scaling. Trimmed mean works better than mean or median
    trc_segment_lengths = np.array([euclidean_distance(Q_coords_scaling.iloc[:,markers.index(pt1)*3:markers.index(pt1)*3+3], 
                        Q_coords_scaling.iloc[:,markers.index(pt2)*3:markers.index(pt2)*3+3]) 
                        for (pt1,pt2) in segment_pairs])
    # trc_segment_lengths = np.median(trc_segment_lengths, axis=1)
    # trc_segment_lengths = np.mean(trc_segment_lengths, axis=1)
    trc_segment_lengths = np.array([trimmed_mean(arr, trimmed_extrema_percent=trimmed_extrema_percent) for arr in trc_segment_lengths])

    # Get model segment lengths
    model_markers = [marker for marker in markers if marker in [m.getName() for m in unscaled_model.getMarkerSet()]]
    model_markers_locs = [unscaled_model.getMarkerSet().get(marker).getLocationInGround(unscaled_model.getWorkingState()).to_numpy() for marker in model_markers]
    model_segment_lengths = np.array([euclidean_distance(model_markers_locs[model_markers.index(pt1)], 
                                                model_markers_locs[model_markers.index(pt2)]) 
                                                for (pt1,pt2) in segment_pairs])
    
    # Calculate ratio for each segment
    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root, right_left_symmetry=right_left_symmetry)
    segment_ratio_dict_temp = segment_markers_dict.copy()
    segment_ratio_dict_temp.update({key: np.mean([segment_ratios[segment_pairs.index(k)] 
                                            for k in segment_markers_dict[key]]) 
                                for key in segment_markers_dict.keys()})
    # Merge X, Y, Z ratios into single key
    segment_ratio_dict={}
    xyz_keys = list(set([key[:-2] for key in segment_ratio_dict_temp.keys()]))
    for key in xyz_keys:
        segment_ratio_dict[key] = [segment_ratio_dict_temp[key+'_X'], segment_ratio_dict_temp[key+'_Y'], segment_ratio_dict_temp[key+'_Z']]
    
    return segment_ratio_dict


def deactivate_measurements(scaling_root):
    '''
    Deactivate all scalings based on marker positions (called 'measurements' in OpenSim) in the scaling setup file.
    (will use scaling based on segment sizes instead (called 'manual' in OpenSim))

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - scaling_root with deactivated measurements.
    '''
    
    measurement_set = scaling_root.find(".//MeasurementSet/objects")
    for measurement in measurement_set.findall('Measurement'):
            apply_elem = measurement.find('apply')
            apply_elem.text = 'false'


def update_scale_values(scaling_root, segment_ratio_dict):
    '''
    Remove previous scaling values ('manual') and 
    add new scaling values based on calculated segment ratios.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - segment_ratio_dict (dict): A dictionary of segment names and their corresponding X, Y, and Z ratios.

    OUTPUTS:
    - scaling_root with updated scaling values.
    '''
    
    # Get the ScaleSet/objects element
    scale_set = scaling_root.find(".//ScaleSet/objects")

    # Remove all existing Scale elements
    for scale in scale_set.findall('Scale'):
        scale_set.remove(scale)

    # Add new Scale elements based on scale_dict
    for segment, scales in segment_ratio_dict.items():
        new_scale = etree.Element('Scale')
        # scales
        scales_elem = etree.SubElement(new_scale, 'scales')
        scales_elem.text = ' '.join(map(str, scales))
        # segment name
        segment_elem = etree.SubElement(new_scale, 'segment')
        segment_elem.text = segment
        # apply True
        apply_elem = etree.SubElement(new_scale, 'apply')
        apply_elem.text = 'true'

        scale_set.append(new_scale)
        

def perform_scaling(trc_file, kinematics_dir, osim_setup_dir, model_name, right_left_symmetry=True, subject_height=1.75, subject_mass=70, 
                    remove_scaling_setup=True, fastest_frames_to_remove_percent=0.1,close_to_zero_speed_m=0.2, large_hip_knee_angles=45, trimmed_extrema_percent=0.5):
    '''
    Perform model scaling based on the (not necessarily static) TRC file:
    - Remove 10% fastest frames (potential outliers)
    - Remove frames where coordinate speed is null (person probably out of frame)
    - Remove 40% most extreme calculated segment values (potential outliers)
    - For each segment, scale on the mean of the remaining segment values
    
    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - kinematics_dir (Path): The directory where the kinematics files are saved.
    - osim_setup_dir (Path): The directory where the OpenSim setup and model files are stored.
    - model_name (str): The name of the model.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.
    - subject_height (float): The height of the subject.
    - subject_mass (float): The mass of the subject.
    - remove_scaling_setup (bool): Whether to remove the scaling setup file after scaling.
    
    OUTPUTS:
    - A scaled OpenSim model file.
    '''

    fastest_frames_to_remove_percent = 0.1 # fasters frames may be outliers
    large_hip_knee_angles = 45 # imprecise coordinates when person is crouching
    trimmed_extrema_percent = 0.2 # proportion of the most extreme segment values to remove before calculating their mean

    try:
        # Load model
        opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup_dir / 'Geometry'))
        unscaled_model_path = get_model_path(model_name, osim_setup_dir)
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        unscaled_model = opensim.Model(str(unscaled_model_path))
        unscaled_model.initSystem()
        scaled_model_path = (kinematics_dir / (trc_file.stem + '.osim')).resolve()

        # Load scaling setup
        scaling_path = get_scaling_setup(model_name, osim_setup_dir)
        scaling_tree = etree.parse(scaling_path)
        scaling_root = scaling_tree.getroot()
        scaling_path_temp = str(kinematics_dir / (trc_file.stem + '_scaling_setup.xml'))
        
        # Remove fastest frames, frames with null speed, and frames with large hip and knee angles
        Q_coords, _, _, markers, _ = read_trc(trc_file)
        Q_coords_low_speeds_low_angles = best_coords_for_measurements(Q_coords, markers, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, close_to_zero_speed=close_to_zero_speed_m)

        # Get manual scale values (mean from remaining frames after trimming the 20% most extreme values)
        segment_ratio_dict = dict_segment_ratio(scaling_root, unscaled_model, Q_coords_low_speeds_low_angles, markers, 
                                                trimmed_extrema_percent=trimmed_extrema_percent, right_left_symmetry=right_left_symmetry)

        # Update scaling setup file
        scaling_root[0].find('mass').text = str(subject_mass)
        scaling_root[0].find('height').text = str(subject_height)
        scaling_root[0].find('GenericModelMaker').find('model_file').text = str(unscaled_model_path)
        scaling_root[0].find(".//scaling_order").text = ' manualScale measurements'
        deactivate_measurements(scaling_root)
        update_scale_values(scaling_root, segment_ratio_dict)
        for mk_f in scaling_root[0].findall(".//marker_file"): mk_f.text = "Unassigned"
        scaling_root[0].find('ModelScaler').find('output_model_file').text = str(scaled_model_path)

        etree.indent(scaling_tree, space='\t', level=0)
        scaling_tree.write(scaling_path_temp, pretty_print=True, xml_declaration=True, encoding='utf-8')
    
        # Run scaling
        opensim.ScaleTool(scaling_path_temp).run()

        # Remove scaling setup
        if remove_scaling_setup:
            Path(scaling_path_temp).unlink()

    except Exception as e:
        logging.error(f"Error during scaling for {trc_file}: {e}")
        raise


def perform_IK(trc_file, kinematics_dir, osim_setup_dir, model_name, remove_IK_setup=True):
    '''
    Perform inverse kinematics based on a TRC file and a scaled OpenSim model:
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - kinematics_dir (Path): The directory where the kinematics files are saved.
    - osim_setup_dir (Path): The directory where the OpenSim setup and model files are stored.
    - model_name (str): The name of the model.
    - remove_IK_setup (bool): Whether to remove the IK setup file after running IK.

    OUTPUTS:
    - A joint angle data file (.mot).
    '''

    try:
        # Retrieve data
        ik_path = get_IK_Setup(model_name, osim_setup_dir)
        ik_path_temp =  str(kinematics_dir / (trc_file.stem + '_ik_setup.xml'))
        scaled_model_path = (kinematics_dir / (trc_file.stem + '.osim')).resolve()
        output_motion_file = Path(kinematics_dir, trc_file.stem + '.mot').resolve()
        if not trc_file.exists():
            raise FileNotFoundError(f"TRC file does not exist: {trc_file}")
        _, _, time_col, _, _ = read_trc(trc_file)
        start_time, end_time = time_col.iloc[0], time_col.iloc[-1]

        # Update IK setup file
        ik_tree = etree.parse(ik_path)
        ik_root = ik_tree.getroot()
        ik_root.find('.//model_file').text = str(scaled_model_path)
        ik_root.find('.//time_range').text = f'{start_time} {end_time}'
        ik_root.find('.//output_motion_file').text = str(output_motion_file)
        ik_root.find('.//marker_file').text = str(trc_file.resolve())
        ik_tree.write(ik_path_temp)

        # Run IK
        opensim.InverseKinematicsTool(str(ik_path_temp)).run()

        # Remove IK setup
        if remove_IK_setup:
            Path(ik_path_temp).unlink()

    except Exception as e:
        logging.error(f"Error during IK for {trc_file}: {e}")
        raise


def kinematics_all(config_dict):
    '''
    Runs OpenSim scaling and inverse kinematics
    
    Scaling:
    - No need for a static trial: scaling is done on the triangulated coordinates (trc file)
    - Remove 10% fastest frames (potential outliers)
    - Remove frames where coordinate speed is null (person probably out of frame)
    - Remove 40% most extreme calculated segment values (potential outliers)
    - For each segment, scale on the mean of the remaining segment values
    
    Inverse Kinematics:
    - Run on the scaled model with the same trc file
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - config_dict (dict): Generated from a .toml calibration file

    OUTPUTS:
    - A scaled .osim model for each person
    - Joint angle data files (.mot) for each person
    - Optionally, OpenSim scaling and IK setup files saved to the kinematics directory
    - Pose2Sim and OpenSim logs saved to files
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = Path(project_dir) / '..'
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    use_augmentation = config_dict.get('kinematics').get('use_augmentation')
    if use_augmentation: model_name = 'LSTM'
    else: model_name = config_dict.get('pose').get('pose_model').upper()
    right_left_symmetry = config_dict.get('kinematics').get('right_left_symmetry')
    subject_height = config_dict.get('project').get('participant_height')
    subject_mass = config_dict.get('project').get('participant_mass')

    remove_scaling_setup = config_dict.get('kinematics').get('remove_individual_scaling_setup')
    remove_IK_setup = config_dict.get('kinematics').get('remove_individual_IK_setup')
    fastest_frames_to_remove_percent = config_dict.get('kinematics').get('fastest_frames_to_remove_percent')
    large_hip_knee_angles = config_dict.get('kinematics').get('large_hip_knee_angles')
    trimmed_extrema_percent = config_dict.get('kinematics').get('trimmed_extrema_percent')
    close_to_zero_speed_m = config_dict.get('kinematics').get('close_to_zero_speed_m')

    pose3d_dir = Path(project_dir) / 'pose-3d'
    kinematics_dir = Path(project_dir) / 'kinematics'
    kinematics_dir.mkdir(parents=True, exist_ok=True)
    osim_setup_dir = get_opensim_setup_dir()
    
    # OpenSim logs saved to a different file
    opensim_logs_file = kinematics_dir / 'opensim_logs.txt'
    opensim.Logger.setLevelString('Info')
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(str(opensim_logs_file))

    # Find all trc files
    trc_files = []
    if use_augmentation:
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' in f.name]
        if len(trc_files) == 0:
            model_name = config_dict.get('pose').get('pose_model').upper()
            logging.warning("No LSTM trc files found. Using non augmented trc files instead.")
    if len(trc_files) == 0: # filtered files by default
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_filt' in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0: 
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0:
        raise ValueError(f'No trc files found in {pose3d_dir}.')
    sorted(trc_files, key=natural_sort_key)

    # Get subject heights and masses
    if subject_height is None or subject_height == 0:
        subject_height = [1.75] * len(trc_files)
        logging.warning("No subject height found in Config.toml. Using default height of 1.75m.")
    elif not type(subject_height) == list: # int or float
        subject_height = [subject_height]
    elif len(subject_height) < len(trc_files):
        logging.warning("Number of subject heights does not match number of TRC files. Missing heights are set to 1.75m.")
        subject_height += [1.75] * (len(trc_files) - len(subject_height))

    if subject_mass is None or subject_mass == 0:
        subject_mass = [70] * len(trc_files)
        logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")
    elif not type(subject_mass) == list:
        subject_mass = [subject_mass]
    elif len(subject_mass) < len(trc_files):
        logging.warning("Number of subject masses does not match number of TRC files. Missing masses are set to 70kg.\n")
        subject_mass += [70] * (len(trc_files) - len(subject_mass))

    # Perform scaling and IK for each trc file
    for p, trc_file in enumerate(trc_files):
        logging.info(f"Processing TRC file: {trc_file.resolve()}")

        logging.info("Scaling...")
        perform_scaling(trc_file, kinematics_dir, osim_setup_dir, model_name, right_left_symmetry=right_left_symmetry, subject_height=subject_height[p], subject_mass=subject_mass[p], 
                        remove_scaling_setup=remove_scaling_setup, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent,close_to_zero_speed_m=close_to_zero_speed_m)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")
        
        logging.info("Inverse Kinematics...")
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, model_name, remove_IK_setup=remove_IK_setup)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tJoint angle data saved to {(kinematics_dir / (trc_file.stem + '.mot')).resolve()}\n")