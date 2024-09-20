#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## KINEMATICS PROCESSING                                                 ##
###########################################################################

Process kinematic data using OpenSim tools.

This script performs scaling, inverse kinematics, and related processing
on 3D motion capture data (TRC files). The scaling process adjusts the
generic model to match the subject's physical dimensions, while inverse
kinematics computes the joint angles based on the motion data.

Set your parameters in Config.toml.

INPUTS:
- a directory containing TRC files
- kinematic processing parameters in Config.toml

OUTPUT:
- scaled OpenSim model files (.osim)
- joint angle data files (.mot)
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

from Pose2Sim.common import natural_sort_key, euclidean_distance, trimmed_mean
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


## FUNCTIONS
def read_trc(trc_path):
    '''
    Read a TRC file and extract its contents.

    INPUTS:
    - trc_path (str): The path to the TRC file.

    OUTPUTS:
    - tuple: A tuple containing the Q coordinates, frames column, time column, and header.
    '''

    try:
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split('\t')[2::3][:-1]
        
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1, -1]], axis=1)

        return Q_coords, frames_col, time_col, markers, header
    
    except Exception as e:
        logging.error(f"Error reading TRC file at {trc_path}: {e}")
        raise


def get_opensim_setup_dir():
    '''
    Locate the OpenSim setup directory within the Pose2Sim package.

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
    Retrieve the OpenSim scaling setup file path.

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
    Retrieve the OpenSim inverse kinematics setup file path.

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
        ik_setup_file = 'IK_Setup_Pose2Sim_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{model_name}' not found.")

    ik_setup_path = osim_setup_dir / ik_setup_file
    return ik_setup_path


# def get_output_dir(config_dir, person_id):
    '''
    Determines the correct output directory based on the configuration and the person identifier.

    INPUTS:
    - config_dir (Path): The root directory where the configuration file is located.
    - person_id (str): Identifier for the person (e.g., 'SinglePerson', 'P1').

    OUTPUTS:
    - Path: The path where the output files should be stored.
    '''

    output_dir = config_dir / 'kinematics'  # Assuming 'opensim' as the default output subdirectory

    # Append the person_id to the output directory if it's a multi-person setup
    if person_id != "SinglePerson":
        output_dir = output_dir / person_id

    logging.info(f"Output directory determined as: {output_dir}")

    # Create the directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def get_kpt_pairs_from_tree(root_node):
    '''
    Get name pairs for all parent-child relationships in the tree.
    # Excludes the root node.

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
    Get name pairs for all marker pairs in the scaling setup file.
    '''

    pairs = [pair.find('markers').text.strip().split(' ') for pair in scaling_root[0].findall(".//MarkerPair")]

    return pairs


def dict_segment_marker_pairs(scaling_root, right_left_symmetry=True):
    '''
    
    '''

    measurement_dict = {}
    for measurement in scaling_root.findall(".//Measurement"):
        # Collect all marker pairs for this measurement
        marker_pairs = [pair.find('markers').text.strip().split() for pair in measurement.findall(".//MarkerPair")]

        # Collect all body scales for this measurement
        for body_scale in measurement.findall(".//BodyScale"):
            body_name = body_scale.get('name')
            if right_left_symmetry:
                measurement_dict[body_name] = marker_pairs
            else:
                if body_name.endswith('_r'):
                    marker_pairs_r = [pair for pair in marker_pairs if any([pair[0].startswith('R'), pair[1].startswith('R')])]
                    measurement_dict[body_name] = marker_pairs_r
                elif body_name.endswith('_l'):
                    marker_pairs_l = [pair for pair in marker_pairs if any([pair[0].startswith('L'), pair[1].startswith('L')])]
                    measurement_dict[body_name] = marker_pairs_l

    return measurement_dict


def dict_segment_ratio(scaling_root, unscaled_model, Q_coords_scaling, markers, right_left_symmetry=True):
    '''
    '''

    # segment_pairs = get_kpt_pairs_from_tree(eval(model_name))
    segment_pairs = get_kpt_pairs_from_scaling(scaling_root)

    # Get model segment lengths
    model_markers_locs = [unscaled_model.getMarkerSet().get(marker).getLocationInGround(unscaled_model.getWorkingState()).to_numpy() for marker in markers]
    model_segment_lengths = np.array([euclidean_distance(model_markers_locs[markers.index(pt1)], 
                                                model_markers_locs[markers.index(pt2)]) 
                                                for (pt1,pt2) in segment_pairs])
    
    # Get median segment lengths from Q_coords_scaling. Trimmed mean works better than mean or median
    trc_segment_lengths = np.array([euclidean_distance(Q_coords_scaling.iloc[:,markers.index(pt1)*3:markers.index(pt1)*3+3], 
                        Q_coords_scaling.iloc[:,markers.index(pt2)*3:markers.index(pt2)*3+3]) 
                        for (pt1,pt2) in segment_pairs])
    # trc_segment_lengths = np.median(trc_segment_lengths, axis=1)
    # trc_segment_lengths = np.mean(trc_segment_lengths, axis=1)
    trc_segment_lengths = np.array([trimmed_mean(arr, trimmed_percent=0.5) for arr in trc_segment_lengths])

    # Calculate ratio for each segment
    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root, right_left_symmetry=right_left_symmetry)
    segment_ratio_dict = segment_markers_dict.copy()
    segment_ratio_dict.update({key: np.mean([segment_ratios[segment_pairs.index(k)] 
                                            for k in segment_markers_dict[key]]) 
                                for key in segment_markers_dict.keys()})
    
    return segment_ratio_dict


def deactivate_measurements(scaling_root):
    '''
    '''
    
    measurement_set = scaling_root.find(".//MeasurementSet/objects")
    for measurement in measurement_set.findall('Measurement'):
            apply_elem = measurement.find('apply')
            apply_elem.text = 'false'


def update_scale_values(scaling_root, segment_ratio_dict):
    '''
    '''
    
    # Get the ScaleSet/objects element
    scale_set = scaling_root.find(".//ScaleSet/objects")

    # Remove all existing Scale elements
    for scale in scale_set.findall('Scale'):
        scale_set.remove(scale)

    # Add new Scale elements based on scale_dict
    for segment, scale in segment_ratio_dict.items():
        new_scale = etree.Element('Scale')
        # scales
        scales_elem = etree.SubElement(new_scale, 'scales')
        scales_elem.text = ' '.join([str(scale)]*3)
        # segment name
        segment_elem = etree.SubElement(new_scale, 'segment')
        segment_elem.text = segment
        # apply True
        apply_elem = etree.SubElement(new_scale, 'apply')
        apply_elem.text = 'true'

        scale_set.append(new_scale)
        

def perform_scaling(trc_file, kinematics_dir, osim_setup_dir, model_name, right_left_symmetry=True, subject_height=1.75, subject_mass=70, remove_scaling_setup=True):
    '''
    Perform model scaling based on the (not necessarily static) TRC file:
    - Retrieve the 80% slowest frames, excluding frames where the person is out of frame.
    - From these frames, measure median segment lengths.
    - Calculate ratio between model and measured segment lengths -> OpenSim manual scaling.
    
    INPUTS:
    - config_dict (dict): The configuration dictionary.
    - person_id (str): The person identifier (e.g., 'SinglePerson', 'P1').
    - trc_files (list): List of TRC files to be processed.
    - output_dir (Path): The directory where the output files should be saved.
    '''

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
        scaling_path_temp = str(kinematics_dir / Path(scaling_path).name)

        # Read trc file
        Q_coords, _, _, markers, _ = read_trc(trc_file)

        # Using 80% slowest frames for scaling, removing frames when person is out of frame
        Q_diff = Q_coords.diff(axis=0).sum(axis=1)
        Q_diff = Q_diff[Q_diff != 0] # remove when speed is 0 (person out of frame)
        min_speed_indices = Q_diff.abs().nsmallest(int(len(Q_diff) * 0.8)).index
        Q_coords_scaling = Q_coords.iloc[min_speed_indices].reset_index(drop=True)

        # Get manual scale values (scale on trimmed mean of measured segments rather than on raw keypoints)
        segment_ratio_dict = dict_segment_ratio(scaling_root, unscaled_model, Q_coords_scaling, markers, right_left_symmetry=right_left_symmetry)

        # Update scaling setup file
        scaling_root[0].find('mass').text = str(subject_mass)
        scaling_root[0].find('height').text = str(subject_height)
        scaling_root[0].find('GenericModelMaker').find('model_file').text = str(unscaled_model_path)
        scaling_root[0].find(".//scaling_order").text = ' manualScale measurements'
        deactivate_measurements(scaling_root)
        update_scale_values(scaling_root, segment_ratio_dict)
        for mk_f in scaling_root[0].findall(".//marker_file"):
            mk_f.text = "Unassigned"
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
    Perform inverse kinematics on the TRC files according to the OpenSim configuration.

    INPUTS:
    - config_dict (dict): The configuration dictionary.
    - person_id (str): The person identifier (e.g., 'SinglePerson', 'P1').
    - trc_files (list): List of TRC files to be processed.
    - output_dir (Path): The directory where the output files should be saved.
    '''

    try:
        # Retrieve data
        ik_path = get_IK_Setup(model_name, osim_setup_dir)
        ik_path_temp =  str(kinematics_dir / Path(ik_path).name)
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


def kinematics(config_dict):
    '''
    Runs OpenSim scaling and inverse kinematics on the trc files of triangulated coordinates.

    INPUTS:
    - config_dict (dict): Generated from a .toml calibration file

    OUTPUTS:
    - A scaled .osim model for each person.
    - Joint angle data files (.mot) for each person.
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
    remove_scaling_setup = config_dict.get('kinematics').get('remove_individual_scaling_setup')
    remove_IK_setup = config_dict.get('kinematics').get('remove_individual_IK_setup')
    subject_height = config_dict.get('project').get('participant_height')
    subject_mass = config_dict.get('project').get('participant_mass')

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
        logging.warning("Number of subject masses does not match number of TRC files. Missing masses are set to 70kg.")
        subject_mass += [70] * (len(trc_files) - len(subject_mass))

    # Perform scaling and IK for each trc file
    for p, trc_file in enumerate(trc_files):
        logging.info(f"Processing TRC file: {trc_file.resolve()}")

        logging.info("Scaling...")
        perform_scaling(trc_file, kinematics_dir, osim_setup_dir, model_name, right_left_symmetry=right_left_symmetry, subject_height=subject_height[p], subject_mass=subject_mass[p], remove_scaling_setup=remove_scaling_setup)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")
        
        logging.info("\nInverse Kinematics...")
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, model_name, remove_IK_setup=remove_IK_setup)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tJoint angle data saved to {(kinematics_dir / (trc_file.stem + '.mot')).resolve()}\n")
