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
from lxml import etree
import logging
from anytree import PreOrderIter

import opensim

from Pose2Sim.common import natural_sort_key, euclidean_distance, trimmed_mean, read_trc, \
                            best_coords_for_measurements, compute_height
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


def get_model_path(use_contact_muscles, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim model file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - pose_model_path: (Path) Path to the OpenSim model file.
    '''

    if use_contact_muscles:
        pose_model_file = 'Model_Pose2Sim_contacts_muscles.osim'
    else:
        pose_model_file = 'Model_Pose2Sim.osim'

    unscaled_model_path = osim_setup_dir / pose_model_file

    return unscaled_model_path


def get_markers_path(pose_model, osim_setup_dir):
    '''
    Retrieve the path of the marker file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - markers_path: (Path) Path to the marker file.
    '''

    if pose_model == 'BODY_25B':
        marker_file = 'Markers_Body25b.xml'
    elif pose_model == 'BODY_25':
        marker_file = 'Markers_Body25.xml'
    elif pose_model == 'BODY_135':
        marker_file = 'Markers_Body135.xml'
    elif pose_model == 'BLAZEPOSE':
        marker_file = 'Markers_Blazepose.xml'
    elif pose_model == 'HALPE_26':
        marker_file = 'Markers_Halpe26.xml'
    elif pose_model == 'HALPE_68' or pose_model == 'HALPE_136':
        marker_file = 'Markers_Halpe68_136.xml'
    elif pose_model == 'COCO_133' or pose_model == 'COCO_133_WRIST':
        marker_file = 'Markers_Coco133.xml'
    # elif pose_model == 'COCO' or pose_model == 'MPII':
    #     marker_file = 'Markers_Coco.xml'
    elif pose_model == 'COCO_17':
        marker_file = 'Markers_Coco17.xml'
    elif pose_model == 'LSTM':
        marker_file = 'Markers_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

    markers_path = osim_setup_dir / marker_file

    return markers_path


def get_scaling_setup(pose_model, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim scaling setup file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - scaling_setup_path: (Path) Path to the OpenSim scaling setup file.
    '''

    if pose_model == 'BODY_25B':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body25b.xml'
    elif pose_model == 'BODY_25':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body25.xml'
    elif pose_model == 'BODY_135':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Body135.xml'
    elif pose_model == 'BLAZEPOSE':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Blazepose.xml'
    elif pose_model == 'HALPE_26':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Halpe26.xml'
    elif pose_model == 'HALPE_68' or pose_model == 'HALPE_136':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Halpe68_136.xml'
    elif pose_model == 'COCO_133' or pose_model == 'COCO_133_WRIST':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco133.xml'
    # elif pose_model == 'COCO' or pose_model == 'MPII':
    #     scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco.xml'
    elif pose_model == 'COCO_17':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco17.xml'
    elif pose_model == 'LSTM':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

    scaling_setup_path = osim_setup_dir / scaling_setup_file

    return scaling_setup_path


def get_IK_Setup(pose_model, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim inverse kinematics setup file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - ik_setup_path: (Path) Path to the OpenSim IK setup file.
    '''
    
    if pose_model == 'BODY_25B':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25b.xml'
    elif pose_model == 'BODY_25':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25.xml'
    elif pose_model == 'BODY_135':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body135.xml'
    elif pose_model == 'BLAZEPOSE':
        ik_setup_file = 'IK_Setup_Pose2Sim_Blazepose.xml'
    elif pose_model == 'HALPE_26':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe26.xml'
    elif pose_model == 'HALPE_68' or pose_model == 'HALPE_136':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe68_136.xml'
    elif pose_model == 'COCO_133' or pose_model == 'COCO_133_WRIST':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco133.xml'
    # elif pose_model == 'COCO' or pose_model == 'MPII':
    #     ik_setup_file = 'IK_Setup_Pose2Sim_Coco.xml'
    elif pose_model == 'COCO_17':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco17.xml'
    elif pose_model == 'LSTM':
        ik_setup_file = 'IK_Setup_Pose2Sim_withHands_LSTM.xml'
    else:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

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

    # segment_pairs = get_kpt_pairs_from_tree(eval(pose_model))
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
        

def perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, 
                    use_contacts_muscles=False, right_left_symmetry=True, subject_height=1.75, subject_mass=70, 
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
    - pose_model (str): The name of the model.
    - use_contacts_muscles (bool): Whether to use the model with contact spheres and muscles.
    - right_left_symmetry (bool): Whether to consider right and left side of equal size.
    - subject_height (float): The height of the subject.
    - subject_mass (float): The mass of the subject.
    - remove_scaling_setup (bool): Whether to remove the scaling setup file after scaling.
    - fastest_frames_to_remove_percent (float): Fasters frames may be outliers
    - large_hip_knee_angles (float): Imprecise coordinates when person is crouching
    - trimmed_extrema_percent (float): Proportion of the most extreme segment values to remove before calculating their mean
    
    OUTPUTS:
    - A scaled OpenSim model file.
    '''

    try:
        # Load model
        opensim.ModelVisualizer.addDirToGeometrySearchPaths(str(osim_setup_dir / 'Geometry'))
        unscaled_model_path = get_model_path(use_contacts_muscles, osim_setup_dir)
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        unscaled_model = opensim.Model(str(unscaled_model_path))
        # Add markers to model
        markers_path = get_markers_path(pose_model, osim_setup_dir)
        markerset = opensim.MarkerSet(str(markers_path))
        unscaled_model.set_MarkerSet(markerset)
        # Initialize and save model with markers
        unscaled_model.initSystem()
        scaled_model_path = str((kinematics_dir / (trc_file.stem + '.osim')).resolve())
        unscaled_model.printToXML(scaled_model_path)

        # Load scaling setup
        scaling_path = get_scaling_setup(pose_model, osim_setup_dir)
        scaling_tree = etree.parse(scaling_path)
        scaling_root = scaling_tree.getroot()
        scaling_path_temp = str(kinematics_dir / (trc_file.stem + '_scaling_setup.xml'))
        
        # Remove fastest frames, frames with null speed, and frames with large hip and knee angles
        Q_coords, _, _, markers, _ = read_trc(trc_file)
        Q_coords_low_speeds_low_angles = best_coords_for_measurements(Q_coords, markers, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, close_to_zero_speed=close_to_zero_speed_m)

        if Q_coords_low_speeds_low_angles.size == 0:
            logging.warning(f"\nNo frames left after removing fastest frames, frames with null speed, and frames with large hip and knee angles for {trc_file}. The person may be static, or crouched, or incorrectly detected.")
            logging.warning(f"Running with fastest_frames_to_remove_percent=0, close_to_zero_speed_m=0, large_hip_knee_angles=0, trimmed_extrema_percent=0. You can edit these parameters in your Config.toml file.\n")
            Q_coords_low_speeds_low_angles = Q_coords

        # Get manual scale values (mean from remaining frames after trimming the 20% most extreme values)
        segment_ratio_dict = dict_segment_ratio(scaling_root, unscaled_model, Q_coords_low_speeds_low_angles, markers, 
                                                trimmed_extrema_percent=trimmed_extrema_percent, right_left_symmetry=right_left_symmetry)

        # Update scaling setup file
        scaling_root[0].find('mass').text = str(subject_mass)
        scaling_root[0].find('height').text = str(subject_height)
        scaling_root[0].find('GenericModelMaker').find('model_file').text = scaled_model_path
        scaling_root[0].find(".//scaling_order").text = ' manualScale measurements'
        deactivate_measurements(scaling_root)
        update_scale_values(scaling_root, segment_ratio_dict)
        for mk_f in scaling_root[0].findall(".//marker_file"): mk_f.text = "Unassigned"
        scaling_root[0].find('ModelScaler').find('output_model_file').text = scaled_model_path

        etree.indent(scaling_tree, space='\t', level=0)
        scaling_tree.write(scaling_path_temp, pretty_print=True, xml_declaration=True, encoding='utf-8')

        # Run scaling
        opensim.ScaleTool(scaling_path_temp).run()

        # Remove scaling setup
        if remove_scaling_setup:
            Path(scaling_path_temp).unlink()

    except Exception as e:
        logging.error(f"Error during scaling for {trc_file}: {e}\.")
        raise


def perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=True):
    '''
    Perform inverse kinematics based on a TRC file and a scaled OpenSim model:
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - kinematics_dir (Path): The directory where the kinematics files are saved.
    - osim_setup_dir (Path): The directory where the OpenSim setup and model files are stored.
    - pose_model (str): The name of the model.
    - remove_IK_setup (bool): Whether to remove the IK setup file after running IK.

    OUTPUTS:
    - A joint angle data file (.mot).
    '''

    try:
        # Retrieve data
        ik_path = get_IK_Setup(pose_model, osim_setup_dir)
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
    use_contacts_muscles = config_dict.get('kinematics').get('use_contacts_muscles')

    right_left_symmetry = config_dict.get('kinematics').get('right_left_symmetry')
    subject_height = config_dict.get('project').get('participant_height')
    subject_mass = config_dict.get('project').get('participant_mass')

    fastest_frames_to_remove_percent = config_dict.get('kinematics').get('fastest_frames_to_remove_percent')
    large_hip_knee_angles = config_dict.get('kinematics').get('large_hip_knee_angles')
    trimmed_extrema_percent = config_dict.get('kinematics').get('trimmed_extrema_percent')
    close_to_zero_speed = config_dict.get('kinematics').get('close_to_zero_speed_m')
    default_height = config_dict.get('kinematics').get('default_height')

    remove_scaling_setup = config_dict.get('kinematics').get('remove_individual_scaling_setup')
    remove_IK_setup = config_dict.get('kinematics').get('remove_individual_ik_setup')

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
            pose_model = config_dict.get('pose').get('pose_model').upper()
            use_augmentation = False
            logging.warning("No LSTM trc files found. Using non augmented trc files instead.")
    if len(trc_files) == 0: # filtered files by default
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_filt' in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0: 
        trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_scaling' not in f.name]
    if len(trc_files) == 0:
        raise ValueError(f'No trc files found in {pose3d_dir}.')
    sorted(trc_files, key=natural_sort_key)

    if use_augmentation: 
        pose_model = 'LSTM'
    else: 
        pose_model = config_dict.get('pose').get('pose_model').upper()
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
        elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
        elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
        # else:
        #     raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')

    # Calculate subject heights
    if subject_height is None or subject_height == 0:
        subject_height = [1.75] * len(trc_files)
        logging.warning("No subject height found in Config.toml. Using default height of 1.75m.")
    elif isinstance(subject_height, str) and subject_height.lower() == 'auto':
        subject_height = []
        for trc_file in trc_files:
            try:
                trc_data, _, _, markers, _ = read_trc(trc_file)
                height = compute_height(
                    trc_data,
                    markers,
                    fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
                    close_to_zero_speed=close_to_zero_speed,
                    large_hip_knee_angles=large_hip_knee_angles,
                    trimmed_extrema_percent=trimmed_extrema_percent
                )
                if not np.isnan(height):
                    logging.info(f"Subject height automatically calculated for {os.path.basename(trc_file)}: {round(height,2)} m\n")
                else:
                    logging.warning(f"Could not compute height from {os.path.basename(trc_file)}. Using default height of {default_height}m.")
                    logging.warning(f"The person may be static, or crouched, or incorrectly detected. You may edit fastest_frames_to_remove_percent, close_to_zero_speed_m, large_hip_knee_angles, trimmed_extrema_percent, default_height in your Config.toml file.")
                    height = default_height
            except Exception as e:
                logging.warning(f"Could not compute height from {os.path.basename(trc_file)}. Using default height of {default_height}m.")
                logging.warning(f"The person may be static, or crouched, or incorrectly detected. You may edit fastest_frames_to_remove_percent, close_to_zero_speed_m, large_hip_knee_angles, trimmed_extrema_percent, default_height in your Config.toml file.")
                height = default_height
            subject_height.append(height)
    elif not type(subject_height) == list: # int or float
        subject_height = [subject_height]
    if len(subject_height) < len(trc_files):
        logging.warning("\nNumber of subject heights does not match number of TRC files. Missing heights are set to 1.75m.")
        subject_height += [1.75] * (len(trc_files) - len(subject_height))

    # Get subject masses
    if subject_mass is None or subject_mass == 0:
        subject_mass = [70] * len(trc_files)
        logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")
    elif not type(subject_mass) == list:
        subject_mass = [subject_mass]
    if len(subject_mass) < len(trc_files):
        logging.warning("Number of subject masses does not match number of TRC files. Missing masses are set to 70kg.\n")
        subject_mass += [70] * (len(trc_files) - len(subject_mass))

    # Perform scaling and IK for each trc file
    for p, trc_file in enumerate(trc_files):
        logging.info(f"Processing TRC file: {trc_file.resolve()}")

        logging.info("\nScaling...")
        perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, use_contacts_muscles, right_left_symmetry=right_left_symmetry, subject_height=subject_height[p], subject_mass=subject_mass[p], 
                        remove_scaling_setup=remove_scaling_setup, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent,close_to_zero_speed_m=close_to_zero_speed)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")
        
        logging.info("\nInverse Kinematics...")
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=remove_IK_setup)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tJoint angle data saved to {(kinematics_dir / (trc_file.stem + '.mot')).resolve()}\n")