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
import copy
from pathlib import Path
import numpy as np
np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
from lxml import etree
import logging
from anytree import PreOrderIter

import opensim

from Pose2Sim.common import natural_sort_key, euclidean_distance, trimmed_mean, read_trc, \
                            best_coords_for_measurements, compute_height
from Pose2Sim.skeletons import *

import locale 
locale.setlocale(locale.LC_NUMERIC, 'C')


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


def get_model_path(use_simple_model, osim_setup_dir):
    '''
    Retrieve the path of the OpenSim model file.

    INPUTS:
    - pose_model (str): Name of the model
    - osim_setup_dir (Path): Path to the OpenSim setup directory.

    OUTPUTS:
    - pose_model_path: (Path) Path to the OpenSim model file.
    '''

    if use_simple_model:
        pose_model_file = 'Model_Pose2Sim_simple.osim'
    else:
        pose_model_file = 'Model_Pose2Sim_muscles_flex.osim'

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

    pose_model = ''.join(pose_model.split('_')).lower()
    if pose_model == 'halpe68' or pose_model == 'halpe136':
        marker_file = 'Markers_Halpe68_136.xml'.lower()
    elif pose_model == 'coco133' or pose_model == 'coco133wrist':
        marker_file = 'Markers_Coco133.xml'.lower()
    else:
        marker_file = f'Markers_{pose_model}.xml'.lower()

    try:
        markers_path = [
            f for f in osim_setup_dir.glob('Markers_*.xml')
            if f.name.lower() == marker_file
        ][0]
    except:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

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

    pose_model = ''.join(pose_model.split('_')).lower()
    if pose_model == 'halpe68' or pose_model == 'halpe136':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Halpe68_136.xml'.lower()
    elif pose_model == 'coco133' or pose_model == 'coco133wrist':
        scaling_setup_file = 'Scaling_Setup_Pose2Sim_Coco133.xml'.lower()
    else:
        scaling_setup_file = f'Scaling_Setup_Pose2Sim_{pose_model}.xml'.lower()

    try:
        scaling_setup_path = [
            f for f in osim_setup_dir.glob('Scaling_Setup_Pose2Sim_*.xml')
            if f.name.lower() == scaling_setup_file
        ][0]
    except:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

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
    
    pose_model = ''.join(pose_model.split('_')).lower()
    if pose_model == 'halpe68' or pose_model == 'halpe136':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe68_136.xml'.lower()
    elif pose_model == 'coco133' or pose_model == 'coco133wrist':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco133.xml'.lower()
    elif pose_model == 'lstm':
        ik_setup_file = 'IK_Setup_Pose2Sim_withHands_LSTM.xml'.lower()
    else:
        ik_setup_file = f'IK_Setup_Pose2Sim_{pose_model}.xml'.lower()

    try:
        ik_setup_path = [
            f for f in osim_setup_dir.glob('IK_Setup_Pose2Sim_*.xml')
            if f.name.lower() == ik_setup_file
        ][0]
    except:
        raise ValueError(f"Pose model '{pose_model}' not supported yet.")

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
    
    # Filter out segment pairs where markers are missing in TRC
    # This is crucial for partial augmentation (e.g., upper-only or lower-only)
    valid_segment_pairs = []
    skipped_pairs = []
    for (pt1, pt2) in segment_pairs:
        if pt1 in markers and pt2 in markers:
            valid_segment_pairs.append((pt1, pt2))
        else:
            skipped_pairs.append((pt1, pt2))
    
    if skipped_pairs:
        logging.info(f"Skipped {len(skipped_pairs)} segment pairs due to missing markers in TRC:")
        logging.info(f"  First few: {skipped_pairs[:3]}{'...' if len(skipped_pairs) > 3 else ''}")
    
    if not valid_segment_pairs:
        raise ValueError("No valid segment pairs found in TRC file. Cannot perform scaling.")
    
    # Use only valid segment pairs for scaling
    segment_pairs = valid_segment_pairs

    # Get median segment lengths from Q_coords_scaling. Trimmed mean works better than mean or median
    trc_segment_lengths = np.array([euclidean_distance(Q_coords_scaling.iloc[:,markers.index(pt1)*3:markers.index(pt1)*3+3], 
                        Q_coords_scaling.iloc[:,markers.index(pt2)*3:markers.index(pt2)*3+3]) 
                        for (pt1,pt2) in segment_pairs])
    # trc_segment_lengths = np.median(trc_segment_lengths, axis=1)
    # trc_segment_lengths = np.mean(trc_segment_lengths, axis=1)
    trc_segment_lengths = np.array([trimmed_mean(arr, trimmed_extrema_percent=trimmed_extrema_percent) for arr in trc_segment_lengths])

    # Get model segment lengths
    # Build a mapping of marker name to model location
    model_marker_locs_dict = {}
    for marker_name in markers:
        # Check if this marker exists in the model
        try:
            marker_loc = unscaled_model.getMarkerSet().get(marker_name).getLocationInGround(unscaled_model.getWorkingState()).to_numpy()
            model_marker_locs_dict[marker_name] = marker_loc
        except:
            # If marker not found in model, skip it (e.g., virtual markers that don't exist in HALPE_26 model)
            logging.debug(f"Marker {marker_name} not found in model - will skip segments using this marker")
            continue
    
    # Calculate model segment lengths using the available markers
    model_segment_lengths = []
    valid_pairs_for_model = []
    for (pt1, pt2) in segment_pairs:
        if pt1 in model_marker_locs_dict and pt2 in model_marker_locs_dict:
            length = euclidean_distance(model_marker_locs_dict[pt1], model_marker_locs_dict[pt2])
            model_segment_lengths.append(length)
            valid_pairs_for_model.append((pt1, pt2))
        else:
            logging.debug(f"Skipping segment pair ({pt1}, {pt2}) - one or both markers not found in model")
    
    model_segment_lengths = np.array(model_segment_lengths)
    
    # Update segment_pairs to only include valid pairs
    if len(valid_pairs_for_model) < len(segment_pairs):
        logging.info(f"Reduced segment pairs from {len(segment_pairs)} to {len(valid_pairs_for_model)} due to missing markers in model")
        segment_pairs = valid_pairs_for_model
        # Also need to recalculate TRC segment lengths for matching pairs
        trc_segment_lengths = np.array([euclidean_distance(Q_coords_scaling.iloc[:,markers.index(pt1)*3:markers.index(pt1)*3+3], 
                            Q_coords_scaling.iloc[:,markers.index(pt2)*3:markers.index(pt2)*3+3]) 
                            for (pt1,pt2) in segment_pairs])
        trc_segment_lengths = np.array([trimmed_mean(arr, trimmed_extrema_percent=trimmed_extrema_percent) for arr in trc_segment_lengths])
    
    # Calculate ratio for each segment
    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root, right_left_symmetry=right_left_symmetry)
    
    # Filter segment_markers_dict to only include valid marker pairs
    # This prevents trying to calculate ratios for segments with missing markers
    filtered_segment_markers_dict = {}
    for key, pairs in segment_markers_dict.items():
        # Only keep pairs that are in our valid_segment_pairs
        valid_pairs = [pair for pair in pairs if pair in segment_pairs]
        if valid_pairs:
            filtered_segment_markers_dict[key] = valid_pairs
    
    segment_ratio_dict_temp = filtered_segment_markers_dict.copy()
    segment_ratio_dict_temp.update({key: np.mean([segment_ratios[segment_pairs.index(k)] 
                                            for k in filtered_segment_markers_dict[key]]) 
                                for key in filtered_segment_markers_dict.keys()})
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
        

def create_hybrid_scaling_setup(trc_file, osim_setup_dir, trc_markers_list, has_upper_virtual, has_lower_virtual):
    '''
    Create a hybrid scaling setup for partial augmentation scenarios.
    
    Key principle: For scaling, always use ORIGINAL markers (HALPE_26) to maintain consistency.
    Virtual markers will be utilized in IK stage with higher weights.
    
    This function only disables measurements that use markers not present in TRC.
    
    INPUTS:
    - trc_file (Path): The path to the TRC file
    - osim_setup_dir (Path): OpenSim setup directory
    - trc_markers_list (list): List of markers in TRC file
    - has_upper_virtual (bool): Whether upper limb virtual markers are present
    - has_lower_virtual (bool): Whether lower limb virtual markers are present
    
    OUTPUTS:
    - hybrid_scaling_path (Path): Path to the created hybrid scaling setup XML file
    '''
    
    # Start from HALPE_26 as base (uses ONLY original markers for scaling)
    halpe_scaling_path = osim_setup_dir / 'Scaling_Setup_Pose2Sim_Halpe26.xml'
    base_tree = etree.parse(halpe_scaling_path)
    base_root = base_tree.getroot()
    
    logging.info("  Creating hybrid scaling setup based on HALPE_26 (using original markers only)...")
    
    # Get all Measurement elements
    measurements = base_root.findall('.//Measurement')
    
    # Track disabled measurements
    disabled_measurements = []
    
    for measurement in measurements:
        measurement_name = measurement.get('name')
        marker_pairs = measurement.findall('.//MarkerPair')
        
        all_pairs_valid = True
        missing_markers = []
        
        for pair in marker_pairs:
            markers_elem = pair.find('markers')
            markers_text = markers_elem.text.strip().split()
            
            if len(markers_text) == 2:
                pt1, pt2 = markers_text
                
                # Check if markers exist in TRC (should be original HALPE_26 markers)
                if pt1 not in trc_markers_list:
                    all_pairs_valid = False
                    missing_markers.append(pt1)
                if pt2 not in trc_markers_list:
                    all_pairs_valid = False
                    missing_markers.append(pt2)
        
        # Disable measurement if any marker is missing
        if not all_pairs_valid:
            apply_elem = measurement.find('apply')
            if apply_elem is not None:
                apply_elem.text = 'false'
                disabled_measurements.append(f"{measurement_name} (missing: {', '.join(set(missing_markers))})")
    
    # Log changes
    if disabled_measurements:
        logging.info(f"  Disabled {len(disabled_measurements)} measurements due to missing markers:")
        for dm in disabled_measurements[:3]:
            logging.info(f"    - {dm}")
        if len(disabled_measurements) > 3:
            logging.info(f"    ... and {len(disabled_measurements) - 3} more")
    else:
        logging.info("  All HALPE_26 markers present in TRC - no measurements disabled")
    
    # Save hybrid scaling setup
    hybrid_scaling_path = trc_file.parent / f"{trc_file.stem}_hybrid_scaling_setup.xml"
    etree.indent(base_tree, space='\t', level=0)
    base_tree.write(str(hybrid_scaling_path), pretty_print=True, xml_declaration=True, encoding='utf-8')
    
    logging.info(f"  Created hybrid scaling setup: {hybrid_scaling_path.name}")
    logging.info("  Note: Virtual markers will be used in IK stage with higher weights")
    
    return hybrid_scaling_path


def create_hybrid_marker_set(trc_file, pose_model, osim_setup_dir):
    '''
    Create a hybrid marker set that combines available LSTM virtual markers with original pose markers.
    This allows partial augmentation (e.g., only upper or lower limb) to work with kinematics.
    
    INPUTS:
    - trc_file (Path): The path to the TRC file
    - pose_model (str): The pose model name
    - osim_setup_dir (Path): OpenSim setup directory
    
    OUTPUTS:
    - hybrid_markers_path (Path): Path to the created hybrid marker set XML file
    '''
    
    # Read markers from TRC file
    _, _, _, trc_markers, _ = read_trc(trc_file)
    
    # Get original pose model marker set as template
    # For LSTM mode, we need to use the underlying original model (HALPE_26)
    original_pose_model = 'HALPE_26' if pose_model == 'LSTM' else pose_model
    try:
        original_markers_path = get_markers_path(original_pose_model, osim_setup_dir)
    except:
        logging.warning(f"Could not find marker set for {original_pose_model}, using HALPE_26")
        original_markers_path = osim_setup_dir / 'Markers_Halpe26.xml'
    
    # Get LSTM marker set for virtual markers
    lstm_markers_path = osim_setup_dir / 'Markers_LSTM.xml'
    
    # Parse marker sets
    original_tree = etree.parse(original_markers_path)
    original_root = original_tree.getroot()
    
    lstm_tree = etree.parse(lstm_markers_path)
    lstm_root = lstm_tree.getroot()
    
    # Create hybrid marker set based on original template (deep copy)
    hybrid_root = copy.deepcopy(original_root)
    
    # Get the objects container where markers are stored
    hybrid_objects = hybrid_root.find('.//objects')
    
    # Get all LSTM virtual markers
    lstm_markers_dict = {}
    for marker in lstm_root.findall('.//Marker'):
        marker_name = marker.get('name')
        lstm_markers_dict[marker_name] = marker
    
    # Get existing marker names in the hybrid set (from original)
    existing_marker_names = {m.get('name') for m in hybrid_objects.findall('Marker')}
    
    # Add LSTM virtual markers that exist in TRC but not yet in hybrid set
    added_virtual_markers = []
    for trc_marker in trc_markers:
        # If this is a virtual marker (_study) that exists in TRC and LSTM set, but not in hybrid set
        if '_study' in trc_marker and trc_marker in lstm_markers_dict and trc_marker not in existing_marker_names:
            hybrid_objects.append(copy.deepcopy(lstm_markers_dict[trc_marker]))
            added_virtual_markers.append(trc_marker)
    
    # Create hybrid marker file path
    hybrid_markers_path = trc_file.parent / f"{trc_file.stem}_hybrid_markers.xml"
    
    # Write hybrid marker set
    hybrid_tree = etree.ElementTree(hybrid_root)
    etree.indent(hybrid_tree, space='\t', level=0)
    hybrid_tree.write(str(hybrid_markers_path), pretty_print=True, xml_declaration=True, encoding='utf-8')
    
    total_markers = len(existing_marker_names) + len(added_virtual_markers)
    logging.info(f"Created hybrid marker set: {total_markers} markers total")
    logging.info(f"  - Original markers: {len(existing_marker_names)}")
    logging.info(f"  - Added virtual markers: {len(added_virtual_markers)}")
    if added_virtual_markers:
        logging.info(f"  - Virtual markers added: {', '.join(added_virtual_markers[:5])}{'...' if len(added_virtual_markers) > 5 else ''}")
    
    return hybrid_markers_path


def perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, 
                    use_simple_model=False, right_left_symmetry=True, subject_height=1.75, subject_mass=70, 
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
    - use_simple_model (bool): Whether to use the model without constraints and muscles.
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
        unscaled_model_path = get_model_path(use_simple_model, osim_setup_dir)
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        unscaled_model = opensim.Model(str(unscaled_model_path))
        
        # Check if this is a hybrid augmentation scenario (partial LSTM markers)
        # Read TRC to check available markers
        _, _, _, trc_markers_list, _ = read_trc(trc_file)
        
        # Define standard LSTM virtual markers for lower and upper limbs
        # Note: TRC files ALWAYS contain original markers + virtual markers (if augmented)
        # We need to check if ALL expected virtual markers are present
        lstm_lower_markers = [
            'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
            'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
            'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
            'r_5meta_study', 'r_calc_study', 'L_knee_study', 
            'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
            'L_toe_study', 'L_calc_study', 'L_5meta_study', 
            'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
            'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
            'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
            'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
            'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        
        lstm_upper_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]
        
        # Check which virtual markers actually exist in TRC
        lower_markers_present = [m for m in lstm_lower_markers if m in trc_markers_list]
        upper_markers_present = [m for m in lstm_upper_markers if m in trc_markers_list]
        
        # Calculate coverage percentage
        lower_coverage = len(lower_markers_present) / len(lstm_lower_markers) if lstm_lower_markers else 0
        upper_coverage = len(upper_markers_present) / len(lstm_upper_markers) if lstm_upper_markers else 0
        
        # Partial augmentation: not all virtual markers are present
        # Full augmentation: both lower AND upper are complete
        has_full_lower = lower_coverage >= 0.95
        has_full_upper = upper_coverage >= 0.95

        # Full augmentation: both complete OR both empty
        is_full_augmentation = (has_full_lower and has_full_upper)
        
        # Partial augmentation: anything else (one complete, one empty OR incomplete coverage)
        is_partial_augmentation = not is_full_augmentation
        
        if is_partial_augmentation:
            logging.info(f"Detected partial marker augmentation. Creating hybrid marker set and scaling setup...")
            logging.info(f"  Lower limb virtual markers: {len(lower_markers_present)}/{len(lstm_lower_markers)} ({lower_coverage*100:.1f}%)")
            logging.info(f"  Upper limb virtual markers: {len(upper_markers_present)}/{len(lstm_upper_markers)} ({upper_coverage*100:.1f}%)")
            
            # Create hybrid marker set (HALPE_26 original markers + available virtual markers)
            markers_path = create_hybrid_marker_set(trc_file, pose_model, osim_setup_dir)
            
            # Create hybrid scaling setup that intelligently uses virtual markers where available
            hybrid_scaling_path = create_hybrid_scaling_setup(
                trc_file, osim_setup_dir, trc_markers_list, 
                has_full_upper, has_full_lower
            )
            scaling_pose_model = None  # Will use hybrid_scaling_path directly
        else:
            # Standard marker set selection (full augmentation or no augmentation)
            markers_path = get_markers_path(pose_model, osim_setup_dir)
            scaling_pose_model = pose_model
            hybrid_scaling_path = None
        
        # Add markers to model
        markerset = opensim.MarkerSet(str(markers_path))
        unscaled_model.set_MarkerSet(markerset)
        # Initialize and save model with markers
        unscaled_model.initSystem()
        scaled_model_path = str((kinematics_dir / (trc_file.stem + '.osim')).resolve())
        unscaled_model.printToXML(scaled_model_path)

        # Load scaling setup
        if hybrid_scaling_path is not None:
            # Use the hybrid scaling setup for partial augmentation
            scaling_path = hybrid_scaling_path
        else:
            # Use standard scaling setup
            scaling_path = get_scaling_setup(scaling_pose_model, osim_setup_dir)
        
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
        logging.error(f"Error during scaling for {trc_file}: {e}.")
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
        # Check if this was a partial augmentation case
        scaled_model_path = (kinematics_dir / (trc_file.stem + '.osim')).resolve()
        
        # Read TRC to check if this is partial augmentation
        _, _, _, trc_markers_list, _ = read_trc(trc_file)
        
        # Define LSTM virtual markers
        lstm_lower_markers = [
            'r.ASIS_study', 'L.ASIS_study', 'r.PSIS_study',
            'L.PSIS_study', 'r_knee_study', 'r_mknee_study', 
            'r_ankle_study', 'r_mankle_study', 'r_toe_study', 
            'r_5meta_study', 'r_calc_study', 'L_knee_study', 
            'L_mknee_study', 'L_ankle_study', 'L_mankle_study',
            'L_toe_study', 'L_calc_study', 'L_5meta_study', 
            'r_shoulder_study', 'L_shoulder_study', 'C7_study', 
            'r_thigh1_study', 'r_thigh2_study', 'r_thigh3_study',
            'L_thigh1_study', 'L_thigh2_study', 'L_thigh3_study',
            'r_sh1_study', 'r_sh2_study', 'r_sh3_study', 'L_sh1_study',
            'L_sh2_study', 'L_sh3_study', 'RHJC_study', 'LHJC_study']
        
        lstm_upper_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]
        
        # Check coverage
        lower_markers_present = [m for m in lstm_lower_markers if m in trc_markers_list]
        upper_markers_present = [m for m in lstm_upper_markers if m in trc_markers_list]
        lower_coverage = len(lower_markers_present) / len(lstm_lower_markers) if lstm_lower_markers else 0
        upper_coverage = len(upper_markers_present) / len(lstm_upper_markers) if lstm_upper_markers else 0
        
        has_full_lower = lower_coverage >= 0.95
        has_full_upper = upper_coverage >= 0.95
        is_partial_augmentation = not (has_full_lower and has_full_upper)
        
        # For partial augmentation with upper limb virtual markers, create hybrid IK setup
        if is_partial_augmentation and has_full_upper and pose_model == 'LSTM':
            logging.info("  Creating hybrid IK setup for partial upper limb augmentation...")
            # Start from HALPE_26 IK setup as base
            halpe_ik_path = get_IK_Setup('HALPE_26', osim_setup_dir)
            ik_tree = etree.parse(halpe_ik_path)
            ik_root = ik_tree.getroot()
            
            # Define weight adjustments for virtual markers (higher weight for more accurate virtual markers)
            virtual_marker_weights = {
                'r_melbow_study': 2.0,  # Virtual elbow markers get higher weight
                'L_melbow_study': 2.0,
                'r_mwrist_study': 2.0,  # Virtual wrist markers get higher weight
                'L_mwrist_study': 2.0,
                'r_lelbow_study': 1.5,
                'L_lelbow_study': 1.5,
                'r_lwrist_study': 1.5,
                'L_lwrist_study': 1.5
            }
            
            # Lower weights for original markers that have virtual equivalents
            original_marker_weights = {
                'RElbow': 0.5,  # Lower weight since we have r_melbow_study
                'LElbow': 0.5,
                'RWrist': 0.5,  # Lower weight since we have r_mwrist_study
                'LWrist': 0.5
            }
            
            # Get IKTaskSet
            ik_task_set = ik_root.find('.//IKTaskSet/objects')
            if ik_task_set is not None:
                # Update existing marker weights
                for task in ik_task_set.findall('IKMarkerTask'):
                    marker_name = task.get('name')
                    weight_elem = task.find('weight')
                    if marker_name in original_marker_weights and weight_elem is not None:
                        weight_elem.text = str(original_marker_weights[marker_name])
                
                # Add virtual marker tasks
                for virtual_marker, weight in virtual_marker_weights.items():
                    if virtual_marker in trc_markers_list:
                        # Create new IKMarkerTask for virtual marker
                        new_task = etree.Element('IKMarkerTask', name=virtual_marker)
                        apply_elem = etree.SubElement(new_task, 'apply')
                        apply_elem.text = 'true'
                        weight_elem = etree.SubElement(new_task, 'weight')
                        weight_elem.text = str(weight)
                        ik_task_set.append(new_task)
            
            # Save temporary hybrid IK setup
            ik_path_temp = str(kinematics_dir / (trc_file.stem + '_ik_setup.xml'))
        else:
            # Standard IK setup
            ik_path = get_IK_Setup(pose_model, osim_setup_dir)
            ik_tree = etree.parse(ik_path)
            ik_root = ik_tree.getroot()
            ik_path_temp = str(kinematics_dir / (trc_file.stem + '_ik_setup.xml'))
        
        # Common IK setup updates
        output_motion_file = Path(kinematics_dir, trc_file.stem + '.mot').resolve()
        
        if not trc_file.exists():
            raise FileNotFoundError(f"TRC file does not exist: {trc_file}")
        _, _, time_col, _, _ = read_trc(trc_file)
        start_time, end_time = time_col.iloc[0], time_col.iloc[-1]

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
    use_simple_model = config_dict.get('kinematics').get('use_simple_model', False)
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
        perform_scaling(trc_file, pose_model, kinematics_dir, osim_setup_dir, use_simple_model, right_left_symmetry=right_left_symmetry, subject_height=subject_height[p], subject_mass=subject_mass[p], 
                        remove_scaling_setup=remove_scaling_setup, fastest_frames_to_remove_percent=fastest_frames_to_remove_percent, large_hip_knee_angles=large_hip_knee_angles, trimmed_extrema_percent=trimmed_extrema_percent,close_to_zero_speed_m=close_to_zero_speed)
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tScaled model saved to {(kinematics_dir / (trc_file.stem + '_scaled.osim')).resolve()}")
        
        logging.info("\nInverse Kinematics...")
        import time
        start_time = time.time()
        perform_IK(trc_file, kinematics_dir, osim_setup_dir, pose_model, remove_IK_setup=remove_IK_setup)
        end_time = time.time()
        print(f"\tIK took {round(end_time - start_time, 2)} seconds for {trc_file.name}.")
        logging.info(f"\tDone. OpenSim logs saved to {opensim_logs_file.resolve()}.")
        logging.info(f"\tJoint angle data saved to {(kinematics_dir / (trc_file.stem + '.mot')).resolve()}\n")