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
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree
import logging
import opensim


## FUNCTIONS
def find_config_and_pose3d(project_dir):
    """
    Find configuration files and associated pose-3d directories in the project directory.

    Args:
        project_dir (str): The root directory of the project.

    Returns:
        list: A list of tuples containing the config path and the corresponding pose-3d directory.
    """
    config_paths = []
    for root, dirs, files in os.walk(project_dir):
        if 'Config.toml' in files:
            config_path = Path(root) / 'Config.toml'
            possible_pose3d_dir = Path(root) / 'pose-3d'
            if not possible_pose3d_dir.exists():
                possible_pose3d_dir = Path(root).parent / 'pose-3d'
            if possible_pose3d_dir.exists():
                config_paths.append((config_path, possible_pose3d_dir))
            else:
                logging.warning(f"No pose-3d directory found for config: {config_path}")
    return config_paths


def get_grouped_files(directory, pattern='*.trc'):
    """
    Group TRC files by person ID or treat them as single-person if no ID is found.

    Args:
        directory (str): The directory containing TRC files.
        pattern (str): The file pattern to search for.

    Returns:
        dict: A dictionary grouping TRC files by person ID.
    """
    files = list(Path(directory).glob(pattern))
    grouped_files = defaultdict(list)

    for file in files:
        parts = file.stem.split('_')
        if len(parts) > 2 and 'P' in parts[2]:  # Multi-person file naming convention
            person_id = parts[2]
        else:
            person_id = "SinglePerson"
        grouped_files[person_id].append(file)

    return grouped_files


def process_all_groups(config_dict):
    """
       Process all groups (single or multi-person) based on the configuration.

       Args:
           config_dict (dict): The configuration dictionary containing project details.
    """
    logging.info("Processing all groups in the project.")
    project_dir = config_dict.get('project', {}).get('project_dir')
    config_and_pose3d_paths = find_config_and_pose3d(project_dir)

    for config_path, pose3d_dir in config_and_pose3d_paths:
        logging.info(f"Processing setup with config: {config_path}")

        trc_groups = get_grouped_files(pose3d_dir)
        trial_name = Path(pose3d_dir).parent.name  # Use the parent directory name as the trial name

        for person_id, trc_files in trc_groups.items():
            filtered_trc_files = load_trc(config_dict, trc_files)

            # Ensure output directory includes the trial name
            trial_output_dir = get_output_dir(Path(config_dict['project']['project_dir']).parent / trial_name, person_id)
            perform_scaling(config_dict, person_id, filtered_trc_files, trial_output_dir)
            perform_inverse_kinematics(config_dict, person_id, filtered_trc_files, trial_output_dir)

def load_trc(config_dict, trc_files):
    """
    Load and filter TRC files according to the configuration.

    Args:
        config_dict (dict): The configuration dictionary.
        trc_files (list): A list of TRC file paths.

    Returns:
        list: A list of filtered TRC files based on the criteria specified in the configuration.
    """
    opensim_config = config_dict.get('opensim', {})
    use_lstm = opensim_config.get('use_augmentation', False)
    load_trc_name = opensim_config.get('load_trc_name', 'default')

    # Filter out any scaled TRC files
    unscaled_trc_files = [file for file in trc_files if '_scaling' not in str(file)]

    logging.info(f"Starting TRC file filtering with criteria: use_lstm = {use_lstm}, load_trc_name = {load_trc_name}")
    logging.info(f"Initial list of TRC files: {unscaled_trc_files}")

    # Initialize the list to store filtered TRC files
    trc_files = []

    # Check for LSTM files if LSTM is being used
    if use_lstm:
        lstm_files = [file for file in unscaled_trc_files if '_LSTM.trc' in str(file)]
        if not lstm_files:
            raise FileNotFoundError("No LSTM TRC file found in the provided list.")
        trc_files.extend(lstm_files)

    # Check for default or filtered TRC files
    if load_trc_name == 'default':
        default_files = [file for file in unscaled_trc_files if '_LSTM' not in str(file) and '_filt_butterworth' not in str(file)]
        trc_files.extend(default_files)
    elif load_trc_name == 'filtered':
        filtered_files = [file for file in unscaled_trc_files if '_filt_butterworth' in str(file) and '_LSTM' not in str(file)]
        trc_files.extend(filtered_files)

    # If no TRC files are found after filtering, raise an error
    if not trc_files:
        logging.error(f"No suitable TRC files found with the specified criteria: use_lstm = {use_lstm}, load_trc_name = {load_trc_name}")
        raise FileNotFoundError(f"No suitable TRC files found in the provided list with the specified criteria: use_lstm = {use_lstm}, load_trc_name = {load_trc_name}")

    logging.info(f"Filtered TRC files: {trc_files}")

    return trc_files


def read_trc(trc_path):
    """
    Read a TRC file and extract its contents.

    Args:
        trc_path (str): The path to the TRC file.

    Returns:
        tuple: A tuple containing the Q coordinates, frames column, time column, and header.
    """
    try:
        logging.info(f"Attempting to read TRC file: {trc_path}")
        with open(trc_path, 'r') as trc_file:
            header = [next(trc_file) for _ in range(5)]
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        return Q_coords, frames_col, time_col, header
    except Exception as e:
        logging.error(f"Error reading TRC file at {trc_path}: {e}")
        raise


def make_trc_with_Q(Q, header, trc_path):
    """
    Write the processed Q coordinates back to a TRC file.

    Args:
        Q (pd.DataFrame): The Q coordinates data.
        header (list): The header of the original TRC file.
        trc_path (str): Path to save the new TRC file.
    """
    header_2_split = header[2].split('\t')
    header_2_split[2] = str(len(Q))
    header_2_split[-1] = str(len(Q))
    header[2] = '\t'.join(header_2_split) + '\n'

    time = pd.Series(np.arange(len(Q)) / float(header_2_split[0]), name='t')
    Q.insert(0, 't', time)

    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')


def get_key(config_dict):
    """
    Determine the key for the OpenSim model and setup files based on the configuration.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        str: The key used to select the model and setup files.
    """
    use_augmentation = config_dict.get('opensim', {}).get('use_augmentation', False)

    if use_augmentation:
        return 'LSTM'

    pose_model = config_dict.get('pose', {}).get('pose_model', '').upper()
    if not pose_model:
        raise ValueError(f"Invalid or missing 'pose_model' in config: {pose_model}")

    return pose_model


def get_OpenSim_Setup():
    """
    Locate the OpenSim setup directory within the Pose2Sim package.

    Returns:
        Path: The path to the OpenSim setup directory.
    """
    pose2sim_path = Path(sys.modules['Pose2Sim'].__file__).resolve().parent
    setup_dir = pose2sim_path / 'OpenSim_Setup'
    return setup_dir


def get_Model(config_dict):
    """
    Retrieve the OpenSim model file path based on the configuration.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        str: The path to the OpenSim model file.
    """
    setup_key = get_key(config_dict)
    setup_dir = get_OpenSim_Setup()

    if setup_key == 'LSTM':
        pose_model_file = 'Model_Pose2Sim_LSTM.osim'
    elif setup_key == 'BLAZEPOSE':
        pose_model_file = 'Model_Pose2Sim_Blazepose.osim'
    elif setup_key == 'BODY_25':
        pose_model_file = 'Model_Pose2Sim_Body25.osim'
    elif setup_key == 'BODY_25B':
        pose_model_file = 'Model_Setup_Pose2Sim_Body25b.osim'
    elif setup_key == 'BODY_135':
        pose_model_file = 'Model_Pose2Sim_Body135.osim'
    elif setup_key == 'COCO_17':
        pose_model_file = 'Model_Pose2Sim_Coco17.osim'
    elif setup_key == 'COCO_133':
        pose_model_file = 'Model_Pose2Sim_Coco133.osim'
    elif setup_key == 'HALPE_26':
        pose_model_file = 'Model_Pose2Sim_Halpe26.osim'
    elif setup_key == 'HALPE_68':
        pose_model_file = 'Model_Pose2Sim_Halpe68_136.osim'
    else:
        raise ValueError(f"pose_model '{setup_key}' not found.")

    pose_model_path = os.path.join(setup_dir, pose_model_file)
    return pose_model_path


def get_Scale_Setup(config_dict):
    """
    Retrieve the OpenSim scaling setup file path based on the configuration.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        str: The path to the OpenSim scaling setup file.
    """
    setup_key = get_key(config_dict)
    setup_dir = get_OpenSim_Setup()

    if setup_key == 'LSTM':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_LSTM.xml'
    elif setup_key == 'BLAZEPOSE':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Blazepose.xml'
    elif setup_key == 'BODY_25':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Body25.xml'
    elif setup_key == 'BODY_25B':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Body25b.xml'
    elif setup_key == 'BODY_135':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Body135.xml'
    elif setup_key == 'COCO_17':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Coco17.xml'
    elif setup_key == 'COCO_133':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Coco133.xml'
    elif setup_key == 'HALPE_26':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Halpe26.xml'
    elif setup_key == 'HALPE_68':
        scale_setup_file = 'Scaling_Setup_Pose2Sim_Halpe68_136.xml'
    else:
        raise ValueError(f"pose_model '{setup_key}' not found.")

    scale_setup_path = os.path.join(setup_dir, scale_setup_file)
    return scale_setup_path


def get_IK_Setup(config_dict):
    """
    Retrieve the OpenSim inverse kinematics setup file path based on the configuration.

    Args:
        config_dict (dict): The configuration dictionary.

    Returns:
        str: The path to the OpenSim inverse kinematics setup file.
    """
    setup_key = get_key(config_dict)
    setup_dir = get_OpenSim_Setup()

    if setup_key == 'LSTM':
        ik_setup_file = 'IK_Setup_Pose2Sim_LSTM.xml'
    elif setup_key == 'BLAZEPOSE':
        ik_setup_file = 'IK_Setup_Pose2Sim_Blazepose.xml'
    elif setup_key == 'BODY_25':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25.xml'
    elif setup_key == 'BODY_25B':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body25b.xml'
    elif setup_key == 'BODY_135':
        ik_setup_file = 'IK_Setup_Pose2Sim_Body135.xml'
    elif setup_key == 'COCO_17':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco17.xml'
    elif setup_key == 'COCO_133':
        ik_setup_file = 'IK_Setup_Pose2Sim_Coco133.xml'
    elif setup_key == 'HALPE_26':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe26.xml'
    elif setup_key == 'HALPE_68':
        ik_setup_file = 'IK_Setup_Pose2Sim_Halpe68_136.xml'
    else:
        raise ValueError(f"pose_model '{setup_key}' not found.")

    ik_setup_path = os.path.join(setup_dir, ik_setup_file)
    return ik_setup_path


def get_output_dir(config_dir, person_id):
    """
    Determines the correct output directory based on the configuration and the person identifier.

    Args:
        config_dir (Path): The root directory where the configuration file is located.
        person_id (str): Identifier for the person (e.g., 'SinglePerson', 'P1').

    Returns:
        Path: The path where the output files should be stored.
    """
    output_dir = config_dir / 'kinematics'  # Assuming 'kinematics' as the default output subdirectory

    # Append the person_id to the output directory if it's a multi-person setup
    if person_id != "SinglePerson":
        output_dir = output_dir / person_id

    logging.debug(f"Output directory determined as: {output_dir}")

    # Create the directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def perform_scaling(config_dict, person_id, trc_files, output_dir):
    """
    Perform scaling on the TRC files according to the OpenSim configuration.

    Args:
        config_dict (dict): The configuration dictionary.
        person_id (str): The person identifier (e.g., 'SinglePerson', 'P1').
        trc_files (list): List of TRC files to be processed.
        output_dir (Path): The directory where the output files should be saved.
    """
    geometry_path = Path(get_OpenSim_Setup()) / 'Geometry'
    geometry_path_str = str(geometry_path)
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path_str)

    try:
        athlete_config = config_dict.get('project', {})
        athlete_height = athlete_config.get('participant_height', -1)
        athlete_weight = athlete_config.get('participant_mass', -1)

        if person_id == "SinglePerson":
            if not isinstance(athlete_height, float) or not isinstance(athlete_weight, float):
                raise ValueError("For a single person configuration, 'participant_height' and 'participant_mass' must be floats.")
        else:
            if person_id.startswith("P"):
                try:
                    person_idx = int(person_id.replace('P', '')) - 1
                    athlete_height = athlete_height[person_idx]
                    athlete_weight = athlete_weight[person_idx]
                except (ValueError, IndexError) as e:
                    raise ValueError(f"Error processing multi-person data for '{person_id}': {e}")
            else:
                raise ValueError(f"Unexpected person_id format: '{person_id}'")

        logging.debug(f"Performing scaling. Output directory: {output_dir}")

        pose_model = get_Model(config_dict)
        if not pose_model:
            raise ValueError(f"Model path not found for pose_model: {pose_model}")

        for trc_file in trc_files:
            trc_file = Path(trc_file)
            scaling_path = get_Scale_Setup(config_dict)

            Q_coords, _, _, header = read_trc(trc_file)

            Q_diff = Q_coords.diff(axis=0).sum(axis=1)
            Q_diff = Q_diff[Q_diff != 0]
            min_speed_indices = Q_diff.abs().nsmallest(int(len(Q_diff) * 0.1)).index
            Q_coords_scaling = Q_coords.iloc[min_speed_indices].reset_index(drop=True)

            trc_scaling_path = trc_file.parent / (trc_file.stem + '_scaling.trc')
            make_trc_with_Q(Q_coords_scaling, header, str(trc_scaling_path))

            scaling_file_path = str(trc_file.parent / (trc_file.stem + '_' + Path(scaling_path).name))
            scaled_model_path = (output_dir / (trc_file.stem + '_scaled.osim')).resolve()
            scaling_tree = etree.parse(str(scaling_path))
            scaling_root = scaling_tree.getroot()

            scaling_root[0].find('mass').text = str(athlete_weight)
            scaling_root[0].find('height').text = str(athlete_height)
            scaling_root[0].find('GenericModelMaker').find('model_file').text = str(pose_model)
            scaling_root[0].find('ModelScaler').find('marker_file').text = trc_scaling_path.name
            scaling_root[0].find('ModelScaler').find('time_range').text = '0 ' + str(Q_coords_scaling['t'].iloc[-1])
            scaling_root[0].find('ModelScaler').find('output_model_file').text = str(scaled_model_path)
            scaling_root[0].find('MarkerPlacer').find('marker_file').text = trc_scaling_path.name
            scaling_root[0].find('MarkerPlacer').find('time_range').text = '0 ' + str(Q_coords_scaling['t'].iloc[-1])
            scaling_root[0].find('MarkerPlacer').find('output_model_file').text = str(scaled_model_path)
            scaling_tree.write(scaling_file_path)

            logging.debug(f"Running ScaleTool with scaling file: {scaling_file_path}")
            opensim.ScaleTool(scaling_file_path).run()

    except Exception as e:
        logging.error(f"Error during scaling for {person_id}: {e}")
        raise

def perform_inverse_kinematics(config_dict, person_id, trc_files, output_dir):
    """
    Perform inverse kinematics on the TRC files according to the OpenSim configuration.

    Args:
        config_dict (dict): The configuration dictionary.
        person_id (str): The person identifier (e.g., 'SinglePerson', 'P1').
        trc_files (list): List of TRC files to be processed.
        output_dir (Path): The directory where the output files should be saved.
    """
    try:
        logging.debug(f"Performing inverse kinematics. Output directory: {output_dir}")

        for trc_file in trc_files:
            trc_file_path = Path(trc_file).resolve()
            scaled_model_path = Path(output_dir) / (trc_file_path.stem + '_scaled.osim')

            ik_setup_path = get_IK_Setup(config_dict)
            Q_coords, frames_col, time_col, header = read_trc(trc_file_path)
            ik_time_range = config_dict.get('opensim', {}).get('IK_timeRange', [])

            if not ik_time_range:
                start_time = time_col.iloc[0]
                end_time = time_col.iloc[-1]
            else:
                start_time, end_time = ik_time_range[0], ik_time_range[1]

            ik_file_path = Path(trc_file_path.parent / (trc_file_path.stem + '_' + Path(ik_setup_path).name)).resolve()
            scaled_model_path = scaled_model_path.resolve()
            output_motion_file = Path(output_dir, trc_file_path.stem + '.mot').resolve()

            ik_tree = etree.parse(ik_setup_path)
            ik_root = ik_tree.getroot()
            ik_root.find('.//model_file').text = str(scaled_model_path)
            ik_root.find('.//time_range').text = f'{start_time} {end_time}'
            ik_root.find('.//output_motion_file').text = str(output_motion_file)
            ik_root.find('.//marker_file').text = str(trc_file_path)
            ik_tree.write(ik_file_path)

            logging.info(f"Running InverseKinematicsTool with TRC file: {trc_file_path}")
            if not trc_file_path.exists():
                raise FileNotFoundError(f"TRC file does not exist: {trc_file_path}")

            logging.debug(f"Running InverseKinematicsTool with IK setup file: {ik_file_path}")
            opensim.InverseKinematicsTool(str(ik_file_path)).run()

    except Exception as e:
        logging.error(f"Error during IK for {person_id}: {e}")
        raise


def opensimProcessing(config_dict):
    logging.info("Starting OpenSim processing...")
    process_all_groups(config_dict)
    logging.info("OpenSim processing completed successfully.")