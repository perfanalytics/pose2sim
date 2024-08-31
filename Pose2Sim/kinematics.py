import os
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree
import opensim

# Mapping pose_model to respective model and setup files
model_map = {
    'LSTM': 'Model_Pose2Sim_LSTM.osim',
    'Blazepose': 'Model_Pose2Sim_Blazepose.osim',
    'BODY_25': 'Model_Pose2Sim_Body25.osim',
    'BODY_25B': 'Model_Setup_Pose2Sim_Body25b.osim',
    'BODY_135': 'Model_Pose2Sim_Body135.osim',
    'COCO_17': 'Model_Pose2Sim_Coco17.osim',
    'COCO_133': 'Model_Pose2Sim_Coco133.osim',
    'HALPE_26': 'Model_Pose2Sim_Halpe26.osim',
    'HALPE_68': 'Model_Pose2Sim_Halpe68_136.osim'
}

scale_map = {
    'LSTM': 'Scaling_Setup_Pose2Sim_LSTM.xml',
    'Blazepose': 'Scaling_Setup_Pose2Sim_Blazepose.xml',
    'BODY_25': 'Scaling_Setup_Pose2Sim_Body25.xml',
    'BODY_25B': 'Scaling_Setup_Pose2Sim_Body25b.xml',
    'BODY_135': 'Scaling_Setup_Pose2Sim_Body135.xml',
    'COCO_17': 'Scaling_Setup_Pose2Sim_Coco17.xml',
    'COCO_133': 'Scaling_Setup_Pose2Sim_Coco133.xml',
    'HALPE_26': 'Scaling_Setup_Pose2Sim_Halpe26.xml',
    'HALPE_68': 'Scaling_Setup_Pose2Sim_Halpe68_136.xml'
}

ik_map = {
    'LSTM': 'IK_Setup_Pose2Sim_LSTM.xml',
    'Blazepose': 'IK_Setup_Pose2Sim_Blazepose.xml',
    'BODY_25': 'IK_Setup_Pose2Sim_Body25.xml',
    'BODY_25B': 'IK_Setup_Pose2Sim_Body25b.xml',
    'BODY_135': 'IK_Setup_Pose2Sim_Body135.xml',
    'COCO_17': 'IK_Setup_Pose2Sim_Coco17.xml',
    'COCO_133': 'IK_Setup_Pose2Sim_Coco133.xml',
    'HALPE_26': 'IK_Setup_Pose2Sim_Halpe26.xml',
    'HALPE_68': 'IK_Setup_Pose2Sim_Halpe68_136.xml'
}


def load_trc(config_dict):
    # Determine the directory where TRC files are stored
    trc_dir = os.path.join(os.path.dirname(config_dict.get('opensim', {}).get('output_dir', '')), 'pose-3d')

    opensim_config = config_dict.get('opensim', {})
    use_lstm = opensim_config.get('LSTM', False)
    load_trc_name = opensim_config.get('load_trc_name', 'default')

    trc_filename = None

    # Ensure we are only looking at .trc files
    valid_trc_files = [file for file in os.listdir(trc_dir) if file.endswith('.trc')]

    # Determine which TRC file to load based on LSTM and load_trc_name settings
    if use_lstm:
        for file in valid_trc_files:
            if file.endswith('_LSTM.trc'):
                trc_filename = file
                break
        if trc_filename is None:
            raise FileNotFoundError("No LSTM TRC file found in the 'pose-3d' directory.")
    else:
        if load_trc_name == 'default':
            for file in valid_trc_files:
                if '_filt_butterworth' not in file and '_LSTM' not in file:
                    trc_filename = file
                    break
        elif load_trc_name == 'filtered':
            for file in valid_trc_files:
                if '_filt_butterworth' in file and '_LSTM' not in file:
                    trc_filename = file
                    break
        else:
            raise ValueError(f"Invalid load_trc_name '{load_trc_name}' specified in config.")

        if trc_filename is None:
            raise FileNotFoundError(
                f"No suitable TRC file found in the 'pose-3d' directory with the specified criteria: load_trc_name = {load_trc_name}")

    trc_path = os.path.join(trc_dir, trc_filename)
    return trc_path


import logging


def read_trc(trc_path):
    try:
        logging.info(f"Attempting to read TRC file: {trc_path}")
        with open(trc_path, 'r') as trc_file:  # Specify encoding
            header = [next(trc_file) for line in range(5)]
        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding='utf-8')  # Specify encoding
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        return Q_coords, frames_col, time_col, header
    except Exception as e:
        logging.error(f"Error reading TRC file at {trc_path}: {e}")
        raise


def get_grouped_files(directory, pattern='*.trc', excluded_pattern=None):
    if excluded_pattern:
        files = [file for file in Path(directory).glob(pattern) if not file.match(excluded_pattern)]
    else:
        files = list(Path(directory).glob(pattern))
    grouped_files = defaultdict(list)
    for file in files:
        video_id = file.stem.split('_')[0]
        grouped_files[video_id].append(file)
    return list(grouped_files.values())


def make_trc_with_Q(Q, header, trc_path):
    header_2_split = header[2].split('\t')
    header_2_split[2] = str(len(Q))
    header_2_split[-1] = str(len(Q))
    header[2] = '\t'.join(header_2_split) + '\n'

    time = pd.Series(np.arange(len(Q)) / float(header_2_split[0]), name='t')
    Q.insert(0, 't', time)

    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')


def get_Scale_Setup(config_dict):
    try:
        opensim_config = config_dict.get('opensim', {})
        use_lstm = opensim_config.get('LSTM', False)
        setup_dir = opensim_config.get('setup_dir', 'Pose2Sim/OpenSim_Setup')

        # Determine the IK setup file based on the config
        if use_lstm:
            scale_setup_file = scale_map['LSTM']
        else:
            pose_model = config_dict.get('pose', {}).get('pose_model', None)
            if pose_model and pose_model in scale_map:
                scale_setup_file = scale_map[pose_model]
            else:
                raise ValueError(f"pose_model '{pose_model}' not found in Scale map and LSTM is not set to True.")

        scale_setup_path = os.path.join(setup_dir, scale_setup_file)
        return scale_setup_path
    except Exception as e:
        logging.error(f"Error determining Scale setup path: {e}")
        raise


def get_Model(config_dict):
    try:
        opensim_config = config_dict.get('opensim', {})
        use_lstm = opensim_config.get('LSTM', False)
        setup_dir = opensim_config.get('setup_dir', 'Pose2Sim/OpenSim_Setup')

        # Determine the IK setup file based on the config
        if use_lstm:
            pose_model_file = model_map['LSTM']
        else:
            pose_model = config_dict.get('pose', {}).get('pose_model', None)
            if pose_model and pose_model in model_map:
                pose_model_file = model_map[pose_model]
            else:
                raise ValueError(f"pose_model '{pose_model}' not found in Model map and LSTM is not set to True.")

        pose_model_path = os.path.join(setup_dir, pose_model_file)
        return pose_model_path
    except Exception as e:
        logging.error(f"Error determining Model setup path: {e}")
        raise


def get_IK_Setup(config_dict):
    try:
        opensim_config = config_dict.get('opensim', {})
        use_lstm = opensim_config.get('LSTM', False)
        setup_dir = opensim_config.get('setup_dir', 'Pose2Sim/OpenSim_Setup')

        # Determine the IK setup file based on the config
        if use_lstm:
            ik_setup_file = ik_map['LSTM']
        else:
            pose_model = config_dict.get('pose', {}).get('pose_model', None)
            if pose_model and pose_model in ik_map:
                ik_setup_file = ik_map[pose_model]
            else:
                raise ValueError(f"pose_model '{pose_model}' not found in IK map and LSTM is not set to True.")

        ik_setup_path = os.path.join(setup_dir, ik_setup_file)
        return ik_setup_path

    except Exception as e:
        logging.error(f"Error determining IK setup path: {e}")
        raise


def perform_scaling(config_dict):
    # Get the directory of the current script
    current_dir = Path(__file__).parent
    # Construct the path to the 'Geometry' directory 
    geometry_path = current_dir / 'OpenSim_Setup' / 
    geometry_path_str = str(geometry_path)
    print(f"Geometry path: {geometry_path_str}")
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(geometry_path_str)

    try:
        athlete_config = config_dict.get('project', {})
        athlete_height = athlete_config.get('participant_height', -1)
        athlete_weight = athlete_config.get('participant_mass', -1)

        if athlete_height == -1 or athlete_weight == -1:
            raise ValueError(f"config not found or height/weight not found")

        output_dir = config_dict.get('opensim', {}).get('output_dir', '')
        # Check if the output_dir exists, and if not, create it
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        # Get the pose model from the config
        pose_model = get_Model(config_dict)

        # If model_path is empty, handle the error or set a default value
        if not pose_model:
            raise ValueError(f"Model path not found for pose_model: {pose_model}")

        # Combine the model path with the setup directory

        # Now model_path holds the full path to the model file based on the pose_model
        print(f"Model Path: {pose_model}")

        trc_file = Path(load_trc(config_dict))
        scaling_path = get_Scale_Setup(config_dict)

        #Processing TRC I guess
        Q_coords, _, _, header = read_trc(trc_file)

        Q_diff = Q_coords.diff(axis=0).sum(axis=1)
        Q_diff = Q_diff[Q_diff != 0]  # did not work for V03 -> trimmed
        min_speed_indices = Q_diff.abs().nsmallest(int(len(Q_diff) * 0.1)).index
        Q_coords_scaling = Q_coords.iloc[min_speed_indices]
        Q_coords_scaling = Q_coords_scaling.reset_index(drop=True)

        #processed TRC
        trc_scaling_path = trc_file.parent / (trc_file.stem + '_scaling.trc')
        make_trc_with_Q(Q_coords_scaling, header, str(trc_scaling_path))

        scaling_file_path = str(trc_file.parent / (trc_file.stem + '_' + Path(scaling_path).name))
        scaled_model_path = os.path.join(output_dir, trc_file.stem + '_scaled.osim')
        scaling_tree = etree.parse(str(scaling_path))
        scaling_root = scaling_tree.getroot()

        scaling_root[0].find('mass').text = str(athlete_weight)
        scaling_root[0].find('height').text = str(athlete_height)
        scaling_root[0].find('GenericModelMaker').find('model_file').text = str(pose_model)
        scaling_root[0].find('ModelScaler').find('marker_file').text = trc_scaling_path.name
        scaling_root[0].find('ModelScaler').find('time_range').text = '0 ' + str(Q_coords_scaling['t'].iloc[-1])
        scaling_root[0].find('ModelScaler').find('output_model_file').text = scaled_model_path
        scaling_root[0].find('MarkerPlacer').find('marker_file').text = trc_scaling_path.name
        scaling_root[0].find('MarkerPlacer').find('time_range').text = '0 ' + str(Q_coords_scaling['t'].iloc[-1])
        scaling_root[0].find('MarkerPlacer').find('output_model_file').text = scaled_model_path
        scaling_tree.write(scaling_file_path)


        opensim.ScaleTool(scaling_file_path).run()

        return scaled_model_path
    except Exception as e:
        logging.error(f"Error during scaling: {e}")
        raise


def perform_IK(config_dict):
    """
    Perform Inverse Kinematics (IK) using OpenSim, with configuration options.

    :param config_dict: Dictionary containing configuration details.
    :param trc_file: Path to the TRC file used for IK.
    :param scaled_model_path: Path to the scaled model.
    :param output_dir: Directory to save the IK results.
    :return: Path to the IK results file.
    """
    try:
        # Get required path based on the configuration
        ik_setup_path = get_IK_Setup(config_dict)
        opensim_config = config_dict.get('opensim', {})
        trc_file = load_trc(config_dict)
        trc_file_path = Path(trc_file)
        output_dir = config_dict.get('opensim', {}).get('output_dir', '')
        # Check if the output_dir exists, and if not, create it
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")

        #Get the scaled model
        scaled_model_path = os.path.join(config_dict.get('opensim', {}).get('output_dir', output_dir),
                                         trc_file_path.stem + '_scaled.osim')
        if not scaled_model_path:
            logging.error(f"Scaled model path not found. Expected path: {scaled_model_path}")

        # Load the TRC file to determine the time range
        Q_coords, frames_col, time_col, header = read_trc(trc_file_path)
        ik_time_range = opensim_config.get('IK_timeRange', [])

        # If IK_timeRange is not specified, use the longest possible range
        if not ik_time_range:
            start_time = time_col.iloc[0]
            end_time = time_col.iloc[-1]
        else:
            start_time, end_time = ik_time_range[0], ik_time_range[1]

        # Parse the IK setup XML and adjust the time range
        ik_file_path = str(trc_file_path.parent / (trc_file_path.stem + '_' + Path(ik_setup_path).name))
        ik_tree = etree.parse(ik_setup_path)
        ik_root = ik_tree.getroot()
        ik_root.find('.//model_file').text = str(scaled_model_path)
        ik_root.find('.//time_range').text = f'{start_time} {end_time}'
        ik_root.find('.//output_motion_file').text = os.path.join(output_dir, trc_file_path.stem + '.mot')
        ik_root.find('.//marker_file').text = str(trc_file_path)
        ik_tree.write(ik_file_path)

        opensim.InverseKinematicsTool(ik_file_path).run()

    except:
        print(f'Error with {trc_file.stem}')

    return ik_file_path
