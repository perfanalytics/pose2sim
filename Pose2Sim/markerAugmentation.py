#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
###########################################################################
## AUGMENT MARKER DATA                                                   ##
###########################################################################

Augment trc 3D coordinates. 

Estimate the position of 43 additional markers.
Uses the LSTM model trained on Stanford data, converted to ONNX.
    
INPUTS: 
- a trc file
- filtering parameters in Config.toml

OUTPUT: 
- a filtered trc file
'''


## INIT
import os
import copy
import numpy as np
import pandas as pd
import onnxruntime as ort
import glob
import logging

from Pose2Sim.common import convert_to_c3d, natural_sort_key, read_trc, compute_height


## AUTHORSHIP INFORMATION
__author__ = "Antoine Falisse, adapted by HunMin Kim and David Pagnon"
__copyright__ = "Copyright 2022, OpenCap"
__credits__ = ["Antoine Falisse", "HunMin Kim", "David Pagnon"]
__license__ = "Apache-2.0 License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def add_neck_hip_data(trc_data, markers, header):
    '''
    Add neck and midhip data to trc_data if not present.
    Also update header and markers.
    '''

    midpoints = {
        'Neck': ['RShoulder', 'LShoulder'],
        'Hip': ['RHip', 'LHip']}

    for mk_name, r_l_markers in midpoints.items():
        if mk_name not in markers:
            # Add marker name
            markers.append(mk_name)

            # Update header
            header[2] = '\t'.join(part if i != 3 else str(len(markers)) for i, part in enumerate(header[2].split('\t')))
            header[3] = header[3].replace('\t\t\t\n', f'\t\t\t{mk_name}\t\t\t\n')
            header[4] = ['\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(markers))]) + '\t\n'][0]

            # update trc_data
            r_l_data = [trc_data[marker] for marker in r_l_markers]
            mid_data = pd.DataFrame(sum([data.values for data in r_l_data])/2, columns=[mk_name]*3)
            trc_data = pd.concat([trc_data, mid_data], axis=1)

    return trc_data, markers, header


def getOpenPoseMarkers_lowerExtremity2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
        "RBigToe", "LBigToe"]

    response_markers = [
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

    return feature_markers, response_markers


def getMarkers_upperExtremity_noPelvis2():
    feature_markers = [
        "Neck", "RShoulder", "LShoulder", "RElbow", "LElbow", "RWrist",
        "LWrist"]

    response_markers = ["r_lelbow_study", "r_melbow_study", "r_lwrist_study",
                        "r_mwrist_study", "L_lelbow_study", "L_melbow_study",
                        "L_lwrist_study", "L_mwrist_study"]

    return feature_markers, response_markers


def augment_markers_all(config_dict):
    # get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    pose_3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    feet_on_floor = config_dict.get('markerAugmentation').get('feet_on_floor')
    make_c3d = config_dict.get('markerAugmentation').get('make_c3d')
    frame_range = config_dict.get('project').get('frame_range')
    subject_height = config_dict.get('project').get('participant_height')
    subject_mass = config_dict.get('project').get('participant_mass')
    
    fastest_frames_to_remove_percent = config_dict.get('kinematics').get('fastest_frames_to_remove_percent')
    close_to_zero_speed = config_dict.get('kinematics').get('close_to_zero_speed_m')
    large_hip_knee_angles = config_dict.get('kinematics').get('large_hip_knee_angles')
    trimmed_extrema_percent = config_dict.get('kinematics').get('trimmed_extrema_percent')
    default_height = config_dict.get('kinematics').get('default_height')

    augmenterDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MarkerAugmenter')
    augmenterModelName = 'LSTM'
    augmenter_model = 'v0.3'

    # Apply all trc files
    all_trc_files = [f for f in glob.glob(os.path.join(pose_3d_dir, '*.trc')) if augmenterModelName not in f]
    trc_no_filtering = [f for f in glob.glob(os.path.join(pose_3d_dir, '*.trc')) if
                        augmenterModelName not in f and 'filt' not in f]
    trc_filtering = [f for f in glob.glob(os.path.join(pose_3d_dir, '*.trc')) if augmenterModelName not in f and 'filt' in f]

    if len(all_trc_files) == 0:
        raise ValueError('No trc files found.')
    if len(trc_filtering) > 0:
        trc_files = trc_filtering
    else:
        trc_files = trc_no_filtering
    sorted(trc_files, key=natural_sort_key)

    # Calculate subject heights
    if subject_height is None or subject_height == 0:
        subject_height = [default_height] * len(trc_files)
        logging.warning(f"No subject height found in Config.toml. Using default height of {default_height}m.")
    elif isinstance(subject_height, str) and subject_height.lower() == 'auto':
        subject_height = []
        for trc_file in trc_files:
            try:
                # Import TRC file
                trc_data, frames_col, time_col, markers, header = read_trc(trc_file)

                # frame range selection
                f_range = [[frames_col.iloc[0], frames_col.iloc[-1]]+1 if frame_range in ('all', 'auto', []) else frame_range][0]
                f_index = [frames_col[frames_col==f_range[0]].index[0], frames_col[frames_col==f_range[1]-1].index[0]+1]
                trc_data = trc_data.iloc[f_index[0]:f_index[1]].reset_index(drop=True)

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
        logging.warning("Number of subject heights does not match number of TRC files. Missing heights are set to {default_height}m.")
        subject_height += [default_height] * (len(trc_files) - len(subject_height))

    # Get subject masses
    if subject_mass is None or subject_mass == 0:
        subject_mass = [70] * len(trc_files)
        logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")
    elif not type(subject_mass) == list:
        subject_mass = [subject_mass]
    if len(subject_mass) < len(trc_files):
        logging.warning("Number of subject masses does not match number of TRC files. Missing masses are set to 70kg.")
        subject_mass += [70] * (len(trc_files) - len(subject_mass))

    for p in range(len(subject_mass)):
        trc_file = trc_files[p]
        trc_file_out = os.path.splitext(trc_file)[0] + f'_{augmenterModelName}.trc'
        
        # Lower body           
        augmenterModelType_lower = '{}_lower'.format(augmenter_model)
        feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
        # Upper body
        augmenterModelType_upper = '{}_upper'.format(augmenter_model)
        feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
        augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
        feature_markers_all = [feature_markers_lower, feature_markers_upper]
        response_markers_all = [response_markers_lower, response_markers_upper]
        logging.info(f'Using Stanford {augmenterModelName} {augmenter_model} augmenter model. Feet are {"not " if feet_on_floor else ""}vertically offset to be at floor level.\n')
        
        # %% Process data.
        # Import TRC file
        trc_data, frames_col, time_col, markers, header = read_trc(trc_file)

        # add neck and midhip data if not in file
        trc_data, markers, header = add_neck_hip_data(trc_data, markers, header)

        # frame range selection
        f_range = [[frames_col.iloc[0], frames_col.iloc[-1]+1] if frame_range in ('all', 'auto', []) else frame_range][0]
        frame_nb = f_range[1] - f_range[0]
        f_index = [frames_col[frames_col==f_range[0]].index[0], frames_col[frames_col==f_range[1]-1].index[0]+1]
        trc_data = trc_data.iloc[f_index[0]:f_index[1]].reset_index(drop=True)
        frames_col = frames_col.iloc[f_index[0]:f_index[1]].reset_index(drop=True)
        time_col = time_col.iloc[f_index[0]:f_index[1]].reset_index(drop=True)

        trc_path_out = trc_file.replace('.trc', '_LSTM.trc')
        trc_file_out = os.path.basename(trc_path_out)
        header[0] = header[0].replace(os.path.basename(trc_file), trc_file_out)
        header[2] = '\t'.join(part if i != 2 else str(frame_nb) for i, part in enumerate(header[2].split('\t')))
        header[2] = '\t'.join(part if i != 7 else str(frame_nb)+'\n' for i, part in enumerate(header[2].split('\t')))

        # Verify that all feature markers are present in the TRC file.
        feature_markers_joined = set(feature_markers_all[0]+feature_markers_all[1])
        missing_markers = list(feature_markers_joined - set(markers))
        if len(missing_markers) > 0:
            raise ValueError(f'Marker augmentation requires {missing_markers} markers and they are not present in the TRC file.')

        # Loop over augmenter types to handle separate augmenters for lower and
        # upper bodies.
        outputs_all = {}
        for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
            outputs_all[idx_augm] = {}
            feature_markers = feature_markers_all[idx_augm]
            response_markers = response_markers_all[idx_augm]
            
            augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                             augmenterModelType)
            
            # %% Pre-process inputs.
            # Step 1: import .trc file with OpenPose marker trajectories.  
            trc_data_feature = trc_data[feature_markers]

            # Step 2: Normalize with reference marker position.
            norm_trc_data_feature = trc_data_feature.values - np.tile(trc_data['Hip'], (1,trc_data_feature.shape[1]//3))

            # Step 3: Normalize with subject's height.
            norm2_trc_data_feature = copy.deepcopy(norm_trc_data_feature)
            norm2_trc_data_feature = norm2_trc_data_feature / subject_height[p]
            
            # Step 4: Add remaining features.
            inputs = copy.deepcopy(norm2_trc_data_feature)
            inputs = np.concatenate(
                    (inputs, subject_height[p]*np.ones((inputs.shape[0],1))), axis=1)
            inputs = np.concatenate(
                    (inputs, subject_mass[p]*np.ones((inputs.shape[0],1))), axis=1)
                
            # Step 5: Pre-process data
            pathMean = os.path.join(augmenterModelDir, "mean.npy")
            trainFeatures_mean = np.load(pathMean, allow_pickle=True)
 
            pathSTD = os.path.join(augmenterModelDir, "std.npy")
            trainFeatures_std = np.load(pathSTD, allow_pickle=True)

            inputs = (inputs - trainFeatures_mean) / trainFeatures_std
                
            # Step 6: Reshape inputs for LSTM model.
            inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
                
            # %% Load model and weights, and predict outputs.
            onnx_path = os.path.join(augmenterModelDir, "model.onnx")
            session = ort.InferenceSession(onnx_path)
            outputs = session.run(['output_0'], {'inputs': inputs.astype(np.float32)})[0]

            # %% Post-process outputs.
            # Step 1: Reshape from LSTM output
            outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
                
            # Step 2: Un-normalize with subject's height.
            unnorm_outputs = outputs * subject_height[p]
            
            # Step 2: Un-normalize with reference marker position.
            unnorm2_outputs = unnorm_outputs + np.tile(trc_data['Hip'], (1,unnorm_outputs.shape[1]//3))


            # %% Add response markers to trc_data and update markers and header.
            trc_data_response = pd.DataFrame(unnorm2_outputs, columns=[m for m in response_markers for _ in range(3)])
            trc_data = pd.concat([trc_data, trc_data_response], axis=1)

            markers += response_markers
            
            header[2] = '\t'.join(part if i != 3 else str(len(markers)) for i, part in enumerate(header[2].split('\t')))
            response_markers_str = '\t\t\t'.join(response_markers)
            header[3] = header[3].replace('\t\t\t\n', f'\t\t\t{response_markers_str}\t\t\t\n')
            header[4] = ['\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(markers))]) + '\t\n'][0]
            
        # %% Extract minimum y-position across response markers. This is used
        # to align feet and floor when visualizing.
        response_markers_conc = [m for resp in response_markers_all for m in resp]
        min_y_pos = trc_data[response_markers_conc].iloc[:,1::3].min().min()
            
        # %% If offset
        if feet_on_floor:
            trc_data.iloc[:,1::3] = trc_data.iloc[:,1::3] - (min_y_pos-0.01)
            
        # %% Return augmented .trc file   
        with open(trc_path_out, 'w') as trc_o:
            [trc_o.write(line) for line in header]
            trc_data.insert(0, 'Frame#', frames_col)
            trc_data.insert(1, 'Time', time_col)
            trc_data.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')
        logging.info(f'Augmented marker coordinates are stored at {trc_path_out}.')

        # Save c3d
        if make_c3d:
            convert_to_c3d(trc_path_out)
            logging.info(f'Augmented trc files have been converted to c3d.')
            
    return min_y_pos
