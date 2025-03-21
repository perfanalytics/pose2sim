#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
###########################################################################
## AUGMENT MARKER DATA                                                   ##
###########################################################################

Augment trc 3D coordinates. 

Estimate the position of 43 additional markers.
    
INPUTS: 
- a trc file
- filtering parameters in Config.toml

OUTPUT: 
- a filtered trc file
'''


## INIT
import os
import numpy as np
import copy
import tensorflow as tf
import logging

from Pose2Sim.MarkerAugmenter import utilsDataman
from Pose2Sim.MarkerAugmenter.utils import TRC2numpy
from Pose2Sim.common import convert_to_c3d, read_trc, compute_height


## AUTHORSHIP INFORMATION
__author__ = "Antoine Falisse, adapted by HunMin Kim and David Pagnon"
__copyright__ = "Copyright 2022, OpenCap"
__credits__ = ["Antoine Falisse", "HunMin Kim", "David Pagnon"]
__license__ = "Apache-2.0 License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
# subject_height must be in meters
def check_midhip_data(trc_file):
    try:
        # Find MidHip data
        midhip_data = trc_file.marker("Hip")
        if midhip_data is None or len(midhip_data) == 0:
            raise ValueError("MidHip data is empty")
    except (KeyError, ValueError):
        # If MidHip data is not found, calculate it from RHip and LHip
        rhip_data = trc_file.marker("RHip")
        lhip_data = trc_file.marker("LHip")
        midhip_data = (rhip_data + lhip_data) / 2
        trc_file.add_marker('Hip', *midhip_data.T)

    return trc_file


def check_neck_data(trc_file):
    try:
        # Find Neck data
        neck_data = trc_file.marker("Neck")
        if neck_data is None or len(neck_data) == 0:
            raise ValueError("Neck data is empty")
    except (KeyError, ValueError):
        # If Neck data is not found, calculate it from RShoulder and LShoulder
        rshoulder_data = trc_file.marker("RShoulder")
        lshoulder_data = trc_file.marker("LShoulder")
        neck_data = (rshoulder_data + lshoulder_data) / 2
        trc_file.add_marker('Neck', *neck_data.T)

    return trc_file


def augment_markers_all(config):
    subject_height, subject_mass, trc_files, fastest_frames_to_remove_percent, close_to_zero_speed, large_hip_knee_angles, trimmed_extrema_percent, augmenter_model, augmenterDir, augmenterModelName, pathInputTRCFile, pathOutputTRCFile, make_c3d, offset = config.get_augment_markers_params()
    # Calculate subject heights
    if not subject_height or subject_height == 0:
        subject_height = [config.default_height] * len(trc_files)
        logging.warning(f"No subject height found in Config.toml. Using default height of {config.default_height}m.")
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
                    logging.warning(f"Could not compute height from {os.path.basename(trc_file)}. Using default height of {config.default_height}m.")
                    logging.warning(f"The person may be static, or crouched, or incorrectly detected. You may edit fastest_frames_to_remove_percent, close_to_zero_speed_m, large_hip_knee_angles, trimmed_extrema_percent, default_height in your Config.toml file.")
                    height = config.default_height
            except Exception as e:
                logging.warning(f"Could not compute height from {os.path.basename(trc_file)}. Using default height of {config.default_height}m.")
                logging.warning(f"The person may be static, or crouched, or incorrectly detected. You may edit fastest_frames_to_remove_percent, close_to_zero_speed_m, large_hip_knee_angles, trimmed_extrema_percent, default_height in your Config.toml file.")
                height = config.default_height
            subject_height.append(height)
    elif not type(subject_height) == list: # int or float
        subject_height = [subject_height]
    if len(subject_height) < len(trc_files):
        logging.warning("Number of subject heights does not match number of TRC files. Missing heights are set to {default_height}m.")
        subject_height += [config.default_height] * (len(trc_files) - len(subject_height))

    # Get subject masses
    if not subject_mass or subject_mass == 0:
        subject_mass = [70] * len(trc_files)
        logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")
    if not type(subject_mass) == list:
        subject_mass = [subject_mass]
    if len(subject_mass) < len(trc_files):
        logging.warning("Number of subject masses does not match number of TRC files. Missing masses are set to 70kg.")
        subject_mass += [70] * (len(trc_files) - len(subject_mass))

    for p, trc_file_input in enumerate(trc_files):
        trc_file_input = os.path.splitext(trc_file_input)[0] + '_LSTM.trc'
    
        # This is by default - might need to be adjusted in the future.
        featureHeight = True
        featureWeight = True
        
        # Augmenter types
        if augmenter_model == 'v0.3':
            # Lower body           
            augmenterModelType_lower = '{}_lower'.format(augmenter_model)
            from Pose2Sim.MarkerAugmenter.utils import getOpenPoseMarkers_lowerExtremity2
            feature_markers_lower, response_markers_lower = getOpenPoseMarkers_lowerExtremity2()
            # Upper body
            augmenterModelType_upper = '{}_upper'.format(augmenter_model)
            from Pose2Sim.MarkerAugmenter.utils import getMarkers_upperExtremity_noPelvis2
            feature_markers_upper, response_markers_upper = getMarkers_upperExtremity_noPelvis2()        
            augmenterModelType_all = [augmenterModelType_lower, augmenterModelType_upper]
            feature_markers_all = [feature_markers_lower, feature_markers_upper]
            response_markers_all = [response_markers_lower, response_markers_upper]
        else:
            raise ValueError('Augmenter models other than 0.3 are not supported.')
        logging.info('Using Stanford augmenter model: {}'.format(augmenter_model))
        
        # %% Process data.
        # Import TRC file
        try:
            trc_file = utilsDataman.TRCFile(trc_file_input)
        except:
            raise ValueError('Cannot read TRC file. You may need to enable interpolation in Config.toml while triangulating.')
        
        # add neck and midhip data if not in file
        trc_file = check_midhip_data(trc_file)
        trc_file = check_neck_data(trc_file)
        trc_file.write(trc_file_input)
        
        # Verify that all feature markers are present in the TRC file.
        feature_markers_joined = set(feature_markers_all[0]+feature_markers_all[1])
        trc_markers = set(trc_file.marker_names)
        missing_markers = list(feature_markers_joined - trc_markers)
        if len(missing_markers) > 0:
            raise ValueError(f'Marker augmentation requires {missing_markers} markers and they are not present in the TRC file.')

        # Loop over augmenter types to handle separate augmenters for lower and
        # upper bodies.
        outputs_all = {}
        n_response_markers_all = 0
        for idx_augm, augmenterModelType in enumerate(augmenterModelType_all):
            outputs_all[idx_augm] = {}
            feature_markers = feature_markers_all[idx_augm]
            response_markers = response_markers_all[idx_augm]
            
            augmenterModelDir = os.path.join(augmenterDir, augmenterModelName, 
                                             augmenterModelType)
            
            # %% Pre-process inputs.
            # Step 1: import .trc file with OpenPose marker trajectories.  
            trc_data = TRC2numpy(pathInputTRCFile, feature_markers)
            trc_data_data = trc_data[:,1:]

            # Step 2: Normalize with reference marker position.
            referenceMarker_data = trc_file.marker("Hip")  # instead of trc_file.marker(referenceMarker) # change by HunMin
            norm_trc_data_data = np.zeros((trc_data_data.shape[0],
                                        trc_data_data.shape[1]))
            for i in range(0,trc_data_data.shape[1],3):
                norm_trc_data_data[:,i:i+3] = (trc_data_data[:,i:i+3] - 
                                            referenceMarker_data)
                
                
            # Step 3: Normalize with subject's height.
            norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
            norm2_trc_data_data = norm2_trc_data_data / subject_height[p]
            
            # Step 4: Add remaining features.
            inputs = copy.deepcopy(norm2_trc_data_data)
            if featureHeight:    
                inputs = np.concatenate(
                    (inputs, subject_height[p]*np.ones((inputs.shape[0],1))), axis=1)
            if featureWeight:    
                inputs = np.concatenate(
                    (inputs, subject_mass[p]*np.ones((inputs.shape[0],1))), axis=1)
                
            # Step 5: Pre-process data
            pathMean = os.path.join(augmenterModelDir, "mean.npy")
            pathSTD = os.path.join(augmenterModelDir, "std.npy")
            if os.path.isfile(pathMean):
                trainFeatures_mean = np.load(pathMean, allow_pickle=True)
                inputs -= trainFeatures_mean
            if os.path.isfile(pathSTD):
                trainFeatures_std = np.load(pathSTD, allow_pickle=True)
                inputs /= trainFeatures_std 
                
            # Step 6: Reshape inputs if necessary (eg, LSTM)
            if augmenterModelName == "LSTM":
                inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1]))
                
            # %% Load model and weights, and predict outputs.
            json_file = open(os.path.join(augmenterModelDir, "model.json"), 'r')
            pretrainedModel_json = json_file.read()
            json_file.close()
            model = tf.keras.models.model_from_json(pretrainedModel_json, custom_objects={
                                    'Sequential': tf.keras.models.Sequential,
                                    'Dense': tf.keras.layers.Dense
                                    })
            model.load_weights(os.path.join(augmenterModelDir, "weights.h5"))  
            outputs = model.predict(inputs)
            tf.keras.backend.clear_session()

            # %% Post-process outputs.
            # Step 1: Reshape if necessary (eg, LSTM)
            if augmenterModelName == "LSTM":
                outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
                
            # Step 2: Un-normalize with subject's height.
            unnorm_outputs = outputs * subject_height[p]
            
            # Step 2: Un-normalize with reference marker position.
            unnorm2_outputs = np.zeros((unnorm_outputs.shape[0],
                                        unnorm_outputs.shape[1]))
            for i in range(0,unnorm_outputs.shape[1],3):
                unnorm2_outputs[:,i:i+3] = (unnorm_outputs[:,i:i+3] + 
                                            referenceMarker_data)
                
            # %% Add markers to .trc file.
            for c, marker in enumerate(response_markers):
                x = unnorm2_outputs[:,c*3]
                y = unnorm2_outputs[:,c*3+1]
                z = unnorm2_outputs[:,c*3+2]
                trc_file.add_marker(marker, x, y, z)
                
            # %% Gather data for computing minimum y-position.
            outputs_all[idx_augm]['response_markers'] = response_markers   
            outputs_all[idx_augm]['response_data'] = unnorm2_outputs
            n_response_markers_all += len(response_markers)
            
        # %% Extract minimum y-position across response markers. This is used
        # to align feet and floor when visualizing.
        responses_all_conc = np.zeros((unnorm2_outputs.shape[0],
                                       n_response_markers_all*3))
        idx_acc_res = 0
        for idx_augm in outputs_all:
            idx_acc_res_end = (idx_acc_res + 
                               (len(outputs_all[idx_augm]['response_markers']))*3)
            responses_all_conc[:,idx_acc_res:idx_acc_res_end] = (
                outputs_all[idx_augm]['response_data'])
            idx_acc_res = idx_acc_res_end
        # Minimum y-position across response markers.
        min_y_pos = np.min(responses_all_conc[:,1::3])
            
        # %% If offset
        if offset:
            trc_file.offset('y', -(min_y_pos-0.01))
            
        # %% Return augmented .trc file   
        trc_file.write(pathOutputTRCFile)

        logging.info(f'Augmented marker coordinates are stored at {pathOutputTRCFile}.')

        # Save c3d
        if make_c3d:
            convert_to_c3d(pathOutputTRCFile)
            logging.info(f'Augmented trc files have been converted to c3d.')
            
    return min_y_pos
