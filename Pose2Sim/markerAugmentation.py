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
from Pose2Sim.MarkerAugmenter import utilsDataman
import copy
import tensorflow as tf
from Pose2Sim.MarkerAugmenter.utils import TRC2numpy
import json
import glob
import logging


## AUTHORSHIP INFORMATION
__author__ = "Antoine Falisse, adapted by HunMin Kim"
__copyright__ = "Copyright 2022, OpenCap"
__credits__ = ["Antoine Falisse", "HunMin Kim"]
__license__ = "Apache-2.0 License"
__version__ = '0.5'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
# subject_height must be in meters
def get_midhip_data(trc_file):
    try:
        # Find MidHip data
        midhip_data = trc_file.marker("CHip")
        if midhip_data is None or len(midhip_data) == 0:
            raise ValueError("MidHip data is empty")
    except (KeyError, ValueError):
        # If MidHip data is not found, calculate it from RHip and LHip
        rhip_data = trc_file.marker("RHip")
        lhip_data = trc_file.marker("LHip")
        midhip_data = (rhip_data + lhip_data) / 2

    return midhip_data


def augmentTRC(config_dict):

    # get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    session_dir = os.path.realpath(os.path.join(project_dir, '..', '..'))
    pathInputTRCFile = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    pathOutputTRCFile = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    pose_model = config_dict.get('pose').get('pose_model')
    subject_height = config_dict.get('markerAugmentation').get('participant_height')
    if subject_height is None or subject_height == 0:
        raise ValueError("Subject height is not set or invalid in the config file.")
    subject_mass = config_dict.get('markerAugmentation').get('participant_mass')
    augmenterDir = os.path.join(session_dir, '..', '..', 'MarkerAugmenter')
    augmenterModelName = 'LSTM'
    augmenter_model = 'v0.3'
    offset = True

    if pose_model not in ['BODY_25', 'BODY_25B']:
        raise ValueError('Marker augmentation is only supported with OpenPose BODY_25 and BODY_25B models.')

    # Apply all trc files
    trc_files = [f for f in glob.glob(os.path.join(pathInputTRCFile, '*.trc')) if '_LSTM' not in f]
    for pathInputTRCFile in trc_files:
        pathOutputTRCFile = os.path.splitext(pathInputTRCFile)[0] + '_LSTM.trc'
    
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
    logging.info('Using augmenter model: {}'.format(augmenter_model))
    
    # %% Process data.
    # Import TRC file
    trc_file = utilsDataman.TRCFile(pathInputTRCFile)
    
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

        # Calculate the midHip marker as the average of RHip and LHip
        midhip_data = get_midhip_data(trc_file)

        trc_data_data = trc_data[:,1:]

        # Step 2: Normalize with reference marker position.
        with open(os.path.join(augmenterModelDir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        referenceMarker_data = midhip_data  # instead of trc_file.marker(referenceMarker) # change by HunMin
        norm_trc_data_data = np.zeros((trc_data_data.shape[0],
                                    trc_data_data.shape[1]))
        for i in range(0,trc_data_data.shape[1],3):
            norm_trc_data_data[:,i:i+3] = (trc_data_data[:,i:i+3] - 
                                        referenceMarker_data)
            
            
        # Step 3: Normalize with subject's height.
        norm2_trc_data_data = copy.deepcopy(norm_trc_data_data)
        norm2_trc_data_data = norm2_trc_data_data / subject_height
        
        # Step 4: Add remaining features.
        inputs = copy.deepcopy(norm2_trc_data_data)
        if featureHeight:    
            inputs = np.concatenate(
                (inputs, subject_height*np.ones((inputs.shape[0],1))), axis=1)
        if featureWeight:    
            inputs = np.concatenate(
                (inputs, subject_mass*np.ones((inputs.shape[0],1))), axis=1)
            
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
        model = tf.keras.models.model_from_json(pretrainedModel_json)
        model.load_weights(os.path.join(augmenterModelDir, "weights.h5"))  
        outputs = model.predict(inputs)
        
        # %% Post-process outputs.
        # Step 1: Reshape if necessary (eg, LSTM)
        if augmenterModelName == "LSTM":
            outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
            
        # Step 2: Un-normalize with subject's height.
        unnorm_outputs = outputs * subject_height
        
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
    
    return min_y_pos

