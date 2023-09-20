#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## POSE2SIM                                                              ##
    ###########################################################################
    
    This repository offers a way to perform markerless kinematics, and gives an 
    example workflow from an Openpose input to an OpenSim result.

    It offers tools for:
    - 2D pose estimation,
    - Cameras calibration,
    - Tracking the person of interest,
    - Robust triangulation,
    - Filtration.

    It has been tested on Windows 10 but should work similarly on Linux.
    Please subscribe to this issue if you wish to be notified of the code release. 
    See https://github.com/perfanalytics/pose2sim
    
    Installation: 
    # Open Anaconda prompt. Type:
    # - conda create -n Pose2Sim python=3.7 tensorflow-gpu=1.13.1
    # - conda activate Pose2Sim
    # - conda install Pose2Sim

    Usage: 
    # First run Pose estimation and organize your directories (see Readme.md)
    from Pose2Sim import Pose2Sim
    Pose2Sim.calibration()
    Pose2Sim.personAssociation()
    Pose2Sim.triangulation()
    Pose2Sim.filtering()
    # Then run OpenSim (see Readme.md)
    
'''


## INIT
import toml
import os
import time
import logging, logging.handlers


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def read_config_file(config):
    '''
    Read configation file.
    '''

    config_dict = toml.load(config)
    return config_dict


def base_params(config_dict):
    '''
    Retrieve sequence name and frames to be analyzed.
    '''

    project_dir = config_dict.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    frame_range = config_dict.get('project').get('frame_range')
    seq_name = os.path.basename(project_dir)
    frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

    if not os.path.exists('User'): os.mkdir('User')
    with open(os.path.join(project_dir, 'User', 'logs.txt'), 'a+') as log_f: pass
    logging.basicConfig(format='%(message)s', level=logging.INFO, 
        handlers = [logging.handlers.TimedRotatingFileHandler(os.path.join(project_dir, 'User', 'logs.txt'), when='D', interval=7), logging.StreamHandler()])

    return project_dir, seq_name, frames


def poseEstimation(config=os.path.join('User', 'Config.toml')):
    '''
    Estimate pose using BlazePose, OpenPose, AlphaPose, or DeepLabCut.
    
    config can either be a path or a dictionary (for batch processing)
    '''
    
    raise NotImplementedError('This has not been integrated yet. \nPlease read README.md for further explanation')
    
    # TODO
    from Pose2Sim.poseEstimation import pose_estimation_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Pose estimation")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    pose_estimation_all(config_dict)
    
    end = time.time()
    logging.info(f'Pose estimation took {end-start:.2f} s.')


def calibration(config=os.path.join('User', 'Config.toml')):
    '''
    Cameras calibration from checkerboards or from qualisys files.
    
    config can either be a path or a dictionary (for batch processing)
    '''

    from Pose2Sim.calibration import calibrate_cams_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Camera calibration")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    calibrate_cams_all(config_dict)
    
    end = time.time()
    logging.info(f'Calibration took {end-start:.2f} s.')
    

def synchronization(config=os.path.join('User', 'Config.toml')):
    '''
    Synchronize cameras if needed.
    
    config can either be a path or a dictionary (for batch processing)
    '''   

    raise NotImplementedError('This has not been integrated yet. \nPlease read README.md for further explanation')
    
    #TODO
    from Pose2Sim.synchronization import synchronize_cams_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Camera synchronization")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    synchronize_cams_all(config_dict)
    
    end = time.time()
    logging.info(f'Synchronization took {end-start:.2f} s.')    
    
    
def personAssociation(config=os.path.join('User', 'Config.toml')):
    '''
    Tracking of the person of interest in case of multiple persons detection.
    Needs a calibration file.
    
    config can either be a path or a dictionary (for batch processing)
    '''
    
    from Pose2Sim.personAssociation import track_2d_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Tracking of the person of interest for {seq_name}, for {frames}.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    track_2d_all(config_dict)
    
    end = time.time()
    logging.info(f'Tracking took {end-start:.2f} s.')
    
    
def triangulation(config=os.path.join('User', 'Config.toml')):
    '''
    Robust triangulation of 2D points coordinates.
    
    config can either be a path or a dictionary (for batch processing)
    '''

    from Pose2Sim.triangulation import triangulate_all

    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)

    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Triangulation of 2D points for {seq_name}, for {frames}.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    triangulate_all(config_dict)
    
    end = time.time()
    logging.info(f'Triangulation took {end-start:.2f} s.')
    
    
def filtering(config=os.path.join('User', 'Config.toml')):
    '''
    Filter trc 3D coordinates.
    
    config can either be a path or a dictionary (for batch processing)
    '''

    from Pose2Sim.filtering import filter_all

    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Filtering 3D coordinates for {seq_name}, for {frames}.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    
    filter_all(config_dict)


def scalingModel(config=os.path.join('User', 'Config.toml')):
    '''
    Uses OpenSim to scale a model based on a static 3D pose.
    
    config can either be a path or a dictionary (for batch processing)
    '''
    
    raise NotImplementedError('This has not been integrated yet. \nPlease read README.md for further explanation')
    
    # TODO
    from Pose2Sim.scalingModel import scale_model_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Scaling model")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    scale_model_all(config_dict)
    
    end = time.time()
    logging.info(f'Model scaling took {end-start:.2f} s.')
    
    
def inverseKinematics(config=os.path.join('User', 'Config.toml')):
    '''
    Uses OpenSim to perform inverse kinematics.
    
    config can either be a path or a dictionary (for batch processing)
    '''
    
    raise NotImplementedError('This has not been integrated yet. \nPlease read README.md for further explanation')
    
    # TODO
    from Pose2Sim.inverseKinematics import inverse_kinematics_all
    
    if type(config)==dict:
        config_dict = config
    else:
        config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Inverse kinematics")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    start = time.time()
    
    inverse_kinematics_all(config_dict)
    
    end = time.time()
    logging.info(f'Inverse kinematics took {end-start:.2f} s.')