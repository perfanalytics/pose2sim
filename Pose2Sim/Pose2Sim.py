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
    from Pose2Sim import Pose2Sim
    Pose2Sim.track2D()
    Pose2Sim.triangulate3D()
    Pose2Sim.filter3D()
    
'''


## INIT
import toml
import os
import time
import logging, logging.handlers

if not os.path.exists('User'): os.mkdir('User')
with open(os.path.join('User', 'logs.txt'), 'a+') as log_f: pass
logging.basicConfig(format='%(message)s', level=logging.INFO, 
    handlers = [logging.handlers.TimedRotatingFileHandler(os.path.join('User', 'logs.txt'), when='D', interval=7), logging.StreamHandler()]) 


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
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

    return project_dir, seq_name, frames


def calibrateCams(config=os.path.join('User', 'Config.toml')):
    '''
    Cameras calibration from checkerboards or from qualisys files.
    '''

    from Pose2Sim.calibrate_cams import calibrate_cams_all
    
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
    
    
    
def track2D(config=os.path.join('User', 'Config.toml')):
    '''
    Tracking of the person of interest in case of multiple persons detection.
    Needs a calibration file.
    '''
    
    from Pose2Sim.track_2d import track_2d_all
    
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
    
    
def triangulate3D(config=os.path.join('User', 'Config.toml')):
    '''
    Robust triangulation of 2D points coordinates.
    '''

    from Pose2Sim.triangulate_3d import triangulate_all

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
    
    
def filter3D(config=os.path.join('User', 'Config.toml')):
    '''
    Filter trc 3D coordinates.
    '''

    from Pose2Sim.filter_3d import filter_all

    config_dict = read_config_file(config)
    project_dir, seq_name, frames = base_params(config_dict)
    
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info(f"Filtering 3D coordinates for {seq_name}, for {frames}.")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nProject directory: {project_dir}")
    
    filter_all(config_dict)
