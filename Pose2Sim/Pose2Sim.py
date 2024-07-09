#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE2SIM                                                              ##
###########################################################################

This repository offers a way to perform markerless kinematics, and gives an 
example workflow from an Openpose input to an OpenSim result.

It offers tools for:
- Cameras calibration,
- 2D pose estimation,
- Camera synchronization,
- Tracking the person of interest,
- Robust triangulation,
- Filtration, 
- Marker augmentation,
- OpenSim scaling and inverse kinematics

It has been tested on Windows, Linux and MacOS, and works for any Python version >= 3.9

Installation: 
# Open Anaconda prompt. Type:
# - conda create -n Pose2Sim python=3.9
# - conda activate Pose2Sim
# - conda install -c opensim-org opensim -y
# - pip install Pose2Sim

Usage: 
# First run Pose estimation and organize your directories (see Readme.md)
from Pose2Sim import Pose2Sim
Pose2Sim.calibration()
Pose2Sim.poseEstimation()
Pose2Sim.synchronization()
Pose2Sim.personAssociation()
Pose2Sim.triangulation()
Pose2Sim.filtering()
Pose2Sim.markerAugmentation()
# Then run OpenSim (see Readme.md)
'''


## INIT
import toml
import os
import time
from copy import deepcopy
import logging, logging.handlers
from datetime import datetime


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.8.2"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def setup_logging(session_dir):
    '''
    Create logging file and stream handlers
    '''
    with open(os.path.join(session_dir, 'logs.txt'), 'a+') as log_f: pass
    logging.basicConfig(format='%(message)s', level=logging.INFO, 
        handlers = [logging.handlers.TimedRotatingFileHandler(os.path.join(session_dir, 'logs.txt'), when='D', interval=7), logging.StreamHandler()])

    
def recursive_update(dict_to_update, dict_with_new_values):
    '''
    Update nested dictionaries without overwriting existing keys in any level of nesting
    
    Example: 
    dict_to_update = {'key': {'key_1': 'val_1', 'key_2': 'val_2'}}
    dict_with_new_values = {'key': {'key_1': 'val_1_new'}}
    returns {'key': {'key_1': 'val_1_new', 'key_2': 'val_2'}}
    while dict_to_update.update(dict_with_new_values) would return {'key': {'key_1': 'val_1_new'}}
    '''

    for key, value in dict_with_new_values.items():
        if key in dict_to_update and isinstance(value, dict) and isinstance(dict_to_update[key], dict):
            # Recursively update nested dictionaries
            dict_to_update[key] = recursive_update(dict_to_update[key], value)
        else:
            # Update or add new key-value pairs
            dict_to_update[key] = value

    return dict_to_update


def determine_level(config_dir):
    '''
    Determine the level at which the function is called.
    Level = 1: Trial folder
    Level = 2: Participant folder
    Level = 3: Session folder
    '''

    len_paths = [len(root.split(os.sep)) for root,dirs,files in os.walk(config_dir) if 'Config.toml' in files]
    if len_paths == []:
        raise FileNotFoundError('Please run Pose2Sim from a Session, Participant, or Trial directory.')
    level = max(len_paths) - min(len_paths) + 1
    return level


def read_config_files(config):
    '''
    Read Session, Participant, and Trial configuration files, 
    and output a dictionary with all the parameters.
    '''

    if type(config)==dict:
        level = 3 # log_dir = os.getcwd()
        config_dicts = [config]
        if config_dicts[0].get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_PROJECT_DIRECTORY>"})')
    else:
        # if launched without an argument, config == None, else it is the path to the config directory
        config_dir = ['.' if config == None else config][0]  
        level = determine_level(config_dir)
        
        # Trial level
        if level == 1: 
            try:
                # if batch
                session_config_dict = toml.load(os.path.join(config_dir, '..','..','Config.toml'))
                participant_config_dict = toml.load(os.path.join(config_dir, '..','Config.toml'))
                session_config_dict = recursive_update(session_config_dict,participant_config_dict)
                trial_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
                session_config_dict = recursive_update(session_config_dict,trial_config_dict)
            except:
                # if single trial
                session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            session_config_dict.get("project").update({"project_dir":config_dir})
            config_dicts = [session_config_dict]
        
        # Participant level
        if level == 2:
            session_config_dict = toml.load(os.path.join(config_dir, '..','Config.toml'))
            participant_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            config_dicts = []
            # Create config dictionaries for all trials of the participant
            for (root,dirs,files) in os.walk(config_dir):
                if 'Config.toml' in files and root != config_dir:
                    trial_config_dict = toml.load(os.path.join(root, files[0]))
                    # deep copy, otherwise session_config_dict is modified at each iteration within the config_dicts list
                    temp_dict = deepcopy(session_config_dict)
                    temp_dict = recursive_update(temp_dict,participant_config_dict)
                    temp_dict = recursive_update(temp_dict,trial_config_dict)
                    temp_dict.get("project").update({"project_dir":os.path.join(config_dir, os.path.relpath(root))})
                    if not os.path.basename(root) in temp_dict.get("project").get('exclude_from_batch'):
                        config_dicts.append(temp_dict)

        # Session level
        if level == 3:
            session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            config_dicts = []
            # Create config dictionaries for all trials of all participants of the session
            for (root,dirs,files) in os.walk(config_dir):
                if 'Config.toml' in files and root != config_dir:
                    # participant
                    if determine_level(root) == 2:
                        participant_config_dict = toml.load(os.path.join(root, files[0]))
                    # trial 
                    elif determine_level(root) == 1: 
                        trial_config_dict = toml.load(os.path.join(root, files[0]))
                        # deep copy, otherwise session_config_dict is modified at each iteration within the config_dicts list
                        temp_dict = deepcopy(session_config_dict)
                        temp_dict = recursive_update(temp_dict,participant_config_dict)
                        temp_dict = recursive_update(temp_dict,trial_config_dict)
                        temp_dict.get("project").update({"project_dir":os.path.join(config_dir, os.path.relpath(root))})
                        if not os.path.relpath(root) in [os.path.relpath(p) for p in temp_dict.get("project").get('exclude_from_batch')]:
                            config_dicts.append(temp_dict)

    return level, config_dicts


def calibration(config=None):
    '''
    Cameras calibration from checkerboards or from qualisys files.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.calibration import calibrate_cams_all

    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]
    try:
        session_dir = os.path.realpath([os.getcwd() if level==3 else os.path.join(os.getcwd(), '..') if level==2 else os.path.join(os.getcwd(), '..', '..')][0])
        [os.path.join(session_dir, c) for c in os.listdir(session_dir) if 'calib' in c.lower() ][0]
    except:
        session_dir = os.path.realpath(os.getcwd())
    config_dict.get("project").update({"project_dir":session_dir})

    # Set up logging
    setup_logging(session_dir)  
    currentDateAndTime = datetime.now()
    
    # Run calibration
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    logging.info("\n\n---------------------------------------------------------------------")
    logging.info("Camera calibration")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info("---------------------------------------------------------------------")
    logging.info(f"\nCalibration directory: {calib_dir}")
    start = time.time()
    
    calibrate_cams_all(config_dict)
    
    end = time.time()
    logging.info(f'\nCalibration took {end-start:.2f} s.')


def poseEstimation(config=None):
    '''
    Estimate pose using RTMLib
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    from Pose2Sim.poseEstimation import rtm_estimator # The name of the function might change

    level, config_dicts = read_config_files(config)

    if isinstance(config, dict):
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if not frame_range else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Pose estimation for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}")
    
        rtm_estimator(config_dict)
        
        end = time.time()
        elapsed = end - start 
        logging.info(f'\nPose estimation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')
    

def synchronization(config=None):
    '''
    Synchronize cameras if needed.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''   
    
    # Import the function
    from Pose2Sim.synchronization import synchronize_cams_all

    # Determine the level at which the function is called (session:3, participant:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)    

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info("Camera synchronization")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}")
        
        synchronize_cams_all(config_dict)
    
        end = time.time()
        elapsed = end-start 
        logging.info(f'\nSynchronization took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')
    
    
def personAssociation(config=None):
    '''
    Tracking one or several persons of interest.
    Needs a calibration file.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    from Pose2Sim.personAssociation import track_2d_all

    # Determine the level at which the function is called (session:3, participant:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)    

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Associating persons for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}")
    
        track_2d_all(config_dict)
    
        end = time.time()
        elapsed = end-start 
        logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')
    
    
def triangulation(config=None):
    '''
    Robust triangulation of 2D points coordinates.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.triangulation import triangulate_all

    # Determine the level at which the function is called (session:3, participant:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)  

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Triangulation of 2D points for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}")
        
        triangulate_all(config_dict)
    
        end = time.time()
        elapsed = end-start 
        logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')
 
    
def filtering(config=None):
    '''
    Filter trc 3D coordinates.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.filtering import filter_all

    # Determine the level at which the function is called (session:3, participant:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)    

    # Batch process all trials
    for config_dict in config_dicts:
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]
    
        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Filtering 3D coordinates for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}\n")
    
        filter_all(config_dict)


def markerAugmentation(config=None):
    '''
    Augment trc 3D coordinates. 
    Estimate the position of 43 additional markers.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.markerAugmentation import augmentTRC
    level, config_dicts = read_config_files(config)

    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    setup_logging(session_dir)

    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n\n---------------------------------------------------------------------")
        logging.info(f"Augmentation process for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info("---------------------------------------------------------------------")
        logging.info(f"\nProject directory: {project_dir}\n")

        augmentTRC(config_dict)

        end = time.time()
        elapsed = end-start 
        logging.info(f'\nMarker augmentation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')


def opensimProcessing(config=None):
    '''
    Uses OpenSim to run scaling based on a static trc pose
    and inverse kinematics based on a trc motion file.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    raise NotImplementedError('This has not been integrated yet. \nPlease read README.md for further explanation')
    
    # # TODO
    # from Pose2Sim.opensimProcessing import opensim_processing_all
    
    # # Determine the level at which the function is called (session:3, participant:2, trial:1)
    # level, config_dicts = read_config_files(config)

    # if type(config)==dict:
    #     config_dict = config_dicts[0]
    #     if config_dict.get('project').get('project_dir') == None:
    #         raise ValueError('Please specify the project directory in config_dict:\n \
    #                          config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # # Set up logging
    # session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..', '..'))
    # setup_logging(session_dir)    

    # # Batch process all trials
    # for config_dict in config_dicts:
    #     currentDateAndTime = datetime.now()
    #     start = time.time()
    #     project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
    #     seq_name = os.path.basename(project_dir)
    #     frame_range = config_dict.get('project').get('frame_range')
    #     frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

    #     logging.info("\n\n---------------------------------------------------------------------")
    #     # if static_file in project_dir: 
    #     #     logging.info(f"Scaling model with <STATIC TRC FILE>.")
    #     # else:
    #     #     logging.info(f"Running inverse kinematics <MOTION TRC FILE>.")
    #     logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    #     logging.info("---------------------------------------------------------------------")
    #     logging.info(f"\nOpenSim output directory: {project_dir}")
   
    #     opensim_processing_all(config_dict)
    
    #     end = time.time()
    #     elapsed = end-start 
    #     # if static_file in project_dir: 
    #     #     logging.info(f'Model scaling took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')
    #     # else:
    #     #     logging.info(f'Inverse kinematics took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.')

