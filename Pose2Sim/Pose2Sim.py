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
Pose2Sim.kinematics()
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
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def setup_logging(session_dir):
    '''
    Create logging file and stream handlers
    '''

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
    Level = 2: Root folder
    '''

    len_paths = [len(root.split(os.sep)) for root,dirs,files in os.walk(config_dir) if 'Config.toml' in files]
    if len_paths == []:
        raise FileNotFoundError('You need a Config.toml file in each trial or root folder.')
    level = max(len_paths) - min(len_paths) + 1
    return level


def read_config_files(config):
    '''
    Read Root and Trial configuration files,
    and output a dictionary with all the parameters.
    '''

    if type(config)==dict:
        level = 2 # log_dir = os.getcwd()
        config_dicts = [config]
        if config_dicts[0].get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_PROJECT_DIRECTORY>"})')
    else:
        # if launched without an argument, config == None, else it is the path to the config directory
        config_dir = ['.' if config == None else config][0]
        level = determine_level(config_dir)

        # Trial level
        if level == 1: # Trial
            try:
                # if batch
                session_config_dict = toml.load(os.path.join(config_dir, '..','Config.toml'))
                trial_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
                session_config_dict = recursive_update(session_config_dict,trial_config_dict)
            except:
                # if single trial
                session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            session_config_dict.get("project").update({"project_dir":config_dir})
            config_dicts = [session_config_dict]

        # Root level
        if level == 2:
            session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            config_dicts = []
            # Create config dictionaries for all trials of the participant
            for (root,dirs,files) in os.walk(config_dir):
                if 'Config.toml' in files and root != config_dir:
                    trial_config_dict = toml.load(os.path.join(root, files[0]))
                    # deep copy, otherwise session_config_dict is modified at each iteration within the config_dicts list
                    temp_dict = deepcopy(session_config_dict)
                    temp_dict = recursive_update(temp_dict,trial_config_dict)
                    temp_dict.get("project").update({"project_dir":os.path.join(config_dir, os.path.relpath(root))})
                    if not os.path.basename(root) in temp_dict.get("project").get('exclude_from_batch'):
                        config_dicts.append(temp_dict)

    return level, config_dicts

class Pose2SimPipeline:
    def __init__(self, config=None):
        self.level, self.config_dicts = read_config_files(config)
        try:
            self.session_dir = os.path.realpath([os.getcwd() if self.level==2 else os.path.join(os.getcwd(), '..')][0])
            [os.path.join(self.session_dir, c) for c in os.listdir(self.session_dir) if 'calib' in c.lower() and not c.lower().endswith('.py')][0]
        except:
            self.session_dir = os.path.realpath(os.getcwd())
        use_custom_logging = self.config_dicts[0].get('logging').get('use_custom_logging')
        if not use_custom_logging:
            setup_logging(self.session_dir)

    def _log_step_header(self, step_name, config_dict):
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = "all frames" if not frame_range or frame_range in ('all','auto') else f"frames {frame_range[0]} to {frame_range[1]}"
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"{step_name} for {seq_name}, for {frames}.")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

    def calibration(self):
        from Pose2Sim.calibration import calibrate_cams_all
        config_dict = self.config_dicts[0]
        config_dict.get("project").update({"session_dir": self.session_dir})

        try:
            calib_dirs = [
                os.path.join(self.session_dir, c)
                for c in os.listdir(self.session_dir)
                if os.path.isdir(os.path.join(self.session_dir, c)) and 'calib' in c.lower()
            ]
            calib_dir = calib_dirs[0]
        except IndexError:
            logging.error('Could not find the calibration folder or files.')
            raise ValueError('Could not find the calibration folder or files.')

        logging.info("\n---------------------------------------------------------------------")
        logging.info("Camera calibration")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Calibration directory: {calib_dir}")
        logging.info("---------------------------------------------------------------------\n")

        start = time.time()
        calibrate_cams_all(config_dict)
        elapsed = time.time() - start
        logging.info(f'\nCalibration took {elapsed:.2f} seconds.\n')

    def poseEstimation(self):
        from Pose2Sim.poseEstimation import estimate_pose_all
        for config_dict in self.config_dicts:
            self._log_step_header("Pose estimation", config_dict)
            start = time.time()
            estimate_pose_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nPose estimation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def synchronization(self):
        from Pose2Sim.synchronization import synchronize_cams_all
        for config_dict in self.config_dicts:
            self._log_step_header("Camera synchronization", config_dict)
            start = time.time()
            synchronize_cams_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nSynchronization took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def personAssociation(self):
        from Pose2Sim.personAssociation import associate_all
        for config_dict in self.config_dicts:
            self._log_step_header("Associating persons", config_dict)
            start = time.time()
            associate_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def triangulation(self):
        from Pose2Sim.triangulation import triangulate_all
        for config_dict in self.config_dicts:
            self._log_step_header("Triangulation of 2D points", config_dict)
            start = time.time()
            triangulate_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def filtering(self):
        from Pose2Sim.filtering import filter_all
        for config_dict in self.config_dicts:
            self._log_step_header("Filtering 3D coordinates", config_dict)
            filter_all(config_dict)
            logging.info('\n')

    def markerAugmentation(self):
        from Pose2Sim.markerAugmentation import augment_markers_all
        for config_dict in self.config_dicts:
            self._log_step_header("Augmentation process", config_dict)
            start = time.time()
            augment_markers_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nMarker augmentation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def kinematics(self):
        from Pose2Sim.kinematics import kinematics_all
        for config_dict in self.config_dicts:
            self._log_step_header("OpenSim scaling and inverse kinematics", config_dict)
            start = time.time()
            kinematics_all(config_dict)
            elapsed = time.time() - start
            logging.info(f'\nOpenSim scaling and inverse kinematics took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def runAll(self, do_calibration=True, do_poseEstimation=True, do_synchronization=True, 
                do_personAssociation=True, do_triangulation=True, do_filtering=True, 
                do_markerAugmentation=True, do_kinematics=True):
        logging.info("\n\n=====================================================================")
        logging.info(f"RUNNING ALL.")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {self.session_dir}\n")
        logging.info("=====================================================================\n")

        overall_start = time.time()

        if do_calibration:
            logging.info("\n\n=====================================================================")
            logging.info('Running calibration...')
            logging.info("=====================================================================")
            self.calibration()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping calibration.')
            logging.info("=====================================================================")

        if do_poseEstimation:
            logging.info("\n\n=====================================================================")
            logging.info('Running pose estimation...')
            logging.info("=====================================================================")
            self.poseEstimation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping pose estimation.')
            logging.info("=====================================================================")

        if do_synchronization:
            logging.info("\n\n=====================================================================")
            logging.info('Running synchronization...')
            logging.info("=====================================================================")
            self.synchronization()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping synchronization.')
            logging.info("=====================================================================")

        if do_personAssociation:
            logging.info("\n\n=====================================================================")
            logging.info('Running person association...')
            logging.info("=====================================================================")
            self.personAssociation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping person association.')
            logging.info("=====================================================================")

        if do_triangulation:
            logging.info("\n\n=====================================================================")
            logging.info('Running triangulation...')
            logging.info("=====================================================================")
            self.triangulation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping triangulation.')
            logging.info("=====================================================================")

        if do_filtering:
            logging.info("\n\n=====================================================================")
            logging.info('Running filtering...')
            logging.info("=====================================================================")
            self.filtering()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping filtering.')
            logging.info("=====================================================================")

        if do_markerAugmentation:
            logging.info("\n\n=====================================================================")
            logging.info('Running marker augmentation.')
            logging.info("=====================================================================")
            self.markerAugmentation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping marker augmentation.')
            logging.info("\n\n=====================================================================")

        if do_kinematics:
            logging.info("\n\n=====================================================================")
            logging.info("Running OpenSim processing.")
            logging.info("=====================================================================")
            self.kinematics()
        else:
            logging.info("\n\n=====================================================================")
            logging.info('Skipping OpenSim processing.')
            logging.info("\n\n=====================================================================")

        logging.info("Pose2Sim pipeline completed.")
        overall_elapsed = time.time() - overall_start
        logging.info(f'\nRUNNING ALL FUNCTIONS TOOK  {time.strftime("%Hh%Mm%Ss", time.gmtime(overall_elapsed))}.\n')

def calibration(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.calibration()

def poseEstimation(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.poseEstimation()

def synchronization(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.synchronization()

def personAssociation(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.personAssociation()

def triangulation(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.triangulation()

def filtering(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.filtering()

def markerAugmentation(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.markerAugmentation()

def kinematics(config=None):
    pipeline = Pose2SimPipeline(config)
    pipeline.kinematics()

def runAll(config=None, **kwargs):
    pipeline = Pose2SimPipeline(config)
    pipeline.runAll(**kwargs)