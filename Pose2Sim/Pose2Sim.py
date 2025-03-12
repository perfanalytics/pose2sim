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

import os
import time
import logging
from datetime import datetime
from config import Config

# AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# CLASS
class Pose2SimPipeline:
    def __init__(self, config_input=None):
        self.config = Config(config_input)

        if not self.config.use_custom_logging:
            logging.basicConfig(
                format='%(message)s',
                level=logging.INFO,
                handlers=[
                    logging.handlers.TimedRotatingFileHandler(os.path.join(self.config.session_dir, 'logs.txt'), when='D', interval=7),
                    logging.StreamHandler()
                ]
            )

    def _log_step_header(self, step_name, sub_config):
        project_dir = sub_config.project_dir
        seq_name = os.path.basename(project_dir)
        frame_range = sub_config.frame_range
        frames = "all frames" if not frame_range or frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"{step_name} for {seq_name}, for {frames}.")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

    def calibration(self):
        from Pose2Sim.calibration import calibrate_cams_all

        sub_config = self.config.sub_configs[0]

        calib_dir = sub_config.calib_dir

        logging.info("\n---------------------------------------------------------------------")
        logging.info("Camera calibration")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Calibration directory: {calib_dir}")
        logging.info("---------------------------------------------------------------------\n")

        start = time.time()
        calibrate_cams_all(sub_config)
        elapsed = time.time() - start
        logging.info(f'\nCalibration took {elapsed:.2f} seconds.\n')

    def poseEstimation(self):
        from Pose2Sim.poseEstimation import estimate_pose_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Pose estimation", sub_config)
            start = time.time()
            estimate_pose_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nPose estimation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def synchronization(self):
        from Pose2Sim.synchronization import synchronize_cams_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Camera synchronization", sub_config)
            start = time.time()
            synchronize_cams_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nSynchronization took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def personAssociation(self):
        from Pose2Sim.personAssociation import associate_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Associating persons", sub_config)
            start = time.time()
            associate_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def triangulation(self):
        from Pose2Sim.triangulation import triangulate_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Triangulation of 2D points", sub_config)
            start = time.time()
            triangulate_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def filtering(self):
        from Pose2Sim.filtering import filter_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Filtering 3D coordinates", sub_config)
            filter_all(sub_config)
            logging.info('\n')

    def markerAugmentation(self):
        from Pose2Sim.markerAugmentation import augment_markers_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("Augmentation process", sub_config)
            start = time.time()
            augment_markers_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nMarker augmentation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def kinematics(self):
        from Pose2Sim.kinematics import kinematics_all
        for sub_config in self.config.sub_configs:
            self._log_step_header("OpenSim scaling and inverse kinematics", sub_config)
            start = time.time()
            kinematics_all(sub_config)
            elapsed = time.time() - start
            logging.info(f'\nOpenSim scaling and inverse kinematics took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    def runAll(self, do_calibration=True, do_poseEstimation=True, do_synchronization=True, 
               do_personAssociation=True, do_triangulation=True, do_filtering=True, 
               do_markerAugmentation=True, do_kinematics=True):
        logging.info("\n\n=====================================================================")
        logging.info("RUNNING ALL.")
        logging.info(f"On {datetime.now().strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {self.session_dir}\n")
        logging.info("=====================================================================\n")

        overall_start = time.time()

        if do_calibration:
            logging.info("\n\n=====================================================================")
            logging.info("Running calibration...")
            logging.info("=====================================================================")
            self.calibration()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping calibration.")
            logging.info("=====================================================================")
        if do_poseEstimation:
            logging.info("\n\n=====================================================================")
            logging.info("Running pose estimation...")
            logging.info("=====================================================================")
            self.poseEstimation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping pose estimation.")
            logging.info("=====================================================================")
        if do_synchronization:
            logging.info("\n\n=====================================================================")
            logging.info("Running synchronization...")
            logging.info("=====================================================================")
            self.synchronization()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping synchronization.")
            logging.info("=====================================================================")
        if do_personAssociation:
            logging.info("\n\n=====================================================================")
            logging.info("Running person association...")
            logging.info("=====================================================================")
            self.personAssociation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping person association.")
            logging.info("=====================================================================")
        if do_triangulation:
            logging.info("\n\n=====================================================================")
            logging.info("Running triangulation...")
            logging.info("=====================================================================")
            self.triangulation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping triangulation.")
            logging.info("=====================================================================")
        if do_filtering:
            logging.info("\n\n=====================================================================")
            logging.info("Running filtering...")
            logging.info("=====================================================================")
            self.filtering()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping filtering.")
            logging.info("=====================================================================")
        if do_markerAugmentation:
            logging.info("\n\n=====================================================================")
            logging.info("Running marker augmentation.")
            logging.info("=====================================================================")
            self.markerAugmentation()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping marker augmentation.")
            logging.info("\n\n=====================================================================")
        if do_kinematics:
            logging.info("\n\n=====================================================================")
            logging.info("Running OpenSim processing.")
            logging.info("=====================================================================")
            self.kinematics()
        else:
            logging.info("\n\n=====================================================================")
            logging.info("Skipping OpenSim processing.")
            logging.info("\n\n=====================================================================")
        logging.info("Pose2Sim pipeline completed.")
        overall_elapsed = time.time() - overall_start
        logging.info(f'\nRUNNING ALL FUNCTIONS TOOK {time.strftime("%Hh%Mm%Ss", time.gmtime(overall_elapsed))}.\n')

# FUNCTIONS
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
