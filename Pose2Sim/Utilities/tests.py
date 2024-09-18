#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################
    ## Pose2Sim tests                     ##
    ########################################
    
    - SINGLE PERSON:
        - calibration conversion from .qca.txt
        - pose estimation
        - synchronization
        - person association
        - triangulation
        - filtering
        - marker augmentation
        - opensim scaling and inverse kinematics
        
        - run all

    - MULTI PERSON:
        - calibration conversion from .qca.txt
        - pose estimation
        - NO synchronization
        - person association
        - triangulation
        - filtering
        - marker augmentation
        - opensim scaling and inverse kinematics

        - run all

    - BATCH SESSION, RUN ALL:
        - Calibration conversion from .qca.txt
        - Single person:
            - pose estimation
            - NO synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation
            - opensim scaling and inverse kinematics
        - Multi-person:
            - pose estimation
            - NO synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation
            - opensim scaling and inverse kinematics
            

    N.B.: 
    1. Calibration from scene dimensions is not tested, as it requires the 
    user to click points on the image. 
    2. OpenSim scaling and IK are not tested yet
    3. Not all possible configuration parameters are extensively tested.
    
    Usage: 
    cd Pose2Sim/Utilities
    python tests.py
'''

## INIT
import os
import toml
from unittest.mock import patch
import unittest

from Pose2Sim import Pose2Sim


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
class TestWorkflow(unittest.TestCase):
    @patch('builtins.input', return_value='no')  # Mock input() to return 'no'
    def test_workflow(self, mock_input):
        '''
        SINGLE-PERSON, MULTI-PERSON, BATCH PROCESSING:
            - calibration
            - pose estimation
            - synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation
            - opensim scaling and inverse kinematics
            - run all

        N.B.: Calibration from scene dimensions is not tested, as it requires the 
        user to click points on the image. 
        Not all possible configuration parameters are extensively tested.
        
        Usage: 
        from Pose2Sim import tests; tests.test_workflow()
        python tests.py 
        '''


        ###################
        # SINGLE-PERSON   #
        ###################

        project_dir = '../Demo_SinglePerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))

        os.chdir(project_dir)
        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("pose").update({"mode":'lightweight'})
        config_dict.get("pose").update({"display_detection":False})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict['filtering']['display_figures'] = False

        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        Pose2Sim.synchronization(config_dict)
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        Pose2Sim.markerAugmentation(config_dict)
        Pose2Sim.kinematics(config_dict)

        config_dict.get("pose").update({"overwrite_pose":False})
        Pose2Sim.runAll(config_dict)
        

        ####################
        # MULTI-PERSON     #
        ####################
        
        project_dir = '../Demo_MultiPerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))
        
        os.chdir(project_dir)
        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("pose").update({"mode":'lightweight'})
        config_dict.get("pose").update({"display_detection":False})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict['filtering']['display_figures'] = False

        # Step by step
        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        # Pose2Sim.synchronization(config_dict) # No synchronization for multi-person for now
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        Pose2Sim.markerAugmentation(config_dict)
        Pose2Sim.kinematics(config_dict)

        # Run all
        config_dict.get("pose").update({"overwrite_pose":False})
        Pose2Sim.runAll(config_dict, do_synchronization=False)


        ####################
        # BATCH PROCESSING #
        ####################

        project_dir = '../Demo_Batch'
        os.chdir(project_dir)

        Pose2Sim.runAll(do_synchronization=False)


if __name__ == '__main__':
    unittest.main()
