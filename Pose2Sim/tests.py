#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################
    ## Pose2Sim tests                     ##
    ########################################
    
    - BATCH SESSION:
        - Calibration conversion from .qca.txt
        - Single person:
            - synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation
        - Multi-person:
            - synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation
            
    - SINGLE TRIAL:
        - calibration
        - synchronization
        - person association
        - triangulation
        - filtering
        - marker augmentation

    N.B.: 
    1. Calibration from scene dimensions is not tested, as it requires the 
    user to click points on the image. 
    2. OpenSim scaling and IK are not tested yet
    3. Not all possible configuration parameters are extensively tested.
    
    Usage: 
    from Pose2Sim.S00_Demo_BatchSession import tests; tests.test_workflow()
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
__version__ = "0.8.2"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
class TestWorkflow(unittest.TestCase):
    @patch('builtins.input', return_value='no')  # Mock input() to return 'yes'
    def test_workflow(self, mock_input):
        '''
        SINGLE-PERSON and MULTI-PERSON:
            - calibration
            - pose estimation
            - synchronization
            - person association
            - triangulation
            - filtering
            - marker augmentation

        N.B.: Calibration from scene dimensions is not tested, as it requires the 
        user to click points on the image. 
        Not all possible configuration parameters are extensively tested.
        Scaling and inverse kinematics are not tested yet.
        
        Usage: 
        from Pose2Sim import tests; tests.test_workflow()
        python tests.py 
        '''

        ##################
        # MULTI-PERSON   #
        ##################
        
        project_dir = 'Demo_MultiPerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))
        
        os.chdir(project_dir)
        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("project").update({"multi_person":False})
        config_dict.get("pose").update({"mode":'lightweight'})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict['filtering']['display_figures'] = False

        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        Pose2Sim.synchronization(config_dict)
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        Pose2Sim.markerAugmentation(config_dict)
        # Pose2Sim.kinematics(config_dict)


        ##################
        # SINGLE-PERSON  #
        ##################

        project_dir = 'Demo_SinglePerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))

        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("project").update({"multi_person":True})
        config_dict.get("pose").update({"mode":'lightweight'})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict.get("markerAugmentation").update({"participant_height":[1.21, 1.72]})
        config_dict.get("markerAugmentation").update({"participant_mass":[25.0, 70.0]})
        # config_dict['triangulation']['reorder_trc'] = False

        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        Pose2Sim.synchronization(config_dict)
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        Pose2Sim.markerAugmentation(config_dict)
        # Pose2Sim.kinematics(config_dict)
        
    
if __name__ == '__main__':
    unittest.main()
