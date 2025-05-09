#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################
    ## Pose2Sim tests                     ##
    ########################################
    
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

    Display of images and plots deactivated for testing purposes. Synchronization deactivated in multi_person mode.
    Testing single person, multi person, and batch processing.
    Testing stage by stage or all at once (runAll).
    Testing openvino backend and cpu device in lightweight and balanced modes, automatic backend and device selection with body pose model in RTMO mode.
    Testing overwritting pose or not.
    Testing with det_frequency
    Testing automatic and manual participant height estimation, frame_rate detection.
    Testing det_frequency 1 and 10.
    Testing synchronization with all markers or only ['RWrist'].
    Testing with and without marker augmentation.
    
    N.B.: Calibration from scene dimensions is not tested, as it requires the 
    user to click points on the image. 
    Not all possible configuration parameters are extensively tested.
    
    Usage: 
    tests_pose2sim
        OR
    cd Pose2Sim/Utilities
    python tests.py
        OR
    from Pose2Sim.Utilities.tests import TestWorkflow; TestWorkflow.test_workflow(mock_input='no')
'''

## INIT
import os
import sys
import toml
from unittest.mock import patch
import unittest

from Pose2Sim import Pose2Sim


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

            Display of images and plots deactivated for testing purposes. Synchronization deactivated in multi_person mode.
            Testing single person, multi person, and batch processing.
            Testing stage by stage or all at once (runAll).
            Testing openvino backend and cpu device in lightweight and balanced modes, automatic backend and device selection with body pose model in RTMO mode.
            Testing overwritting pose or not.
            Testing with det_frequency
            Testing automatic and manual participant height estimation, frame_rate detection.
            Testing det_frequency 1 and 10.
            Testing synchronization with all markers or only ['RWrist'].
            Testing with and without marker augmentation.
            
            N.B.: Calibration from scene dimensions is not tested, as it requires the 
            user to click points on the image. 
            Not all possible configuration parameters are extensively tested.
            
            Usage: 
            cd Pose2Sim/Utilities
            python tests.py
                OR
            from Pose2Sim.Utilities.tests import TestWorkflow; TestWorkflow.test_workflow(mock_input='no')
            '''

        root_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(root_dir)

        ###################
        # SINGLE-PERSON   #
        ###################

        project_dir = '../Demo_SinglePerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))

        # lightweight, openvino, cpu
        os.chdir(project_dir)
        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("pose").update({"mode":'lightweight'})
        config_dict.get("pose").update({"display_detection":False})
        config_dict.get("pose").update({"backend":'openvino'})
        config_dict.get("pose").update({"device":'cpu'})
        config_dict.get("synchronization").update({"synchronization_gui":False})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict.get("filtering").update({"display_figures":False})

        # Step by step
        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        Pose2Sim.synchronization(config_dict)
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        Pose2Sim.markerAugmentation(config_dict)
        Pose2Sim.kinematics(config_dict)


        # Run all
        # overwrite pose, balanced
        config_dict.get("project").update({"participant_height":1.7})
        config_dict.get("project").update({"frame_rate":60})
        config_dict.get("pose").update({"det_frequency":10})
        config_dict.get("pose").update({"mode":'balanced'})
        config_dict.get("pose").update({"overwrite_pose":True})
        config_dict.get("pose").update({"save_video":'none'})
        config_dict.get('synchronization').update({'keypoints_to_consider':['RWrist']})
        Pose2Sim.runAll(config_dict)
        

        ####################
        # MULTI-PERSON     #
        ####################
        
        project_dir = '../Demo_MultiPerson'
        config_dict = toml.load(os.path.join(project_dir, 'Config.toml'))
        
        # Body model with RTMO
        os.chdir(project_dir)
        config_dict.get("project").update({"project_dir":project_dir})
        config_dict.get("pose").update({"pose_model":'Body'})
        config_dict.get("pose").update({"mode":"""{'pose_class':'RTMO', 
                                                'pose_model':'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip', 
                                                'pose_input_size':[640, 640]}"""})
        config_dict.get("pose").update({"display_detection":False})
        config_dict.get("pose").update({"save_video":'none'})
        config_dict.get("synchronization").update({"synchronization_gui":False})
        config_dict.get("synchronization").update({"display_sync_plots":False})
        config_dict.get("filtering").update({"display_figures":False})

        # Step by step
        Pose2Sim.calibration(config_dict)
        Pose2Sim.poseEstimation(config_dict)
        # Pose2Sim.synchronization(config_dict) # No test for synchronization for multi-person
        Pose2Sim.personAssociation(config_dict)
        Pose2Sim.triangulation(config_dict)
        Pose2Sim.filtering(config_dict)
        # Pose2Sim.markerAugmentation(config_dict) # Marker augmentation requires markers that are not provided by RTMO: ['RHeel', 'RBigToe', 'RSmallToe', 'LSmallToe', 'LHeel', 'LBigToe']
        Pose2Sim.kinematics(config_dict)

        # Run all
        # No marker augmentation
        config_dict.get("pose").update({"tracking_mode":'deepsort'})
        config_dict.get("pose").update({"deepsort_params":"""{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8, 'embedder':None}"""})
        Pose2Sim.runAll(config_dict, do_synchronization=False, do_markerAugmentation=False)


        ####################
        # BATCH PROCESSING #
        ####################

        project_dir = '../Demo_Batch'
        os.chdir(project_dir)

        Pose2Sim.runAll(do_synchronization=False)


def main():
    '''
    Entry point for running Pose2Sim tests.
    Can be called from command line or as a console script.
    '''

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWorkflow)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())


if __name__ == '__main__':
    main()
