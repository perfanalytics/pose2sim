#! /usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################
    ## Pose2Sim tests                     ##
    ########################################
    
    - BATCH SESSION:
        - Calibration
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
    
    Usage: 
    from Pose2Sim.S00_Demo_BatchSession import tests; tests.test_workflow()
    python tests.py 
'''

## INIT
import os
import toml
from Pose2Sim import Pose2Sim


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.8'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def test_workflow():
    '''
    - BATCH SESSION:
        - Calibration
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
        
    Usage: 
    from Pose2Sim.S00_Demo_BatchSession import tests; tests.test_workflow()
    python tests.py 
    '''
    
    config_dict = toml.load('Config.toml')
    
    
    #################
    # BATCH SESSION #
    #################
    
    ###############
    # Calibration #
    ###############
    
    config_dict.get("project").update({"project_dir":"."})
    Pose2Sim.calibration(config_dict)
    # edit config_dict for calibration based on keypoints
    # Pose2Sim.calibration(config_dict)
    
    
    #################
    # Single person #
    #################
    
    # STATIC TRIAL
    project_dir = os.path.join("S00_P00_SingleParticipant","S00_P00_T00_StaticTrial")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("synchronization").update({"reset_sync":True})
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # BALANCING TRIAL
    project_dir = os.path.join("S00_P00_SingleParticipant","S00_P00_T01_BalancingTrial")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict['filtering']['display_figures'] = False
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)
    
    
    ################
    # Multi person #
    ################
    
    # STATIC TRIAL PERSON 1
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T00_StaticTrialParticipant1")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("markerAugmentation").update({"participant_height":1.21})
    config_dict.get("markerAugmentation").update({"participant_mass":25.0})
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # STATIC TRIAL PERSON 2
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T01_StaticTrialParticipant2")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("markerAugmentation").update({"participant_height":1.72})
    config_dict.get("markerAugmentation").update({"participant_mass":70.0})
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # BALANCING & YOGA TRIAL
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T02_Participants1-2")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("project").update({"multi_person":True})
    config_dict.get("markerAugmentation").update({"participant_height":[1.21, 1.72]})
    config_dict.get("markerAugmentation").update({"participant_mass":[25.0, 70.0]})
    config_dict['triangulation']['reorder_trc'] = False
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)
    
    
    #################
    # SINGLE TRIAL  #
    #################
    
    config_dict = toml.load('../S01_Demo_SingleTrial/Config.toml')
    project_dir = os.path.join("../S01_Demo_SingleTrial")
    os.chdir(project_dir)
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("synchronization").update({"display_sync_plots":False})
    config_dict['filtering']['display_figures'] = False
    Pose2Sim.calibration(config_dict)
    # Pose2Sim.poseEstimation(config_dict)
    Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)
    
    
if __name__ == '__main__':
    test_workflow()