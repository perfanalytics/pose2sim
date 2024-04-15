def test_workflow():
    '''
    Test calibration,
    person association, triangulation, filtering, marker augmentation,
    for single and multiple person analysis
    '''

    import os
    import toml
    from Pose2Sim import Pose2Sim
    
    # CALIBRATION
    config_dict = toml.load('Config.toml')
    config_dict.get("project").update({"project_dir":"."})
    Pose2Sim.calibration(config_dict)
    # edit config_dict for calibration based on keypoints
    # Pose2Sim.calibration(config_dict)
    
    # SINGLE PERSON
    # Static trial
    project_dir = os.path.join("S00_P00_SingleParticipant","S00_P00_T00_StaticTrial")
    config_dict.get("project").update({"project_dir":project_dir})
    # Pose2Sim.poseEstimation(config_dict)
    # Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # Balancing trial
    project_dir = os.path.join("S00_P00_SingleParticipant","S00_P00_T01_BalancingTrial")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict['filtering']['display_figures'] = False
    # Pose2Sim.poseEstimation(config_dict)
    # Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)


    # MULTI PERSON
    # Static trial person 1
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T00_StaticTrialParticipant1")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("markerAugmentation").update({"participant_height":1.21})
    config_dict.get("markerAugmentation").update({"participant_mass":25.0})
    # Pose2Sim.poseEstimation(config_dict)
    # Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # Static trial person 2
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T01_StaticTrialParticipant2")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("markerAugmentation").update({"participant_height":1.72})
    config_dict.get("markerAugmentation").update({"participant_mass":70.0})
    # Pose2Sim.poseEstimation(config_dict)
    # Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)

    # Balancing & Yoga trial
    project_dir = os.path.join("S00_P01_MultiParticipants","S00_P01_T02_Participants1-2")
    config_dict.get("project").update({"project_dir":project_dir})
    config_dict.get("project").update({"multi_person":True})
    config_dict.get("markerAugmentation").update({"participant_height":[1.21, 1.72]})
    config_dict.get("markerAugmentation").update({"participant_mass":[25.0, 70.0]})
    config_dict['triangulation']['reorder_trc'] = False
    # Pose2Sim.poseEstimation(config_dict)
    # Pose2Sim.synchronization(config_dict)
    Pose2Sim.personAssociation(config_dict)
    Pose2Sim.triangulation(config_dict)
    Pose2Sim.filtering(config_dict)
    Pose2Sim.markerAugmentation(config_dict)
    # Pose2Sim.kinematics(config_dict)
