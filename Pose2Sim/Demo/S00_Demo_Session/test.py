import toml

def test_workflow():
  
  from Pose2Sim import Pose2Sim
  
  config_dict = toml.load('Config.toml')
  config_dict.get("project").update({"project_dir":"."})
  Pose2Sim.calibration(config_dict)
  
  config_dict.get("project").update({"project_dir":r"S00_P00_Participant\S00_P00_T00_StaticTrial"})
  config_dict['filtering']['display_figures'] = False
  # Pose2Sim.poseEstimation(config_dict)
  # Pose2Sim.synchronization(config_dict)
  Pose2Sim.personAssociation(config_dict)
  Pose2Sim.triangulation(config_dict)
  Pose2Sim.filtering(config_dict)
  # Pose2Sim.kinematics(config_dict)

  # config_dict.get("project").update({"project_dir":r"S00_P00_Participant\S00_P00_T00_BalancingTrial"})
  # config_dict['filtering']['display_figures'] = False
  # # Pose2Sim.poseEstimation(config_dict)
  # # Pose2Sim.synchronization(config_dict)
  # Pose2Sim.personAssociation(config_dict)
  # Pose2Sim.triangulation(config_dict)
  # Pose2Sim.filtering(config_dict)
  # # Pose2Sim.kinematics(config_dict)
