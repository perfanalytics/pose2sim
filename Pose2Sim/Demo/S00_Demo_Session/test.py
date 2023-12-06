import os
import toml

def test_workflow():
  from Pose2Sim import Pose2Sim
  config_dict = toml.load(os.path.join('User', 'Config.toml'))
  config_dict['filtering']['display_figures'] = False
  Pose2Sim.calibration(config_dict)
  Pose2Sim.personAssociation(config_dict)
  Pose2Sim.triangulation(config_dict)
  Pose2Sim.filtering(config_dict)
