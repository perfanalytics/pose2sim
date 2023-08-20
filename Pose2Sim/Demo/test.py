import os

def test_workflow():
  from Pose2Sim import Pose2Sim
  Pose2Sim.calibration(os.path.join('User', 'test.toml'))
  Pose2Sim.personAssociation(os.path.join('User', 'test.toml'))
  Pose2Sim.triangulation(os.path.join('User', 'test.toml'))
  Pose2Sim.filtering(os.path.join('User', 'test.toml'))
