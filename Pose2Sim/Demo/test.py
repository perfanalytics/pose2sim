import os

def test_workflow():
  from Pose2Sim import Pose2Sim
  Pose2Sim.calibrateCams(os.path.join('User', 'test.toml'))
  Pose2Sim.track2D(os.path.join('User', 'test.toml'))
  Pose2Sim.triangulate3D(os.path.join('User', 'test.toml'))
  Pose2Sim.filter3D(os.path.join('User', 'test.toml'))
