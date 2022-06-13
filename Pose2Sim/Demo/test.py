import os
print('1')
from Pose2Sim import Pose2Sim
print('2')
Pose2Sim.calibrateCams(os.path.join('User', 'test.toml'))
print('3')
Pose2Sim.track2D(os.path.join('User', 'test.toml'))
Pose2Sim.triangulate3D(os.path.join('User', 'test.toml'))
Pose2Sim.filter3D(os.path.join('User', 'test.toml'))
