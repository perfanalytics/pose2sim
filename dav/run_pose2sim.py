
from Pose2Sim import Pose2Sim
Pose2Sim.runAll(do_calibration=True, 
                do_poseEstimation=True, 
                do_synchronization=False, 
                do_personAssociation=True, 
                do_triangulation=True, 
                do_filtering=True, 
                do_markerAugmentation=True, 
                do_kinematics=True)
    