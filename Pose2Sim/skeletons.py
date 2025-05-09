#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## SKELETONS DEFINITIONS                                                 ##
###########################################################################

The definition and hierarchy of the following skeletons are available: 
- RTMPose HALPE_26, COCO_133, COCO_133_WRIST, COCO_17, HAND, FACE, ANIMAL
- OpenPose BODY_25B, BODY_25, BODY_135, COCO, MPII
- Mediapipe BLAZEPOSE
- AlphaPose HALPE_26, HALPE_68, HALPE_136, COCO_133, COCO, MPII 
(for COCO and MPII, AlphaPose must be run with the flag "--format cmu")
- DeepLabCut CUSTOM: the skeleton will be defined in Config.toml

N.B.: Not all face and hand keypoints are reported in the skeleton architecture, 
since some are redundant for the orientation of some bodies.

N.B.: The corresponding OpenSim model files are provided in the "Pose2Sim/Empty project" folder.
If you wish to use any other, you will need to adjust the markerset in the .osim model file, 
as well as in the scaling and IK setup files.

Check the skeleton structure with:
from anytree import Node, RenderTree
for pre, _, node in RenderTree(model): 
    print(f'{pre}{node.name} id={node.id}')
'''

## INIT
from anytree import Node


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


'''HALPE_26 (full-body without hands, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
HALPE_26 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Head", id=17, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''COCO_133_WRIST (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_133_WRIST = Node("Hip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=20, children=[
                    Node("RSmallToe", id=21),
                ]),
                Node("RHeel", id=22),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=17, children=[
                    Node("LSmallToe", id=18),
                ]),
                Node("LHeel", id=19),
            ]),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0, children=[
            Node("REye", id=2),
            Node("LEye", id=1),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=114),
                    Node("RIndex", id=117),
                    Node("RPinky", id=129),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=93),
                    Node("LIndex", id=96),
                    Node("LPinky", id=108),
                ])
            ]),
        ]),
    ]),
])


'''COCO_133 (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_133 = Node("Hip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=20, children=[
                    Node("RSmallToe", id=21),
                ]),
                Node("RHeel", id=22),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=17, children=[
                    Node("LSmallToe", id=18),
                ]),
                Node("LHeel", id=19),
            ]),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0, children=[
            Node("REye", id=2),
            Node("LEye", id=1),
            Node("REar", id=4),
            Node("LEar", id=3),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb1", id=113, children=[
                        Node("RThumb", id=114, children=[
                            Node("RThumb3", id=115, children=[
                                Node("RThumb4", id=116),
                            ]),
                        ]),
                    ]),
                    Node("RIndex", id=117, children=[
                        Node("RIndex2", id=118, children=[
                            Node("RIndex3", id=119, children=[
                                Node("RIndex4", id=120),
                            ]),
                        ]),
                    ]),
                    Node("RMiddle1", id=121, children=[
                        Node("RMiddle2", id=122, children=[
                            Node("RMiddle3", id=123, children=[
                                Node("RMiddle4", id=124),
                            ]),
                        ]),
                    ]),
                    Node("RRing1", id=125, children=[
                        Node("RRing2", id=126, children=[
                            Node("RRing3", id=127, children=[
                                Node("RRing4", id=128),
                            ]),
                        ]),
                    ]),
                    Node("RPinky", id=129, children=[
                        Node("RPinky2", id=130, children=[
                            Node("RPinky3", id=131, children=[
                                Node("RPinky4", id=132),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb1", id=92, children=[
                        Node("LThumb", id=93, children=[
                            Node("LThumb3", id=94, children=[
                                Node("LThumb4", id=95),
                            ]),
                        ]),
                    ]),
                    Node("LIndex", id=96, children=[
                        Node("LIndex2", id=97, children=[
                            Node("LIndex3", id=98, children=[
                                Node("LIndex4", id=99),
                            ]),
                        ]),
                    ]),
                    Node("LMiddle1", id=100, children=[
                        Node("LMiddle2", id=101, children=[
                            Node("LMiddle3", id=102, children=[
                                Node("LMiddle4", id=103),
                            ]),
                        ]),
                    ]),
                    Node("LRing1", id=104, children=[
                        Node("LRing2", id=105, children=[
                            Node("LRing3", id=106, children=[
                                Node("LRing4", id=107),
                            ]),
                        ]),
                    ]),
                    Node("LPinky", id=108, children=[
                        Node("LPinky2", id=109, children=[
                            Node("LPinky3", id=110, children=[
                                Node("LPinky4", id=111),
                            ]),
                        ]),
                    ]),
                ])
            ]),
        ]),
        Node("Jaw1", id=23, children=[
            Node("Jaw2", id=24, children=[
                Node("Jaw3", id=25, children=[
                    Node("Jaw4", id=26, children=[
                        Node("Jaw5", id=27, children=[
                            Node("Jaw6", id=28, children=[
                                Node("Jaw7", id=29, children=[
                                    Node("Jaw8", id=30, children=[
                                        Node("Jaw9", id=31, children=[
                                            Node("Jaw10", id=32, children=[
                                                Node("Jaw11", id=33, children=[
                                                    Node("Jaw12", id=34, children=[
                                                        Node("Jaw13", id=35, children=[
                                                            Node("Jaw14", id=36, children=[
                                                                Node("Jaw15", id=37, children=[
                                                                    Node("Jaw16", id=38, children=[
                                                                        Node("Jaw17", id=39),
                                                                    ]),
                                                                ]),
                                                            ]),
                                                        ]),
                                                    ]),
                                                ]),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("EyebrowR1", id=40, children=[
            Node("EyebrowR2", id=41, children=[
                Node("EyebrowR3", id=42, children=[
                    Node("EyebrowR4", id=43, children=[
                        Node("EyebrowR5", id=44, children=[
                            Node("EyebrowL1", id=45, children=[
                                Node("EyebrowL2", id=46, children=[
                                    Node("EyebrowL3", id=47, children=[
                                        Node("EyebrowL4", id=48, children=[
                                            Node("EyebrowL5", id=49)
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("Nose1", id=50, children=[
            Node("Nose2", id=51, children=[
                Node("Nose3", id=52, children=[
                    Node("Nose4", id=53, children=[
                        Node("Nose5", id=54, children=[
                            Node("Nose6", id=55, children=[
                                Node("Nose7", id=56, children=[
                                    Node("Nose8", id=57, children=[
                                        Node("Nose9", id=58)
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("REye1", id=59, children=[
            Node("REye2", id=60, children=[
                Node("REye3", id=61, children=[
                    Node("REye4", id=62, children=[
                        Node("REye5", id=63, children=[
                            Node("REye6", id=64)
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("LEye1", id=65, children=[
            Node("LEye2", id=66, children=[
                Node("LEye3", id=67, children=[
                    Node("LEye4", id=68, children=[
                        Node("LEye5", id=69, children=[
                            Node("LEye6", id=70)
                        ]),
                    ]),
                ]),
            ]),
        ]),
        Node("Mouth1", id=71, children=[
            Node("Mouth2", id=72, children=[
                Node("Mouth3", id=73, children=[
                    Node("Mouth4", id=74, children=[
                        Node("Mouth5", id=75, children=[
                            Node("Mouth6", id=76, children=[
                                Node("Mouth7", id=77, children=[
                                     Node("Mouth8", id=78, children=[
                                        Node("Mouth9", id=79, children=[
                                            Node("Mouth10", id=80, children=[
                                                Node("Mouth11", id=81, children=[
                                                    Node("Mouth12", id=82, children=[
                                                        Node("Mouth13", id=83, children=[
                                                            Node("Mouth14", id=84, children=[
                                                                Node("Mouth15", id=85, children=[
                                                                    Node("Mouth16", id=86, children=[
                                                                        Node("Mouth17", id=87, children=[
                                                                            Node("Mouth18", id=88, children=[
                                                                                Node("Mouth19", id=89, children=[
                                                                                    Node("Mouth20", id=90)
                                                                                ]),
                                                                            ]),
                                                                        ]),
                                                                    ]),
                                                                ]),
                                                            ]),
                                                        ]),
                                                    ]),
                                                ]),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
])


'''COCO_17 (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
COCO_17 = Node("Hip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15),
        ]),
    ]),
    Node("Neck", id=None, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''HAND_21 
https://github.com/jin-s13/COCO-WholeBody/'''
HAND_21 = Node("RWrist", id=0, children=[
    Node("RThumb1", id=1, children=[
        Node("RThumb2", id=2, children=[
            Node("RThumb3", id=3, children=[
                Node("RThumb4", id=4),
            ]),
        ]),
    ]),
    Node("RIndex1", id=5, children=[
        Node("RIndex2", id=6, children=[
            Node("RIndex3", id=7, children=[
                Node("RIndex4", id=8),
            ]),
        ]),
    ]),
    Node("RMiddle1", id=9, children=[
        Node("RMiddle2", id=10, children=[
            Node("RMiddle3", id=11, children=[
                Node("RMiddle4", id=12),
            ]),
        ]),
    ]),
    Node("RRing1", id=13, children=[
        Node("RRing2", id=14, children=[
            Node("RRing3", id=15, children=[
                Node("RRing4", id=16),
            ]),
        ]),
    ]),
    Node("RPinky1", id=17, children=[
        Node("RPinky2", id=18, children=[
            Node("RPinky3", id=19, children=[
                Node("RPinky4", id=20),
            ]),
        ]),
    ]),
])


'''FACE_106
https://github.com/jd-opensource/lapa-dataset'''
FACE_106 = Node("root", id=None, children=[
    Node("Jaw0", id=0, children=[
        Node("Jaw1", id=1, children=[
            Node("Jaw2", id=2, children=[
                Node("Jaw3", id=3, children=[
                    Node("Jaw4", id=4, children=[
                        Node("Jaw5", id=5, children=[
                            Node("Jaw6", id=6, children=[
                                Node("Jaw7", id=7, children=[
                                    Node("Jaw8", id=8, children=[
                                        Node("Jaw9", id=9, children=[
                                            Node("Jaw10", id=10, children=[
                                                Node("Jaw11", id=11, children=[
                                                    Node("Jaw12", id=12, children=[
                                                        Node("Jaw13", id=13, children=[
                                                            Node("Jaw14", id=14, children=[
                                                                Node("Jaw15", id=15, children=[
                                                                    Node("Jaw16", id=16, children=[
                                                                        Node("Jaw17", id=17, children=[
                                                                            Node("Jaw18", id=18, children=[
                                                                                Node("Jaw19", id=19, children=[
                                                                                    Node("Jaw20", id=20, children=[
                                                                                        Node("Jaw21", id=21, children=[
                                                                                            Node("Jaw22", id=22, children=[
                                                                                                Node("Jaw23", id=23, children=[
                                                                                                    Node("Jaw24", id=24, children=[
                                                                                                        Node("Jaw25", id=25, children=[
                                                                                                            Node("Jaw26", id=26, children=[
                                                                                                                Node("Jaw27", id=27, children=[
                                                                                                                    Node("Jaw28", id=28, children=[
                                                                                                                        Node("Jaw29", id=29, children=[
                                                                                                                            Node("Jaw30", id=30, children=[
                                                                                                                                Node("Jaw31", id=31, children=[
                                                                                                                                    Node("Jaw32", id=32),
                                                                                                                                ]),
                                                                                                                            ]),
                                                                                                                        ]),
                                                                                                                    ]),
                                                                                                                ]),
                                                                                                            ]),
                                                                                                        ]),
                                                                                                    ]),
                                                                                                ]),
                                                                                            ]),
                                                                                        ]),
                                                                                    ]),
                                                                                ]),
                                                                            ]),
                                                                        ]),
                                                                    ]),
                                                                ]),
                                                            ]),
                                                        ]),
                                                    ]),
                                                ]),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    Node("EyebrowR1", id=33, children=[
        Node("EyebrowR2", id=34, children=[
            Node("EyebrowR3", id=35, children=[
                Node("EyebrowR4", id=36, children=[
                    Node("EyebrowR5", id=37, children=[
                        Node("EyebrowR6", id=38, children=[
                            Node("EyebrowR7", id=39, children=[
                                Node("EyebrowR8", id=40, children=[
                                    Node("EyebrowR9", id=41)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    Node("EyebrowL1", id=42, children=[
        Node("EyebrowL2", id=43, children=[
            Node("EyebrowL3", id=44, children=[
                Node("EyebrowL4", id=45, children=[
                    Node("EyebrowL5", id=46, children=[
                        Node("EyebrowL6", id=47, children=[
                            Node("EyebrowL7", id=48, children=[
                                Node("EyebrowL8", id=49, children=[
                                    Node("EyebrowL9", id=50)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    Node("Nose1", id=51, children=[
        Node("Nose2", id=52, children=[
            Node("Nose3", id=53, children=[
                Node("Nose4", id=54, children=[
                    Node("Nose5", id=55, children=[
                        Node("Nose6", id=56, children=[
                            Node("Nose7", id=57, children=[
                                Node("Nose8", id=58, children=[
                                    Node("Nose9", id=59)
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    Node("REye1", id=60, children=[
        Node("REye2", id=61, children=[
            Node("REye3", id=62, children=[
                Node("REye4", id=63, children=[
                    Node("REye5", id=64, children=[
                        Node("REye6", id=65, children=[
                            Node("REye7", id=66, children=[
                                Node("REye8", id=67)
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
    Node("LEye1", id=68, children=[
        Node("LEye2", id=69, children=[
            Node("LEye3", id=70, children=[
                Node("LEye4", id=71, children=[
                    Node("LEye5", id=72, children=[
                        Node("LEye6", id=73, children=[
                            Node("LEye7", id=74, children=[
                                Node("LEye8", id=75)
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
        Node("Mouth1", id=76, children=[
            Node("Mouth2", id=77, children=[
                Node("Mouth3", id=78, children=[
                    Node("Mouth4", id=79, children=[
                        Node("Mouth5", id=80, children=[
                            Node("Mouth6", id=81, children=[
                                Node("Mouth7", id=82, children=[
                                        Node("Mouth8", id=83, children=[
                                        Node("Mouth9", id=84, children=[
                                            Node("Mouth10", id=85, children=[
                                                Node("Mouth11", id=86, children=[
                                                    Node("Mouth12", id=87, children=[
                                                        Node("Mouth13", id=88, children=[
                                                            Node("Mouth14", id=89, children=[
                                                                Node("Mouth15", id=90, children=[
                                                                    Node("Mouth16", id=91, children=[
                                                                        Node("Mouth17", id=92, children=[
                                                                            Node("Mouth18", id=93, children=[
                                                                                    Node("Mouth19", id=94, children=[
                                                                                    Node("Mouth20", id=95)
                                                                                ]),
                                                                            ]),
                                                                        ]),
                                                                    ]),
                                                                ]),
                                                            ]),
                                                        ]),
                                                    ]),
                                                ]),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
])

'''ANIMAL2D_17 (full-body animal)
https://github.com/AlexTheBad/AP-10K/'''
ANIMAL2D_17 = Node("Hip", id=4, children=[
    Node("RHip", id=14, children=[
        Node("RKnee", id=15, children=[
            Node("RAnkle", id=16),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=3, children=[
        Node("Nose", id=2, children=[
            Node("REye", id=1),
            Node("LEye", id=0),
        ]),
        Node("RShoulder", id=8, children=[
            Node("RElbow", id=9, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''BODY_25B (full-body without hands, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md'''
BODY_25B = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9),
            ]),
        ]),
    ]),
])


'''BODY_25 (full-body without hands, standard, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
BODY_25 = Node("CHip", id=8, children=[
    Node("RHip", id=9, children=[
        Node("RKnee", id=10, children=[
            Node("RAnkle", id=11, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=12, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=14, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''BODY_135 (full-body with hands and face, experimental, from OpenPose)
https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/blob/master/experimental_models/README.md)'''
BODY_135 = Node("CHip", id=None, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=22, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=24),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=19, children=[
                    Node("LSmallToe", id=20),
                ]),
                Node("LHeel", id=21),
            ]),
        ]),
    ]),
    Node("Neck", id=17, children=[
        Node("Head", id=18, children=[
            Node("Nose", id=0),
        ]),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=48),
                    Node("RIndex", id=51),
                    Node("RPinky", id=63),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=27),
                    Node("LIndex", id=30),
                    Node("LPinky", id=42),
                ]),
            ]),
        ]),
    ]),
])


'''BLAZEPOSE (full-body with simplified hand and foot, from mediapipe)
https://google.github.io/mediapipe/solutions/pose'''
BLAZEPOSE = Node("root", id=None, children=[
    Node("right_hip", id=24, children=[
        Node("right_knee", id=26, children=[
            Node("right_ankle", id=28, children=[
                Node("right_heel", id=30),
                Node("right_foot_index", id=32),
            ]),
        ]),
    ]),
    Node("left_hip", id=23, children=[
        Node("left_knee", id=25, children=[
            Node("left_ankle", id=27, children=[
                Node("left_heel", id=29),
                Node("left_foot_index", id=31),
            ]),
        ]),
    ]),
    Node("nose", id=0, children=[
        Node("right_eye", id=5),
        Node("left_eye", id=2),
    ]),
    Node("right_shoulder", id=12, children=[
        Node("right_elbow", id=14, children=[
            Node("right_wrist", id=16, children=[
                Node("right_pinky", id=18),
                Node("right_index", id=20),
                Node("right_thumb", id=22),
            ]),
        ]),
    ]),
    Node("left_shoulder", id=11, children=[
        Node("left_elbow", id=13, children=[
            Node("left_wrist", id=15, children=[
                Node("left_pinky", id=17),
                Node("left_index", id=19),
                Node("left_thumb", id=21),
            ]),
        ]),
    ]),
])


'''HALPE_68 (full-body with hands without face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_68 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=49),
                    Node("RIndex", id=52),
                    Node("RPinky", id=64),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=28),
                    Node("LIndex", id=31),
                    Node("LPinky", id=43),
                ])
            ]),
        ]),
    ]),
])


'''HALPE_136 (full-body with hands and face, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md'''
HALPE_136 = Node("Hip", id=19, children=[
    Node("RHip", id=12, children=[
        Node("RKnee", id=14, children=[
            Node("RAnkle", id=16, children=[
                Node("RBigToe", id=21, children=[
                    Node("RSmallToe", id=23),
                ]),
                Node("RHeel", id=25),
            ]),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=13, children=[
            Node("LAnkle", id=15, children=[
                Node("LBigToe", id=20, children=[
                    Node("LSmallToe", id=22),
                ]),
                Node("LHeel", id=24),
            ]),
        ]),
    ]),
    Node("Neck", id=18, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=6, children=[
            Node("RElbow", id=8, children=[
                Node("RWrist", id=10, children=[
                    Node("RThumb", id=117),
                    Node("RIndex", id=120),
                    Node("RPinky", id=132),
                ]),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=7, children=[
                Node("LWrist", id=9, children=[
                    Node("LThumb", id=96),
                    Node("LIndex", id=99),
                    Node("LPinky", id=111),
                ])
            ]),
        ]),
    ]),
])


'''COCO (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
COCO = Node("CHip", id=None, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])


'''MPII (full-body without hands and feet, from OpenPose, AlphaPose, OpenPifPaf, YOLO-pose, MMPose, etc.)
https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/models'''
MPII = Node("CHip", id=14, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHip", id=11, children=[
        Node("LKnee", id=12, children=[
            Node("LAnkle", id=13),
        ]),
    ]),
    Node("Neck", id=1, children=[
        Node("Nose", id=0),
        Node("RShoulder", id=2, children=[
            Node("RElbow", id=3, children=[
                Node("RWrist", id=4),
            ]),
        ]),
        Node("LShoulder", id=5, children=[
            Node("LElbow", id=6, children=[
                Node("LWrist", id=7),
            ]),
        ]),
    ]),
])
