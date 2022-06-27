#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## SKELETONS DEFINITIONS                                                 ##
    ###########################################################################
    
    Skeletons definition and hierarchy.
    Openpose BODY_25B, BODY_25, HALPE (AlphaPose), BLAZEPOSE (MediaPipe), COCO and MPI are provided.
    Note that not all face keypoints are reported, since less are needed to determine head orientation.

    You can also build your own custom skeleton. Verify your structure by typing: 
    from anytree import Node, RenderTree
    for pre, _, node in RenderTree(CUSTOM): 
            print(f'{pre}{node.name} id={node.id}')
    
    Note that only Body_25 and Body_25b OpenSim files are provided (in "Empty project" folder).
    If you wish to use any other, you will need to adjust the markerset in the .osim model file, 
    as well as in the scaling and IK setup files.
    
'''

## INIT
from anytree import Node, RenderTree


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## SKELETONS
# CUSTOM SKELETON
CUSTOM = Node("Root", id=0, children=[
    Node("Child1", id=1),
    Node("Child2", id=2),
    ])

# BODY_25B
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

# BODY_25
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
    Node("Neck", id=17, children=[
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

# BLAZEPOSE / MEDIAPIPE https://google.github.io/mediapipe/solutions/pose
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


# HALPE / ALPHAPOSE https://github.com/Fang-Haoshu/Halpe-FullBody
ALPHAPOSE = Node("Hip", id=19, children=[
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


# COCO
COCO = Node("CHip", id=None, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHipJ", id=11, children=[
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

# MPI
MPI = Node("CHip", id=14, children=[
    Node("RHip", id=8, children=[
        Node("RKnee", id=9, children=[
            Node("RAnkle", id=10),
        ]),
    ]),
    Node("LHipJ", id=11, children=[
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

# BODY_135
BODY_135 = Node("")
