#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Reproject 3D points on camera planes         ##
    ##################################################
    
    Reproject 3D points from a trc file to the camera planes determined by a 
    toml calibration file.

    The output 2D points can be chosen to follow the DeepLabCut (default) or 
    the OpenPose format. If OpenPose is chosen, the HALPE_26 model is used, 
    with ear and eye at coordinates (0,0) since they are not used by Pose2Sim. 
    You can change the MODEL tree to a different one if you need to reproject 
    in OpenPose format with a different model than HALPLE_26.

    New: Moving cameras and zooming cameras are now supported.
    
    Usage: 
    from Pose2Sim.Utilities import reproj_from_trc_calib; reproj_from_trc_calib.reproj_from_trc_calib_func(r'<input_trc_file>', r'<input_calib_file>', '<output_format>', r'<output_file_root>')
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file -o
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file -o -u
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file -d -o output_file_root
'''


## INIT
import os
import pandas as pd
import numpy as np
import toml
import cv2
import json
from anytree import Node, RenderTree
from copy import deepcopy
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## SKELETON
'''HALPE_26 (full-body without hands, from AlphaPose, MMPose, etc.)
https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md
https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose'''
MODEL = Node("Hip", id=19, children=[
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


## FUNCTIONS
def computeP(calib_file, undistort=False):
    '''
    Compute projection matrices from toml calibration file.
    Zooming or moving cameras are handled.
    
    INPUT:
    - calib_file: calibration .toml file.
    - undistort: boolean
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    K, R, T, Kh, H = [], [], [], [], []
    P = []
    
    calib = toml.load(calib_file)
    for cam in list(calib.keys()):
        if cam != 'metadata':
            S = np.array(calib[cam]['size'])
            K = np.array(calib[cam]['matrix'])

            if len(K.shape) == 2: # static camera
                if undistort:
                    dist = np.array(calib[cam]['distortions'])
                    optim_K = cv2.getOptimalNewCameraMatrix(K, dist, [int(s) for s in S], 1, [int(s) for s in S])[0]
                    Kh = np.block([optim_K, np.zeros(3).reshape(3,1)])
                else:
                    Kh = np.block([K, np.zeros(3).reshape(3,1)])
            elif len(K.shape) == 3: # zooming camera
                if undistort:
                    dist = np.array(calib[cam]['distortions'])
                    optim_K = [cv2.getOptimalNewCameraMatrix(K[f], dist, [int(s) for s in S], 1, [int(s) for s in S])[0] for f in range(len(K))]
                    Kh = [np.block([optim_K[f], np.zeros(3).reshape(3,1)]) for f in range(len(K))]
                else:
                    Kh = [np.block([K[f], np.zeros(3).reshape(3,1)]) for f in range(len(K))]

            R = np.array(calib[cam]['rotation'])
            T = np.array(calib[cam]['translation'])
            if len(R.shape) == 1: # static camera
                R_mat, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
                H = np.block([[R_mat,T.reshape(3,1)], [np.zeros(3), 1 ]])
            elif len(R.shape) == 2: # moving camera
                R_mat = [cv2.Rodrigues(R[f])[0] for f in range(len(R))]
                H = [np.block([[R_mat[f],T[f].reshape(3,1)], [np.zeros(3), 1 ]]) for f in range(len(R))]
                
            if len(K.shape) == 2 and len(R.shape)==1: # static camera
                P.append([Kh @ H])
            elif len(K.shape) == 3 and len(R.shape)==1: # zooming camera
                P.append([Kh[f] @ H for f in range(len(K))])
            elif len(K.shape) == 2 and len(R.shape)==2: # moving camera
                P.append([Kh @ H[f] for f in range(len(R))])
            elif len(K.shape) == 3 and len(R.shape)==2: # zooming and moving camera
                P.append([Kh[f] @ H[f] for f in range(len(K))])

    return np.array(P)
    
    
def retrieve_calib_params(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    Zooming or moving cameras are handled.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - S: (h,w) vectors as list of 2x1 arrays
    - K: intrinsic matrices as list of 3x3 arrays
    - dist: distortion vectors as list of 4x1 arrays
    - optim_K: intrinsic matrices for undistorting points as list of 3x3 arrays
    - R: rotation rodrigue vectors as list of 3x1 arrays
    - T: translation vectors as list of 3x1 arrays
    '''
    
    calib = toml.load(calib_file)

    S, K, dist, optim_K, R, T = [], [], [], [], [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            S.append(np.array(calib[cam]['size']))
            K.append(np.array(calib[cam]['matrix']))
            dist.append(np.array(calib[cam]['distortions']))

            if len(K[c].shape) == 2: # static camera
                optim_K.append(cv2.getOptimalNewCameraMatrix(K[c], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0])
            elif len(K[c].shape) == 3: # zooming camera
                optim_K.append([cv2.getOptimalNewCameraMatrix(K[c][f], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0] for f in range(len(K[c]))])
            
            R.append(np.array(calib[cam]['rotation']))
            T.append(np.array(calib[cam]['translation']))

    calib_params = {'S': S, 'K': K, 'dist': dist, 'optim_K': optim_K, 'R': R, 'T': T}
            
    return calib_params


def reprojection(P_all, Q):
    '''
    Reprojects 3D point on all cameras.
    
    INPUTS:
    - P_all: list of arrays. Projection matrix for all cameras
    - Q: array of triangulated point (x,y,z,1.)

    OUTPUTS:
    - x_calc, y_calc: list of coordinates of point reprojected on all cameras
    '''
    
    x_calc, y_calc = [], []
    for c in range(len(P_all)):  
        P_cam = P_all[c]
        x_calc.append(P_cam[0] @ Q / (P_cam[2] @ Q))
        y_calc.append(P_cam[1] @ Q / (P_cam[2] @ Q))
        
    return x_calc, y_calc
    

def df_from_trc(trc_path):
    '''
    Retrieve header and data from trc path.
    '''

    # DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
    df_header = pd.read_csv(trc_path, sep="\t", skiprows=1, header=None, nrows=2, encoding="ISO-8859-1")
    header = dict(zip(df_header.iloc[0].tolist(), df_header.iloc[1].tolist()))
    
    # Label1_X  Label1_Y    Label1_Z    Label2_X    Label2_Y
    df_lab = pd.read_csv(trc_path, sep="\t", skiprows=3, nrows=1)
    labels = df_lab.columns.tolist()[2:-1:3]
    labels_XYZ = np.array([[labels[i]+'_X', labels[i]+'_Y', labels[i]+'_Z'] for i in range(len(labels))], dtype='object').flatten()
    labels_FTXYZ = np.concatenate((['Frame#','Time'], labels_XYZ))
    
    data = pd.read_csv(trc_path, sep="\t", skiprows=5, index_col=False, header=None, names=labels_FTXYZ)
    
    return header, data


def yup2zup(Q):
    '''
    Turns Y-up system coordinates into Z-up coordinates

    INPUT:
    - Q: pandas dataframe
    N 3D points as columns, ie 3*N columns in Z-up system coordinates
    and frame number as rows

    OUTPUT:
    - Q: pandas dataframe with N 3D points in Y-up system coordinates
    '''
    
    # X->Y, Y->Z, Z->X
    cols = list(Q.columns)
    cols = np.array([[cols[i*3+2],cols[i*3],cols[i*3+1]] for i in range(int(len(cols)/3))]).flatten()
    Q = Q[cols]

    return Q


def reproj_from_trc_calib_func(**args):
    '''
    Reproject 3D points from a trc file to the camera planes determined by a 
    toml calibration file.
    
    The output 2D points can be chosen to follow the DeepLabCut (default) or 
    the OpenPose format. If OpenPose is chosen, the HALPE_26 model is used, 
    with ear and eye at coordinates (0,0) since they are not used by Pose2Sim. 
    You can change the MODEL tree to a different one if you need to reproject 
    in OpenPose format with a different model than HALPLE_26.

    New: Moving cameras and zooming cameras are now supported.
    
    Usage: 
    from Pose2Sim.Utilities import reproj_from_trc_calib; reproj_from_trc_calib.reproj_from_trc_calib_func(input_trc_file = r'<input_trc_file>', input_calib_file = r'<input_calib_file>', openpose_output=True, deeplabcut_output=True, undistort_points=True, output_file_root = r'<output_file_root>')
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file -o
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file --openpose_output --deeplabcut_output --undistort_points --output_file_root output_file_root
    python -m reproj_from_trc_calib -t input_trc_file -c input_calib_file -o -O output_file_root
    '''

    input_trc_file = os.path.realpath(args.get('input_trc_file')) # invoked with argparse
    input_calib_file = os.path.realpath(args.get('input_calib_file'))
    openpose_output = args.get('openpose_output')
    deeplabcut_output = args.get('deeplabcut_output')
    undistort_points = args.get('undistort_points')
    output_file_root = args.get('output_file_root')
    if output_file_root == None:
        output_file_root = input_trc_file.replace('.trc', '_reproj')
    if os.path.exists(output_file_root):
        os.makedirs(output_file_root, exist_ok=True)
    if not openpose_output and not deeplabcut_output:
        raise ValueError('Output_format must be specified either "openpose_output" (-o) or "deeplabcut_output (-d)"') 

    # Extract data from trc file
    header_trc, data_trc = df_from_trc(input_trc_file)
    data_trc_zup = pd.concat([data_trc.iloc[:,:2], yup2zup(data_trc.iloc[:,2:])], axis=1) # yup to zup system coordinates
    bodyparts = [d[:-2] for d in data_trc_zup.columns[2::3]]
    num_bodyparts = int(header_trc['NumMarkers'])
    filename = os.path.splitext(os.path.basename(input_trc_file))[0]
    
    # Extract data from calibration file
    P_all = computeP(input_calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(input_calib_file)
    calib_params_size = [calib_params['S'][i] for i in range(len(P_all))]
    if undistort_points:
        calib_params_R_filt = [calib_params['R'][i] for i in range(len(P_all))]
        calib_params_T_filt = [calib_params['T'][i] for i in range(len(P_all))]
        calib_params_K_filt = [calib_params['K'][i] for i in range(len(P_all))]
        calib_params_dist_filt = [calib_params['dist'][i] for i in range(len(P_all))]

    # Create camera folders
    reproj_dir = os.path.realpath(output_file_root)
    cam_dirs = [os.path.join(reproj_dir, f'cam{cam+1:02d}_json') for cam in range(len(P_all))]
    if not os.path.exists(reproj_dir): os.mkdir(reproj_dir)  
    try:
        [os.mkdir(cam_dir) for cam_dir in cam_dirs]
    except:
        pass

    # header preparation
    num_frames = [len(data_trc) if P_all.shape[1]==1 else min(P_all.shape[1], len(data_trc))][0]
    columns_iterables = [['DavidPagnon'], ['person0'], bodyparts, ['x','y']]
    columns_h5 = pd.MultiIndex.from_product(columns_iterables, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    rows_iterables = [[os.path.join(os.path.splitext(input_trc_file)[0],f'img_{i:03d}.png') for i in range(num_frames)]]
    rows_h5 = pd.MultiIndex.from_product(rows_iterables)
    data_h5 = pd.DataFrame(np.nan, index=rows_h5, columns=columns_h5)

    # Reproject 3D points on all cameras
    data_proj = [deepcopy(data_h5) for cam in range(len(P_all))] # copy data_h5 as many times as there are cameras
    Q = data_trc_zup.iloc[:,2:]
    for frame in range(num_frames):
        coords = [[] for cam in range(len(P_all))]
        P_all_frame = [P_all[cam][0] if P_all.shape[1]==1 else P_all[cam][frame] for cam in range(len(P_all))]
        for keypoint in range(num_bodyparts):
            q = np.append(Q.iloc[frame,3*keypoint:3*keypoint+3], 1)
            if undistort_points:
                coords_2D_all = [cv2.projectPoints(np.array(q[:-1]), calib_params_R_filt[i], calib_params_T_filt[i], calib_params_K_filt[i], calib_params_dist_filt[i])[0] for i in range(len(P_all))]
                x_all = [coords_2D_all[i][0,0,0] for i in range(len(P_all_frame))]
                y_all = [coords_2D_all[i][0,0,1] for i in range(len(P_all_frame))]
            else:
                x_all, y_all = reprojection(P_all_frame, q)
            [coords[cam].extend([x_all[cam], y_all[cam]]) for cam in range(len(P_all_frame))]
        for cam in range(len(P_all_frame)):
            data_proj[cam].iloc[frame,:] = coords[cam]
    
    # Replace by nan when reprojection out of image
    for cam in range(len(P_all_frame)):
        x_above_size = data_proj[cam].iloc[:,::2] < calib_params_size[cam][0]
        data_proj[cam].iloc[:, ::2] = data_proj[cam].iloc[:, ::2].where(x_above_size, np.nan)
        y_above_size = data_proj[cam].iloc[:,1::2] < calib_params_size[cam][1]
        data_proj[cam].iloc[:, 1::2] = data_proj[cam].iloc[:, 1::2].where(y_above_size, np.nan)


    # Save as h5 and csv if DeepLabCut format
    if deeplabcut_output:
        # to h5
        h5_files = [os.path.join(cam_dir,f'{filename}_cam_{i+1:02d}.h5') for i,cam_dir in enumerate(cam_dirs)]
        [data_proj[i].to_hdf(h5_files[i], index=True, key='reprojected_points') for i in range(len(P_all))]

        # to csv
        csv_files = [os.path.join(cam_dir,f'{filename}_cam_{i+1:02d}.csv') for i,cam_dir in enumerate(cam_dirs)]
        [data_proj[i].to_csv(csv_files[i], sep=',', index=True, lineterminator='\n') for i in range(len(P_all))]

    # Save as json if OpenPose format
    elif openpose_output:
        # read model tree
        model = MODEL
        print('Keypoint hierarchy:')
        for pre, _, node in RenderTree(model): 
            print(f'{pre}{node.name} id={node.id}')
        bodyparts_ids = [[node.id for _, _, node in RenderTree(model) if node.name==b][0] for b in bodyparts]
        nb_joints = len(bodyparts_ids)
        #prepare json files
        json_dict = {'version':1.3, 'people':[]}
        json_dict['people'] = [{'person_id':[-1], 
                        'pose_keypoints_2d': np.zeros(nb_joints*3), 
                        'face_keypoints_2d': [], 
                        'hand_left_keypoints_2d':[], 
                        'hand_right_keypoints_2d':[], 
                        'pose_keypoints_3d':[], 
                        'face_keypoints_3d':[], 
                        'hand_left_keypoints_3d':[], 
                        'hand_right_keypoints_3d':[]}]
        # write one json file per camera and per frame
        for cam, cam_dir in enumerate(cam_dirs):
            for frame in range(len(Q)):
                json_dict_copy = deepcopy(json_dict)
                data_proj_frame = data_proj[cam].iloc[frame]['DavidPagnon']['person0']
                # store 2D keypoints and respect model keypoint order
                for (i,b) in zip(bodyparts_ids, bodyparts):
                    # print(repr(data_proj_frame[b].values))
                    json_dict_copy['people'][0]['pose_keypoints_2d'][[i*3,i*3+1,i*3+2]] = np.append(data_proj_frame[b].values, 1)
                json_dict_copy['people'][0]['pose_keypoints_2d'] = json_dict_copy['people'][0]['pose_keypoints_2d'].tolist()
                # write json file
                json_file = os.path.join(cam_dir, f'{filename}_cam_{cam+1:02d}.{frame:05d}.json')
                with open(json_file, 'w') as js_f:
                    js_f.write(json.dumps(json_dict_copy))
            print('Camera #', cam, 'done.')
            
    # Wrong format
    else:
        raise ValueError('output_format must be either "openpose" or "deeplabcut"')
    
    print(f'Reprojected points saved at {output_file_root}.')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_trc_file', required = True, help='trc 3D coordinates input file path')
    parser.add_argument('-c', '--input_calib_file', required = True, help='toml calibration input file path')
    parser.add_argument('-o', '--openpose_output', required=False, action='store_true', help='output format in the openpose json format')
    parser.add_argument('-d', '--deeplabcut_output', required=False, action='store_true', help='output format in the deeplabcut csv and json formats')
    parser.add_argument('-u', '--undistort_points', required=False, action='store_true', help='takes distortion into account if True')
    parser.add_argument('-O', '--output_file_root', required=False, help='output file root path, without extension')
    args = vars(parser.parse_args())

    reproj_from_trc_calib_func(**args) 
