#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################################
    ## Convert DeepLabCut h5 files to OpenPose json files ##
    ########################################################
    
    Translates DeepLabCut (h5) 2D pose estimation files into OpenPose (json) files.
    You may need to install tables: 'pip install tables' or 'conda install pytables'
        
    Usage: 
    DLC_to_OpenPose -i input_h5_file -o output_json_folder
    OR DLC_to_OpenPose -i input_h5_file
    OR from Pose2Sim.Utilities import DLC_to_OpenPose; DLC_to_OpenPose.DLC_to_OpenPose_func(r'input_h5_file', r'output_json_folder')
'''


## INIT
import pandas as pd
import numpy as np
import os
import json
import re
import argparse


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


## FUNCTIONS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required = True, help='input 2D pose coordinates DeepLabCut h5 file')
    parser.add_argument('-o', '--output', required = False, help='output folder for 2D pose coordinates OpenPose json files')
    args = vars(parser.parse_args())
    
    DLC_to_OpenPose_func(args)


def DLC_to_OpenPose_func(*args):
    '''
    Translates DeepLabCut (h5) 2D pose estimation files into OpenPose (json) files.

    Usage: 
    DLC_to_OpenPose -i input_h5_file -o output_json_folder
    OR DLC_to_OpenPose -i input_h5_file
    OR import DLC_to_OpenPose; DLC_to_OpenPose.DLC_to_OpenPose_func(r'input_h5_file', r'output_json_folder')
    '''

    try:
        h5_file_path = os.path.realpath(args[0]['input']) # invoked with argparse
        if args[0]['output'] == None:
            json_folder_path = os.path.splitext(h5_file_path)[0]
        else:
            json_folder_path = os.path.realpath(args[0]['output'])
    except:
        h5_file_path = os.path.realpath(args[0]) # invoked as a function
        try:
            json_folder_path = os.path.realpath(args[1])
        except:
            json_folder_path = os.path.splitext(h5_file_path)[0]
        
    if not os.path.exists(json_folder_path):    
        os.mkdir(json_folder_path)

    # json preparation
    json_dict = {'version':1.3, 'people':[]}
    json_dict['people'] = [{'person_id':[-1], 
                    'pose_keypoints_2d': [], 
                    'face_keypoints_2d': [], 
                    'hand_left_keypoints_2d':[], 
                    'hand_right_keypoints_2d':[], 
                    'pose_keypoints_3d':[], 
                    'face_keypoints_3d':[], 
                    'hand_left_keypoints_3d':[], 
                    'hand_right_keypoints_3d':[]}]
    
    # h5 reader
    h5_file = pd.read_hdf(h5_file_path).fillna(0)
    kpt_nb = int(len(h5_file.columns)//3)
    
    # write each h5 line in json file
    for f, frame in enumerate(h5_file.index):
        h5_line = np.array([[h5_file.iloc[f, 3*k], h5_file.iloc[f, 3*k+1], h5_file.iloc[f, 3*k+2]] for k in range(kpt_nb)]).flatten().tolist()
        json_dict['people'][0]['pose_keypoints_2d'] = h5_line
        json_file = os.path.join(json_folder_path, os.path.splitext(os.path.basename(str(frame).zfill(5)))[0]+'.json')
        with open(json_file, 'w') as js_f:
            js_f.write(json.dumps(json_dict))

    print(f"DeepLabCut h5 files converted to OpenPose json files in {json_folder_path}")


if __name__ == '__main__':
    main()
