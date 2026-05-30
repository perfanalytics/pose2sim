#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################################
    ## Convert AlphaPose json file to OpenPose json files ##
    ########################################################
    
    Converts AlphaPose single json file to OpenPose frame-by-frame files.
        
    Usage: 
    AlphaPose_to_OpenPose -i input_alphapose_json_file -o output_openpose_json_folder
    OR AlphaPose_to_OpenPose -i input_alphapose_json_file
    OR from Pose2Sim.Utilities import AlphaPose_to_OpenPose; AlphaPose_to_OpenPose.AlphaPose_to_OpenPose_func(r'input_alphapose_json_file', r'output_openpose_json_folder')
'''


## INIT
import json
import os
import argparse
from importlib.metadata import version
from pathlib import Path


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_alphapose_json_file', required = True, help='input AlphaPose single json file')
    parser.add_argument('-o', '--output_openpose_json_folder', required = False, help='output folder for frame-by-frame OpenPose json files')
    args = vars(parser.parse_args())
    
    AlphaPose_to_OpenPose_func(args)


def AlphaPose_to_OpenPose_func(*args):
    '''
    Converts AlphaPose single json file to OpenPose frame-by-frame files.
        
    Usage: 
    AlphaPose_to_OpenPose -i input_alphapose_json_file -o output_openpose_json_folder
    OR AlphaPose_to_OpenPose -i input_alphapose_json_file
    OR from Pose2Sim.Utilities import AlphaPose_to_OpenPose; AlphaPose_to_OpenPose.AlphaPose_to_OpenPose_func(r'input_alphapose_json_file', r'output_openpose_json_folder')
    '''

    try:
        input_alphapose_json_file = Path(args[0]['input_alphapose_json_file']).resolve() # invoked with argparse
        if args[0]['output_openpose_json_folder'] == None:
            output_openpose_json_folder = Path(input_alphapose_json_file).stem
        else:
            output_openpose_json_folder = Path(args[0]['output_openpose_json_folder']).resolve()
    except:
        input_alphapose_json_file = Path(args[0]).resolve() # invoked as a function
        try:
            output_openpose_json_folder = Path(args[1]).resolve()
        except:
            output_openpose_json_folder = Path(input_alphapose_json_file).stem
        
    if not Path(output_openpose_json_folder).exists():    
        os.mkdir(output_openpose_json_folder)

    # Open AlphaPose json file
    with open(input_alphapose_json_file, 'r') as alpha_json_f:
        alpha_js = json.load(alpha_json_f)
        json_dict = {'version':1.3, 'people':[]}
        coords = []
        frame_next = int(alpha_js[0].get('image_id').split('.')[0])
        for i, a in enumerate(alpha_js):
            frame_prev = int(a.get('image_id').split('.')[0])
            coords = a.get('keypoints')
            if frame_next != frame_prev or i==0:
                # Save openpose json file with all people contained in the previous frame
                if i != 0:
                    json_file = Path(output_openpose_json_folder) / f'{str(frame_prev-1).zfill(5)}.json'
                    with open(json_file, 'w') as js_f:
                        js_f.write(json.dumps(json_dict))
                # Reset json_dict
                json_dict['people'] = [{'person_id':[-1], 
                    'pose_keypoints_2d': [], 
                    'face_keypoints_2d': [], 
                    'hand_left_keypoints_2d':[], 
                    'hand_right_keypoints_2d':[], 
                    'pose_keypoints_3d':[], 
                    'face_keypoints_3d':[], 
                    'hand_left_keypoints_3d':[], 
                    'hand_right_keypoints_3d':[]}]
            else:
                # Add new person to json_dict
                json_dict['people'] += [{'person_id':[-1], 
                    'pose_keypoints_2d': [], 
                    'face_keypoints_2d': [], 
                    'hand_left_keypoints_2d':[], 
                    'hand_right_keypoints_2d':[], 
                    'pose_keypoints_3d':[], 
                    'face_keypoints_3d':[], 
                    'hand_left_keypoints_3d':[], 
                    'hand_right_keypoints_3d':[]}]
            # Add coordinates to json_dict
            json_dict['people'][-1]['pose_keypoints_2d'] = coords

            frame_next = int(a.get('image_id').split('.')[0])
        
        # Save last frame
        json_file = Path(output_openpose_json_folder) / f'{str(frame_prev).zfill(5)}.json'
        with open(json_file, 'w') as js_f:
            js_f.write(json.dumps(json_dict))

        print(f'OpenPose json files saved in {output_openpose_json_folder}')


if __name__ == '__main__':
    main()