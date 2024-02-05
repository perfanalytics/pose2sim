#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ########################################################
    ## Run BlazePose and save coordinates                 ##
    ########################################################
    
    Runs BlazePose (Mediapipe) on a video
    Saves coordinates to OpenPose format (json files) or DeepLabCut format (csv or h5 table)
    Optionally displays and saves images with keypoints overlayed

    N.B.: First install mediapipe: `pip install mediapipe`
    You may also need to install tables: `pip install tables`
        
    Usage: 
    python -m Blazepose_runsave -i input_file --display --save_images --save_video --to_csv --to_h5 --to_json --model_complexity 2 -o output_folder
    OR python -m Blazepose_runsave -i input_file --display --to_json --save_images 
    OR python -m Blazepose_runsave -i input_file -dJs
    OR from Pose2Sim.Utilities import Blazepose_runsave; Blazepose_runsave.blazepose_detec_func(input_file=r'input_file', save_images=True, to_json=True, model_complexity=2)
'''


## INIT
import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import json
import argparse

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.6'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def save_to_csv_or_h5(kpt_list, output_folder, video_name, to_csv, to_h5):
    '''
    Saves blazepose keypoint coordinates to csv or h5 file, 
    in the DeepLabCut format.

    INPUTS:
    - kpt_list: List of lists of keypoints X and Y coordinates and likelihood, for each frame
    - output_folder: Folder where to save the csv or h5 file
    - video_name: Name of the video
    - to_csv: Boolean, whether to save to csv
    - to_h5: Boolean, whether to save to h5

    OUTPUTS:
    - Creation of csv or h5 file in output_folder
    '''
    
    # Prepare dataframe file
    scorer = ['DavidPagnon']*len(mp_pose.PoseLandmark)*3
    individuals = ['person']*len(mp_pose.PoseLandmark)*3
    bodyparts = [[p.name]*3 for p in mp_pose.PoseLandmark]
    bodyparts = [item for sublist in bodyparts for item in sublist]
    coords = ['x', 'y', 'likelihood']*len(mp_pose.PoseLandmark)

    tuples = list(zip(scorer, individuals, bodyparts, coords))
    index_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    df = pd.DataFrame(np.array(kpt_list).T, index=index_csv).T

    if to_csv:
        csv_file = os.path.join(output_folder, video_name+'.csv')
        df.to_csv(csv_file, sep=',', index=True, lineterminator='\n')

    if to_h5:
        h5_file = os.path.join(output_folder, video_name+'.h5')
        df.to_hdf(h5_file, index=True, key='blazepose_detection')


def save_to_json(kpt_list, output_folder, video_name):
    '''
    Saves blazepose keypoint coordinates to json file, 
    in the OpenPose format.

    INPUTS:
    - kpt_list: List of lists of keypoints X and Y coordinates and likelihood, for each frame
    - output_folder: Folder where to save the csv or h5 file
    - video_name: Name of the video

    OUTPUTS:
    - Creation of json files in output_folder/json_folder    
    '''

    json_folder = os.path.join(output_folder, 'blaze_'+video_name + '_json')
    if not os.path.exists(json_folder):
        os.mkdir(json_folder)
    print(json_folder)

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
    
    # write each h5 line in json file
    for frame, kpt in enumerate(kpt_list):
        json_dict['people'][0]['pose_keypoints_2d'] = kpt
        json_file = os.path.join(json_folder, 'blaze_'+video_name+'.'+str(frame).zfill(5)+'.json')
        with open(json_file, 'w') as js_f:
            js_f.write(json.dumps(json_dict))


def blazepose_detec_func(**args):
    '''
    Runs BlazePose (Mediapipe) on a video
    Saves coordinates to OpenPose format (json files) or DeepLabCut format (csv or h5 table)
    Optionally displays and saves images with keypoints overlayed

    N.B.: First install mediapipe: `pip install mediapipe`
    You may also need to install tables: `pip install tables`
        
    Usage: 
    python -m Blazepose_runsave -i input_file --display --save_images --save_video --to_csv --to_h5 --to_json --model_complexity 2 -o output_folder
    OR python -m Blazepose_runsave -i input_file --display --to_json --save_images
    OR python -m Blazepose_runsave -i input_file -dJs
    OR from Pose2Sim.Utilities import Blazepose_runsave; Blazepose_runsave.blazepose_detec_func(input_file=r'input_file', save_images=True, to_json=True, model_complexity=2)
    '''

    # Retrieve arguments
    video_input = os.path.realpath(args.get('input_file'))
    video_dir = os.path.dirname(video_input)
    video_name = os.path.splitext(os.path.basename(video_input))[0]
    output_folder = args.get('output_folder')

    display = args.get('display')
    save_images = args.get('save_images')
    save_video = args.get('save_video')

    to_csv = args.get('to_csv')
    to_h5 = args.get('to_h5')
    to_json = args.get('to_json')
    
    model_complexity = int(args.get('model_complexity'))
    if 'model_complexity' not in vars(): model_complexity=2

    if to_csv or to_h5 or to_json or save_images or save_video:
        if output_folder == None: 
            output_folder = video_dir
        if not os.path.exists(os.path.realpath(output_folder)):
            os.mkdir(os.path.realpath(output_folder))
    
    # Run Blazepose
    cap = cv2.VideoCapture(video_input)
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    kpt_list = []
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=model_complexity) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:  
                # Blazepose detection
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))            
                try:
                    kpt = [[p.x*W, p.y*H, p.visibility] for p in results.pose_landmarks.landmark]
                    kpt = [item for sublist in kpt for item in sublist]  

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                except:
                    print(f'No person detected by BlazePose on frame {count}')
                    kpt=[np.nan]*3*33

                # Display images
                if display: 
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

                # Save images
                if save_images: 
                    images_folder = os.path.join(output_folder, 'blaze_'+video_name + '_img')
                    if not os.path.exists(images_folder):
                        os.mkdir(images_folder)
                    cv2.imwrite(os.path.join(images_folder, 'blaze_'+video_name+'.'+str(count).zfill(5)+'.png'), frame)

                # Save video
                if save_video:
                    if count == 0:
                        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                        writer = cv2.VideoWriter(os.path.join(output_folder, video_name+'_blaze.mp4'), fourcc, fps, (int(W), int(H)))
                    writer.write(frame)

                # Store coordinates
                if to_csv or to_h5 or to_json:
                    kpt_list.append(kpt)

                count += 1
                
            else:
                break

        cap.release()
        if save_video:
            writer.release()    
        cv2.destroyAllWindows()

    # Save coordinates
    if to_csv or to_h5:
        save_to_csv_or_h5(kpt_list, output_folder, video_name, to_csv, to_h5)
   
    if to_json:
        save_to_json(kpt_list, output_folder, video_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='input video file')
    parser.add_argument('-C', '--to_csv', required=False, action='store_true', help='save coordinates to csv')
    parser.add_argument('-H', '--to_h5', required=False, action='store_true', help='save coordinates to h5')
    parser.add_argument('-J', '--to_json', required=False, action='store_true', help='save coordinates to json')
    parser.add_argument('-d', '--display', required = False, action='store_true', help='display images with overlayed coordinates')
    parser.add_argument('-s', '--save_images', required = False, action='store_true', help='save images with overlayed coordinates')
    parser.add_argument('-v', '--save_video', required = False, action='store_true', help='save video with overlayed coordinates')
    parser.add_argument('-m', '--model_complexity', required = False, default = 2, help='model complexity. 0: fastest but less accurate, 2: most accurate but slowest')
    parser.add_argument('-o', '--output_folder', required=False, help='output folder for coordinates and images')
    
    args = vars(parser.parse_args())
    
    blazepose_detec_func(**args)
