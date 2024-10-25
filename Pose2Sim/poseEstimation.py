#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE ESTIMATION                                                       ##
###########################################################################

    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
'''


## INIT
import os
import glob
import json
import logging
import itertools as it
from tqdm import tqdm
import numpy as np
import cv2
from pathlib import Path
import subprocess
import imageio_ffmpeg as ffmpeg
from datetime import datetime
import sys

from rtmlib import PoseTracker, Body, Wholebody, BodyWithFeet, draw_skeleton
from Pose2Sim.common import natural_sort_key, min_with_single_indices, euclidean_distance


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def save_to_openpose(json_file_path, keypoints, scores):
    '''
    Save the keypoints and scores to a JSON file in the OpenPose format

    INPUTS:
    - json_file_path: Path to save the JSON file
    - keypoints: Detected keypoints
    - scores: Confidence scores for each keypoint

    OUTPUTS:
    - JSON file with the detected keypoints and confidence scores in the OpenPose format
    '''

    # Prepare keypoints with confidence scores for JSON output
    nb_detections = len(keypoints)
    # print('results: ', keypoints, scores)
    detections = []
    for i in range(nb_detections): # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
        detections.append({
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_with_confidence_i,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                })
            
    # Create JSON output structure
    json_output = {"version": 1.3, "people": detections}
    
    # Save JSON output for each frame
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir): os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)

   
def sort_people_sports2d(keyptpre, keypt, scores):
    '''
    Associate persons across frames (Pose2Sim method)
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    N.B.: Requires min_with_single_indices and euclidian_distance function (see common.py)

    INPUTS:
    - keyptpre: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates
    - keypt: idem keyptpre, for current frame
    - score: array of shape K, L with K the number of detected persons,
    L the confidence of detected keypoints
    
    OUTPUTS:
    - sorted_prev_keypoints: array with reordered persons with values of previous frame if current is empty
    - sorted_keypoints: array with reordered persons
    - sorted_scores: array with reordered scores
    '''
    
    # Generate possible person correspondences across frames
    if len(keyptpre) < len(keypt):
        keyptpre = np.concatenate((keyptpre, np.full((len(keypt)-len(keyptpre), keypt.shape[1], 2), np.nan)))
    if len(keypt) < len(keyptpre):
        keypt = np.concatenate((keypt, np.full((len(keyptpre)-len(keypt), keypt.shape[1], 2), np.nan)))
        scores = np.concatenate((scores, np.full((len(keyptpre)-len(scores), scores.shape[1]), np.nan)))
    personsIDs_comb = sorted(list(it.product(range(len(keyptpre)), range(len(keypt)))))
    
    # Compute distance between persons from one frame to another
    frame_by_frame_dist = []
    for comb in personsIDs_comb:
        frame_by_frame_dist += [euclidean_distance(keyptpre[comb[0]],keypt[comb[1]])]
    frame_by_frame_dist = np.mean(frame_by_frame_dist, axis=1)
    
    # Sort correspondences by distance
    _, _, associated_tuples = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    
    # Associate points to same index across frames, nan if no correspondence
    sorted_keypoints, sorted_scores = [], []
    for i in range(len(keyptpre)):
        id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
        if len(id_in_old) > 0:
            sorted_keypoints += [keypt[id_in_old[0]]]
            sorted_scores += [scores[id_in_old[0]]]
        else:
            sorted_keypoints += [keypt[i]]
            sorted_scores += [scores[i]]
    sorted_keypoints, sorted_scores = np.array(sorted_keypoints), np.array(sorted_scores)

    # Keep track of previous values even when missing for more than one frame
    sorted_prev_keypoints = np.where(np.isnan(sorted_keypoints) & ~np.isnan(keyptpre), keyptpre, sorted_keypoints)
    
    return sorted_prev_keypoints, sorted_keypoints, sorted_scores


def process_video(config_dict, video_file_path, pose_tracker, input_frame_range, output_dir):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_file_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - show_realtime_results: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    from Sports2D.Utilities.common import setup_webcam, setup_video, setup_capture_directories

    save_video = True if 'to_video' in config_dict['project']['save_video'] else False
    save_images = True if 'to_images' in config_dict['project']['save_video'] else False
    multi_person = config_dict.get('project').get('multi_person')

    output_format = config_dict['pose']['output_format']
    show_realtime_results = config_dict['pose'].get('show_realtime_results')

    if show_realtime_results is None:
        show_realtime_results = config_dict['pose'].get('display_detection')
        if show_realtime_results is not None:
            print("Warning: 'display_detection' is deprecated. Please use 'show_realtime_results' instead.")

    try:
        cap = cv2.VideoCapture(video_file_path)
        cap.read()
        if cap.read()[0] == False:
            raise
    except:
        raise NameError(f"{video_file_path} is not a video. Images must be put in one subdirectory per camera.")
    
    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(video_file_path, output_dir)

    # Set up video capture
    if video_file_path == "webcam":
        # cap, out_vid, cam_width, cam_height, fps = setup_webcam(webcam_id, save_video, output_video_path, input_size)
        frame_range = [0,sys.maxsize]
        frame_iterator = range(*frame_range)
        logging.warning('Webcam input: the framerate may vary. If results are filtered, Sports2D will use the average framerate as input.')
    else:
        cap, out_vid, cam_width, cam_height, fps = setup_video(video_file_path, save_video, output_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_range = [[total_frames] if input_frame_range==[] else input_frame_range][0]
        frame_iterator = tqdm(range(*frame_range), desc=f'Processing {video_file_path}') # use a progress bar
    
    if show_realtime_results:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_file_path)}", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    frame_processing_times = []
    with frame_iterator as pbar:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()

            if frame_idx > frame_range[1] - 1:
                break

            # If frame not grabbed
            if not success:
                logging.warning(f"Failed to grab frame {frame_idx}.")
                continue
            
            if frame_idx in range(*frame_range):
                start_time = datetime.now()

                # Perform pose estimation on the frame
                keypoints, scores = pose_tracker(frame)

                # Tracking people IDs across frames
                if multi_person:
                    if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores)

                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{output_dir_name}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton on the frame
                if show_realtime_results or save_video or save_images:
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                
                if show_realtime_results:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_file_path)}", img_show)
                    if (cv2.waitKey(1) & 0xFF) == ord('q') or (cv2.waitKey(1) & 0xFF) == 27:
                        break

                if save_video:
                    out_vid.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'), img_show)

                if video_file_path == 'webcam' and save_video:   # To adjust framerate of output video
                    elapsed_time = (datetime.now() - start_time).total_seconds()
                    frame_processing_times.append(elapsed_time)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if save_video:
        out_vid.release()
        if video_file_path == 'webcam':
            actual_framerate = len(frame_processing_times) / sum(frame_processing_times)
            logging.info(f"Rewriting webcam video based on the average framerate {actual_framerate}.")
            resample_video(output_video_path, fps, actual_framerate)
            fps = actual_framerate
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if show_realtime_results:
        cv2.destroyAllWindows()

def resample_video(vid_output_path, fps, desired_framerate):
    '''
    Resample video to the desired fps using ffmpeg.
    '''
   
    ffmpeg_path = ffmpeg.get_ffmpeg_exe()
    new_vid_path = vid_output_path.parent / Path(vid_output_path.stem+'_2'+vid_output_path.suffix)
    subprocess.run([ffmpeg_path, '-i', vid_output_path, '-filter:v', f'setpts={fps/desired_framerate}*PTS', '-r', str(desired_framerate), new_vid_path])
    vid_output_path.unlink()
    new_vid_path.rename(vid_output_path)


def process_images(config_dict, image_folder_path, pose_tracker, input_frame_range, output_dir):
    '''
    Estimate pose estimation from a folder of images
    
    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - vid_img_extension: str. Extension of the image files
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - show_realtime_results: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    from Sports2D.Utilities.common import setup_capture_directories

    save_video = True if 'to_video' in config_dict['project']['save_video'] else False
    save_images = True if 'to_images' in config_dict['project']['save_video'] else False
    multi_person = config_dict.get('project').get('multi_person')
    frame_range = config_dict.get('project').get('frame_range')

    output_format = config_dict['pose']['output_format']
    show_realtime_results = config_dict['pose'].get('show_realtime_results')

    if show_realtime_results is None:
        show_realtime_results = config_dict['pose'].get('display_detection')
        if show_realtime_results is not None:
            print("Warning: 'display_detection' is deprecated. Please use 'show_realtime_results' instead.")
    vid_img_extension = config_dict['pose']['vid_img_extension']

    image_file_stem = image_folder_path.stem
    output_dir_name = f'{image_file_stem}_Sports2D'
    output_dir = os.path.abspath(os.path.join(output_dir, 'pose'))
    if not os.path.isdir(output_dir): os.makedirs(output_dir)
    img_output_dir = os.path.join(output_dir, f'{output_dir_name}_img')
    json_output_dir = os.path.join(output_dir, f'{output_dir_name}_json')
    output_video_path = os.path.join(output_dir, f'{output_dir_name}_pose.mp4')

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(image_folder_path, output_dir)


    image_files = glob.glob(os.path.join(image_folder_path, '*' + vid_img_extension))
    sorted(image_files, key=natural_sort_key)

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        W, H = cv2.imread(image_files[0]).shape[:2][::-1] # Get the width and height from the first image (assuming all images have the same size)
        cap = cv2.VideoWriter(output_video_path, fourcc, 60, (W, H)) # Create the output video file

    if show_realtime_results:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
    
    f_range = [[len(image_files)] if frame_range==[] else frame_range][0]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            try:
                frame = cv2.imread(image_file)
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            # Perform pose estimation on the image
            keypoints, scores = pose_tracker(frame)

            # Tracking people IDs across frames
            if multi_person:
                if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores)
            
            # Extract frame number from the filename
            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            # Draw skeleton on the image
            if show_realtime_results or save_video or save_images:
                img_show = frame.copy()
                img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low

            if show_realtime_results:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                cap.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if show_realtime_results:
        cv2.destroyAllWindows()


def rtm_estimator(config_dict):
    '''
    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
    '''
  
    from Sports2D.Utilities.common import setup_pose_tracker

    # Read config
    project_dir = config_dict['project']['project_dir']
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    frame_range = config_dict.get('project').get('frame_range')
    output_dir = config_dict.get('project').get('project_dir')
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    pose_model = config_dict['pose']['pose_model']
    mode = config_dict['pose']['mode'] # lightweight, balanced, performance
    vid_img_extension = config_dict['pose']['vid_img_extension']
    
    overwrite_pose = config_dict['pose']['overwrite_pose']
    det_frequency = config_dict['pose']['det_frequency']

    pose_tracker = setup_pose_tracker(det_frequency, mode, pose_model)

    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        if not overwrite_pose:
            logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
        else:
            logging.info('Overwriting previous pose estimation. Set overwrite_pose to false in Config.toml if you want to keep the previous results.')
            raise
            
    except:
        logging.info('\nEstimating pose...')
        if vid_img_extension == 'webcam':
            video_files = ['webcam']
        else:
            video_files = glob.glob(os.path.join(video_dir, '*' + vid_img_extension))
        if not len(video_files) == 0:
            # Process video files
            logging.info(f'Found video files with extension {vid_img_extension}.')
            for video_file_path in video_files:
                pose_tracker.reset()
                logging.info(f'Video files {video_file_path}.')
                # process_fun(config_dict, video_file_path, pose_tracker, frame_range, output_dir)
                process_video(config_dict, video_file_path, pose_tracker, frame_range, output_dir)

        else:
            # Process image folders
            logging.info(f'Found image folders with extension {vid_img_extension}.')
            image_folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
            for image_folder in image_folders:
                pose_tracker.reset()
                image_folder_path = os.path.join(video_dir, image_folder)
                process_images(config_dict, image_folder_path, pose_tracker, frame_range)
