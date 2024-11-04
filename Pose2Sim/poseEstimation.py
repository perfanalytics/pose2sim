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
import cv2

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from rtmlib import draw_skeleton
from Pose2Sim.common import natural_sort_key
from Sports2D.Utilities.config import setup_pose_tracker, setup_video_capture, setup_capture_directories, process_video_frames
from Sports2D.Utilities.video_management import display_realtime_results, finalize_video_processing, track_people
from Sports2D.Utilities.utilities import read_frame


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

    # Read config
    project_dir = config_dict['project']['project_dir']
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    frame_range = config_dict.get('project').get('frame_range')
    output_dir = config_dict.get('project').get('project_dir')
    multi_person = config_dict.get('project').get('multi_person')
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    vid_img_extension = config_dict['pose']['vid_img_extension']
    
    webcam_id =  config_dict.get('pose').get('webcam_ids')
    
    overwrite_pose = config_dict['pose']['overwrite_pose']

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
            if isinstance(webcam_id, list):
                video_paths = [f'webcam{cam_id}' for cam_id in webcam_id]
            else:
                video_paths = [f'webcam{webcam_id}']
            frame_ranges = [None] * len(video_paths)
        else:
            video_paths = [f for f in Path(video_dir).rglob('*' + vid_img_extension) if f.is_file()]
            frame_ranges = process_video_frames(config_dict, video_paths)

        if video_paths:
            logging.info(f'Found video files/webcams with extension {vid_img_extension}.')
            logging.info(f'Multi-person is {"" if multi_person else "not "}selected.')

            pose_tracker = setup_pose_tracker(
                config_dict['pose']['det_frequency'],
                config_dict['pose']['mode'],
                config_dict['pose']['pose_model']
            )

            def process_video_thread(args):
                process_video(pose_tracker, *args)

            process_args = []
            for idx, (video_path, frame_range) in enumerate(zip(video_paths, frame_ranges)):
                position = idx
                process_args.append((config_dict, video_path, frame_range, output_dir, position))

            num_threads = min(len(video_paths), os.cpu_count())
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_video_thread, arg) for arg in process_args]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logging.error(f"Error processing video: {e}")

        else:
            image_folders = [f for f in Path(video_dir).iterdir() if f.is_dir()]
            if image_folders:
                logging.info('Found image folders.')
                for image_folder in image_folders:
                    image_folder_path = str(image_folder)
                    video_files_in_folder = list(image_folder.glob('*' + vid_img_extension))
                    if video_files_in_folder:
                        raise NameError(f"{video_files_in_folder[0]} is not an image. Videos must be put in the video directory, not in subdirectories.")
                    else:
                        process_images(config_dict, image_folder_path, frame_range)
            else:
                raise FileNotFoundError(f'No video files or image folders found in {video_dir}.')

def process_video(pose_tracker, config_dict, video_file_path, input_frame_range, output_dir, position):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_file_path: str. Path to the input video file
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
    input_size = config_dict.get('pose').get('input_size')

    save_video = True if 'to_video' in config_dict['project']['save_video'] else False
    save_images = True if 'to_images' in config_dict['project']['save_video'] else False
    
    multi_person = config_dict.get('project').get('multi_person')
    show_realtime_results = config_dict['pose'].get('show_realtime_results')
    if show_realtime_results is None:
        show_realtime_results = config_dict['pose'].get('display_detection')
        if show_realtime_results is not None:
            print("Warning: 'display_detection' is deprecated. Please use 'show_realtime_results' instead.")
    
    output_format = config_dict['pose']['output_format']

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(video_file_path, output_dir, save_images)

    cap, frame_iterator, out_vid, cam_width, cam_height, fps = setup_video_capture(
        video_file_path, save_video, output_video_path, input_size, input_frame_range, position
    )

    # Call to display real-time results if needed
    if show_realtime_results:
        display_realtime_results(video_file_path)
    
    if video_file_path == 'webcam' and save_video:
        total_processing_start_time = datetime.now()

    frames_processed = 0
    prev_keypoints = None
    for frame_idx in frame_iterator:
        frame = read_frame(cap, frame_idx)

        # If frame not grabbed
        if frame is None:
            logging.warning(f"Failed to grab frame {frame_idx}.")
            continue

        # Perform pose estimation on the frame
        keypoints, scores = pose_tracker(frame)

        # Tracking people IDs across frames
        keypoints, scores, prev_keypoints = track_people(
            keypoints, scores, multi_person, None, prev_keypoints, pose_tracker
        )

        if 'openpose' in output_format:
            json_file_path = os.path.join(json_output_dir, f'{output_dir_name}_{frame_idx:06d}.json')
            save_to_openpose(json_file_path, keypoints, scores)

        # Draw skeleton on the frame
        if show_realtime_results or save_video or save_images:
            img_show = frame.copy()
            img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low

        if show_realtime_results:
            cv2.imshow(f"Pose Estimation {os.path.basename(video_file_path)}", img_show)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        # Sauvegarde de la vidéo et des images
        if save_video:
            out_vid.write(img_show)
        if save_images:
            cv2.imwrite(
                os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'),
                img_show
            )

        frames_processed += 1

    cap.release()

    if save_video:
        out_vid.release()
        if video_file_path == 'webcam' and frames_processed > 0:
            fps = finalize_video_processing(frames_processed, total_processing_start_time, output_video_path, fps)
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if show_realtime_results:
        cv2.destroyAllWindows()

def process_images(config_dict, image_folder_path, input_frame_range, output_dir):
    '''
    Estimate pose estimation from a folder of images
    
    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - vid_img_extension: str. Extension of the image files
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

    save_video = True if 'to_video' in config_dict['project']['save_video'] else False
    save_images = True if 'to_images' in config_dict['project']['save_video'] else False
    
    multi_person = config_dict.get('project').get('multi_person')
    show_realtime_results = config_dict['pose'].get('show_realtime_results')
    if show_realtime_results is None:
        show_realtime_results = config_dict['pose'].get('display_detection')
        if show_realtime_results is not None:
            print("Warning: 'display_detection' is deprecated. Please use 'show_realtime_results' instead.")

    output_format = config_dict['pose']['output_format']
     
    vid_img_extension = config_dict['pose']['vid_img_extension']

    pose_model = config_dict['pose']['pose_model']
    mode = config_dict['pose']['mode'] # lightweight, balanced, performance
    det_frequency = config_dict['pose']['det_frequency']

    pose_tracker = setup_pose_tracker(det_frequency, mode, pose_model)

    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = setup_capture_directories(image_folder_path, output_dir)

    image_files = glob.glob(os.path.join(image_folder_path, '*' + vid_img_extension))
    sorted(image_files, key=natural_sort_key)

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        W, H = cv2.imread(image_files[0]).shape[:2][::-1] # Get the width and height from the first image (assuming all images have the same size)
        cap = cv2.VideoWriter(output_video_path, fourcc, 60, (W, H)) # Create the output video file

    if show_realtime_results:
        display_realtime_results(image_folder_path)
    
    frame_range = [[len(image_files)] if input_frame_range==[] else input_frame_range][0]
    prev_keypoints = None
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*frame_range):
            frame = cv2.imread(image_file)

            # Perform pose estimation on the image
            keypoints, scores = pose_tracker(frame)

            # Tracking people IDs across frames
            keypoints, scores, prev_keypoints = track_people(
                keypoints, scores, multi_person, None, prev_keypoints, pose_tracker
            )
            
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