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
import time
import threading
import queue
import concurrent.futures
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from queue import Queue
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

    show_realtime_results = config_dict['pose'].get('show_realtime_results')

    vid_img_extension = config_dict['pose']['vid_img_extension']
    
    webcam_ids = config_dict.get('pose').get('webcam_ids')
    
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
            if isinstance(webcam_ids, list):
                video_paths = [f'webcam{cam_id}' for cam_id in webcam_ids]
            else:
                video_paths = [f'webcam{webcam_ids}']
            frame_ranges = [None] * len(video_paths)
        else :
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

            if vid_img_extension == 'webcam':
                process_synchronized_webcams(config_dict, webcam_ids, pose_tracker, output_dir)
            else:
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

                if show_realtime_results:
                    cv2.destroyAllWindows()

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

    vid_img_extension = config_dict['pose']['vid_img_extension']
    
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
    
    if vid_img_extension == 'webcam' and save_video:
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

    if hasattr(cap, 'stop'):
        cap.stop()
    if hasattr(cap, 'release'):
        cap.release()

    if save_video:
        out_vid.release()
        if vid_img_extension == 'webcam' and frames_processed > 0:
            fps = finalize_video_processing(frames_processed, total_processing_start_time, output_video_path, fps)
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")

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

    pose_tracker = setup_pose_tracker(
        config_dict['pose']['det_frequency'],
        config_dict['pose']['mode'],
        config_dict['pose']['pose_model']
    )

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


def process_synchronized_webcams(config_dict, webcam_ids, pose_tracker, output_dir):
    '''
    Processes multiple webcams by synchronizing frames between them,
    then processing the frames asynchronously and multithreaded.
    Combines the frames into a single image and displays them in one window.

    Args:
        config_dict (dict): Configuration dictionary.
        webcam_ids (list): List of webcam IDs.
        pose_tracker: Pose tracker object.
        output_dir (str): Output directory path.
    '''
    input_size = config_dict['pose'].get('input_size', (640, 480))
    save_video = 'to_video' in config_dict['project'].get('save_video', [])
    save_images = 'to_images' in config_dict['project'].get('save_video', [])
    multi_person = config_dict['project'].get('multi_person')
    show_realtime_results = config_dict['pose'].get('show_realtime_results', False)
    output_format = config_dict['pose'].get('output_format', 'openpose')

    # Create synchronized webcam streams with consistent mapping
    webcam_streams = SynchronizedWebcamStreams(webcam_ids, input_size, save_video, output_dir)
    outputs = {id: setup_capture_directories(f'webcam{id}', output_dir, save_images) for id in webcam_ids}

    frame_idx = 0
    total_processing_start_time = datetime.now()
    frames_processed = 0

    # Create a lock for each output video to ensure thread-safe access
    out_vid_locks = {webcam_id: threading.Lock() for webcam_id in webcam_ids}

    # Create ThreadPoolExecutor for asynchronous frame processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(webcam_ids) * 2) as executor:
        # Create display thread
        display_thread = CombinedDisplayThread(webcam_ids, input_size)
        display_thread.start()

        try:
            while not webcam_streams.stopped:
                # Read synchronized frames with consistent mapping
                frames = webcam_streams.read()
                if frames is None:
                    break
                
                futures = []
                processed_frames = {}

                for id, frame in frames.items():
                    if frame is not None:
                        future = executor.submit(process_single_frame, config_dict, frame, id, frame_idx, outputs[id], pose_tracker,
                                                 multi_person, save_video, save_images, show_realtime_results,
                                                 webcam_streams.out_videos.get(id), output_format, out_vid_locks[id])
                        futures.append(future)
                    else:
                        img_placeholder = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
                        cv2.putText(img_placeholder, f'Webcam {id} Disconnected', (50, input_size[1] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        processed_frames[id] = img_placeholder

                # Collect processed frames
                for future in concurrent.futures.as_completed(futures):
                    id, img = future.result()
                    processed_frames[id] = img

                # Combine frames into a single image
                combined_image = display_thread.combine_frames(processed_frames)

                # Display the combined image
                if show_realtime_results and combined_image is not None:
                    display_thread.display(combined_image)

                frames_processed += 1
                frame_idx += 1
                if display_thread.stopped:
                    break

        except KeyboardInterrupt:
            logging.info("Processing interrupted by user.")
        finally:
            webcam_streams.stop()
            if save_video:
                for out_vid in webcam_streams.out_videos.values():
                    if out_vid is not None:
                        out_vid.release()
            display_thread.stop()
            display_thread.join()
            if frames_processed > 0:
                total_time = (datetime.now() - total_processing_start_time).total_seconds():.2f} FPS.")

def process_single_frame(config_dict, frame, webcam_id, frame_idx, output_dirs, pose_tracker, multi_person, save_video, save_images, show_realtime_results, out_vid, output_format, out_vid_lock):
    '''
    Processes a single frame from a webcam.

    Args:
        config_dict (dict): Configuration dictionary.
        frame (ndarray): Frame image.
        webcam_id (int): Webcam ID.
        frame_idx (int): Frame index.
        output_dirs (tuple): Output directories.
        pose_tracker: Pose tracker object.
        multi_person (bool): Whether to track multiple persons.
        output_format (str): Output format.
        save_video (bool): Whether to save the output video.
        save_images (bool): Whether to save output images.
        show_realtime_results (bool): Whether to display results in real time.
        out_vid (cv2.VideoWriter): Video writer object.
        out_vid_lock (threading.Lock): Lock for video writer.

    Returns:
        tuple: (webcam_id, img_show)
    '''
    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = output_dirs

    # Perform pose estimation on the frame
    with out_vid_lock: 
        keypoints, scores = pose_tracker(frame)

    # Tracking people IDs across frames (if needed)
    keypoints, scores, _ = track_people(
        keypoints, scores, multi_person, None, None, pose_tracker
    )

    if 'openpose' in output_format:
        json_file_path = os.path.join(json_output_dir, f'{output_dir_name}_{frame_idx:06d}.json')
        save_to_openpose(json_file_path, keypoints, scores)

    # Draw skeleton on the frame
    img_show = frame.copy()
    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1)

    # Save video and images
    if save_video:
        if out_vid is not None:
            with out_vid_lock:
                out_vid.write(img_show)
    if save_images:
        cv2.imwrite(
            os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'),
            img_show
        )

    return webcam_id, img_show

class CombinedDisplayThread(threading.Thread):
    '''
    Thread for displaying combined images to avoid thread-safety issues with OpenCV.
    '''
    def __init__(self, sources, input_size):
        super().__init__()
        self.display_queue = Queue()
        self.stopped = False
        self.daemon = True  # Thread will exit when main program exits
        self.sources = sources
        self.input_size = input_size
        self.window_name = "Combined Webcam Feeds"
        self.grid_size = self.calculate_grid_size(len(sources))
        self.black_frame = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)  # Create a black frame only once

    def run(self):
        try:
            while not self.stopped:
                try:
                    combined_image = self.display_queue.get(timeout=0.1)
                    cv2.imshow(self.window_name, combined_image)
                    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                        logging.info("Display window closed by user.")
                        self.stopped = True
                        break
                except queue.Empty:
                    continue
        finally:
            cv2.destroyAllWindows()

    def display(self, combined_image):
        self.display_queue.put(combined_image)

    def combine_frames(self, processed_frames):
        frames_list = [processed_frames.get(source, self.black_frame) for source in self.sources]
        resized_frames = [cv2.resize(frame, self.input_size) for frame in frames_list if frame is not None]
        rows = []
        for i in range(0, len(resized_frames), self.grid_size[1]):
            row_frames = resized_frames[i:i + self.grid_size[1]]
            row_frames.extend([self.black_frame] * (self.grid_size[1] - len(row_frames)))
            rows.append(np.hstack(row_frames))
        return np.vstack(rows)

    def calculate_grid_size(self, num_cams):
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))
        return (rows, cols)

    def stop(self):
        self.stopped = True

class SynchronizedWebcamStreams:
    def __init__(self, webcam_ids, input_size=(640, 480), save_video=False, output_dir=None):
        self.input_size = input_size
        self.stopped = False
        self.webcam_ids = webcam_ids
        self.streams = {}
        self.queues = {webcam_id: queue.Queue(maxsize=1) for webcam_id in webcam_ids}
        self.identifying_infos = {}
        self.out_videos = {}

        # Initialize streams and queues
        for webcam_id in webcam_ids:
            stream = self.init_stream(webcam_id)
            if stream:
                self.streams[webcam_id] = stream
                self.identifying_infos[webcam_id] = stream.identifying_info
                time.sleep(2)
                threading.Thread(target=self.update, args=(webcam_id,), daemon=True).start()
                if save_video and output_dir:
                    for webcam_id in list(self.streams.keys()):
                        stream = self.streams[webcam_id]
                        video_file_path = f'webcam{webcam_id}'
                        _, _, _, _, output_video_path = setup_capture_directories(video_file_path, output_dir, False)
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        out_vid = cv2.VideoWriter(output_video_path, fourcc, stream.fps, (self.input_size[0], self.input_size[1]))
                        self.out_videos[webcam_id] = out_vid
                else:
                    self.out_videos = {webcam_id: None for webcam_id in webcam_ids}

    def init_stream(self, webcam_id):
        stream = WebcamStream(int(webcam_id), self.input_size)
        if stream.identifying_info is None:
            logging.warning(f"Failed to initialize webcam {webcam_id}")
            return None
        return stream

    def update(self, webcam_id):
        stream = self.streams.get(webcam_id)
        while True:
            frame = stream.read()
            if frame is not None:
                try:
                    self.queues[webcam_id].put(frame, timeout=0.1)
                except queue.Full:
                    continue
            else:
                logging.warning(f"Webcam {webcam_id} disconnected. Attempting to re-identify and reconnect...")
                time.sleep(1)  # Pause before reconnecting
                self.reidentify_webcams()

    def reidentify_webcams(self):
        available_ids = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_ids.append(i)
            else:
                cap.release()
                break 

        for physical_id in available_ids:
            cap = cv2.VideoCapture(physical_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                                        [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    for webcam_id, stored_hist in self.identifying_infos.items():
                        if self.compare_histograms(hist, stored_hist):
                            logging.info(f"Webcam {webcam_id} reidentified as physical ID {physical_id}")
                            self.streams[webcam_id] = WebcamStream(physical_id, self.input_size)
                            time.sleep(2)
                            threading.Thread(target=self.update, args=(webcam_id,), daemon=True).start()
                            break
                cap.release()

    def compare_histograms(self, hist1, hist2, threshold=0.9):
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation >= threshold

    def read(self):
        frames = {webcam_id: self.queues[webcam_id].get(timeout=1) if webcam_id in self.queues else None
                  for webcam_id in self.webcam_ids}
        if any(frame is not None for frame in frames.values()):
            return frames
        else:
            return None

    def stop(self):
        self.stopped = True
        for stream in self.streams.values():
            stream.stop()
        logging.info("All webcams have been stopped.")


class WebcamStream:
    def __init__(self, src=0, input_size=(640, 480)):
        self.src = src
        self.input_size = input_size
        self.cap = None
        self.frame = None
        self.identifying_info = None
        self.stopped = False
        self.lock = threading.Lock()
        self.open_camera()
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            logging.warning(f"Could not open webcam #{self.src}. Retrying...")
            self.cap = None
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.identifying_info = self.get_identifying_info()
        logging.info(f"Webcam #{self.src} opened with resolution {self.input_size[0]}x{self.input_size[1]} at {self.fps} FPS.")
        return True
   
    def get_identifying_info(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        return None

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened() and not self.open_camera():
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                logging.warning(f"Frame capture failed on webcam #{self.src}.")
                time.sleep(1)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join()
        if self.cap:
            self.cap.release()
        logging.info(f"Webcam #{self.src} has been stopped and released.")
