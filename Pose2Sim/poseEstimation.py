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
    Estimate pose from webcams, video files, or a folder of images, and write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real-time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you need another detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and in between tracks points (faster but less accurate).

    If a valid CUDA installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
    '''

    # Read config
    project_dir = config_dict['project']['project_dir']
    frame_range = config_dict.get('project').get('frame_range', [])
    output_dir = config_dict.get('project').get('project_dir')
    multi_person = config_dict.get('project').get('multi_person')
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    show_realtime_results = config_dict['pose'].get('show_realtime_results', False)

    vid_img_extension = config_dict['pose']['vid_img_extension']
    webcam_ids = config_dict.get('pose').get('webcam_ids', [])

    overwrite_pose = config_dict['pose']['overwrite_pose']

    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        if not overwrite_pose:
            logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
            return
        else:
            logging.info('Overwriting previous pose estimation. Set overwrite_pose to false in Config.toml if you want to keep the previous results.')
            raise

    except StopIteration:
        pass  # No existing pose directory, proceed with pose estimation

    except Exception as e:
        logging.debug(f"Exception occurred: {e}")
        pass  # Some other exception occurred, proceed with pose estimation

    logging.info('\nEstimating pose...')

    # Prepare list of sources (webcams, videos, image folders)
    sources = []

    if vid_img_extension == 'webcam':
        if not isinstance(webcam_ids, list):
            webcam_ids = [webcam_ids]
        for cam_id in webcam_ids:
            sources.append({'type': 'webcam', 'id': cam_id, 'path': cam_id})
    else:
        # Add video files
        video_paths = [str(f) for f in Path(video_dir).rglob('*' + vid_img_extension) if f.is_file()]
        for idx, video_path in enumerate(video_paths):
            sources.append({'type': 'video', 'id': idx, 'path': video_path})

        # Add image folders
        image_folders = [str(f) for f in Path(video_dir).iterdir() if f.is_dir()]
        for idx, image_folder in enumerate(image_folders, start=len(sources)):
            image_files = list(Path(image_folder).glob('*' + vid_img_extension))
            if image_files:
                sources.append({'type': 'images', 'id': idx, 'path': image_folder})

    if not sources:
        raise FileNotFoundError(f'No video files, image folders, or webcams found in {video_dir}.')

    logging.info(f'Processing sources: {sources}')

    # Initialize pose tracker
    pose_trackers = {}
    for source in sources:
        pose_tracker = setup_pose_tracker(
            config_dict['pose']['det_frequency'],
            config_dict['pose']['mode'],
            config_dict['pose']['pose_model']
        )
        pose_trackers[source['id']] = pose_tracker

    # Create display queue
    display_queue = queue.Queue()

    # Create and start display thread
    input_size = config_dict['pose'].get('input_size', (640, 480))
    display_thread = CombinedDisplayThread(sources, input_size, display_queue)
    display_thread.start()

    # Initialize streams
    stream_manager = StreamManager(sources, config_dict, pose_trackers, display_queue, output_dir, frame_range)
    stream_manager.start()

    try:
        while not display_thread.stopped:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    finally:
        stream_manager.stop()
        display_thread.stop()
        display_thread.join()


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


def process_single_frame(config_dict, frame, source_id, frame_idx, output_dirs, pose_tracker, multi_person, save_video, save_images, show_realtime_results, out_vid, output_format):
    '''
    Processes a single frame from a source.

    Args:
        config_dict (dict): Configuration dictionary.
        frame (ndarray): Frame image.
        source_id (int): Source ID.
        frame_idx (int): Frame index.
        output_dirs (tuple): Output directories.
        pose_tracker: Pose tracker object.
        multi_person (bool): Whether to track multiple persons.
        output_format (str): Output format.
        save_video (bool): Whether to save the output video.
        save_images (bool): Whether to save output images.
        show_realtime_results (bool): Whether to display results in real time.
        out_vid (cv2.VideoWriter): Video writer object.

    Returns:
        tuple: (source_id, img_show, out_vid)
    '''
    try:
        output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = output_dirs

        if frame is None:
            logging.warning(f"No frame received from source {source_id}. Using placeholder.")
            img_show = np.zeros((config_dict['pose']['input_size'][1], config_dict['pose']['input_size'][0], 3), dtype=np.uint8)
            cv2.putText(img_show, f'Source {source_id} Disconnected', (50, config_dict['pose']['input_size'][1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Perform pose estimation on the frame
            keypoints, scores = pose_tracker(frame)

            # Tracking people IDs across frames (if needed)
            keypoints, scores, _ = track_people(
                keypoints, scores, multi_person, None, None, pose_tracker
            )

            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f'{output_dir_name}_{frame_idx:06d}.json')
                save_to_openpose(json_file_path, keypoints, scores)

            # Draw skeleton on the frame
            img_show = draw_skeleton(frame.copy(), keypoints, scores, kpt_thr=0.1)

            # Save video and images
            if save_video:
                if out_vid is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    H, W = img_show.shape[:2]
                    fps = config_dict['pose'].get('fps', 30)
                    out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
                out_vid.write(img_show)
            if save_images:
                cv2.imwrite(
                    os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'),
                    img_show
                )

        return source_id, img_show, out_vid

    except Exception as e:
        logging.error(f"Error processing frame from source {source_id}: {e}")
        img_placeholder = np.zeros((config_dict['pose']['input_size'][1], config_dict['pose']['input_size'][0], 3), dtype=np.uint8)
        cv2.putText(img_placeholder, f'Error on Source {source_id}', (50, config_dict['pose']['input_size'][1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return source_id, img_placeholder, out_vid


class CombinedDisplayThread(threading.Thread):
    '''
    Thread for displaying combined images to avoid thread-safety issues with OpenCV.
    '''
    def __init__(self, sources, input_size, display_queue):
        super().__init__()
        self.display_queue = display_queue
        self.stopped = False
        self.daemon = True  # Thread will exit when main program exits
        self.sources = sources
        self.input_size = input_size
        self.window_name = "Combined Feeds"
        self.grid_size = self.calculate_grid_size(len(sources))
        self.black_frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        self.frames = {}  # Store the latest frame from each source

        # Initialize placeholders for sources that are not connected at all
        for source in self.sources:
            self.frames[source['id']] = self.get_placeholder_frame(source['id'], 'Not Connected')

    def run(self):
        try:
            while not self.stopped:
                try:
                    frames_dict = self.display_queue.get(timeout=0.1)
                    for source_id, frame in frames_dict.items():
                        if frame is not None:
                            self.frames[source_id] = frame
                        else:
                            self.frames[source_id] = self.get_placeholder_frame(source_id, 'Disconnected')
                    combined_image = self.combine_frames()
                    if combined_image is not None:
                        cv2.imshow(self.window_name, combined_image)
                        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                            logging.info("Display window closed by user.")
                            self.stopped = True
                            break
                    else:
                        time.sleep(0.01)
                except queue.Empty:
                    continue
        finally:
            cv2.destroyAllWindows()

    def combine_frames(self):
        frames_list = [self.frames.get(source['id'], self.black_frame) for source in self.sources]
        if not frames_list:
            return None
        # Resize frames for display
        resized_frames = []
        for frame in frames_list:
            if frame.shape[1] > self.input_size[0] or frame.shape[0] > self.input_size[1]:
                # Resize frame if it's larger than input_size
                resized_frame = cv2.resize(frame, self.input_size)
            else:
                resized_frame = frame
            resized_frames.append(resized_frame)
        rows = []
        for i in range(0, len(resized_frames), self.grid_size[1]):
            row_frames = resized_frames[i:i + self.grid_size[1]]
            if len(row_frames) < self.grid_size[1]:
                row_frames.extend([self.black_frame] * (self.grid_size[1] - len(row_frames)))
            # Ensure all frames in the row have the same height
            min_height = min(frame.shape[0] for frame in row_frames)
            row_frames = [cv2.resize(frame, (frame.shape[1], min_height)) for frame in row_frames]
            rows.append(np.hstack(row_frames))
        # Ensure all rows have the same width
        min_width = min(row.shape[1] for row in rows)
        rows = [cv2.resize(row, (min_width, row.shape[0])) for row in rows]
        combined_image = np.vstack(rows)
        return combined_image

    def get_placeholder_frame(self, source_id, message):
        img_placeholder = self.black_frame.copy()
        cv2.putText(img_placeholder, f'Source {source_id} {message}', (10, self.input_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return img_placeholder

    def calculate_grid_size(self, num_sources):
        cols = int(np.ceil(np.sqrt(num_sources)))
        rows = int(np.ceil(num_sources / cols))
        return (rows, cols)

    def stop(self):
        self.stopped = True

class StreamManager:
    def __init__(self, sources, config_dict, pose_trackers, display_queue, output_dir, frame_range):
        self.sources = sources
        self.config_dict = config_dict
        self.pose_trackers = pose_trackers
        self.display_queue = display_queue
        self.output_dir = output_dir
        self.stopped = False
        self.executor = ThreadPoolExecutor(max_workers=len(sources))
        self.frame_ranges = frame_range if frame_range else []
        self.active_streams = set()

        self.streams = {}
        self.outputs = {}
        self.out_videos = {}

        # Initialize streams and outputs
        for source in sources:
            stream = GenericStream(source, config_dict['pose'].get('input_size', (640, 480)), self.frame_ranges)
            self.streams[source['id']] = stream
            self.outputs[source['id']] = setup_capture_directories(
                source['path'], self.output_dir, 'to_images' in config_dict['project'].get('save_video', [])
            )
            self.out_videos[source['id']] = None  # Placeholder for video writer
            self.active_streams.add(source['id'])

    def start(self):
        # Start all streams
        for stream in self.streams.values():
            stream.start()

        # Start processing loop
        threading.Thread(target=self.process_streams, daemon=True).start()

    def process_streams(self):
        while not self.stopped and self.active_streams:
            frames = {}
            for source_id, stream in self.streams.items():
                frame = stream.read()
                if frame is not None:
                    frames[source_id] = frame
                else:
                    # If the stream is finished and it's not a webcam, remove it from active streams
                    if stream.stopped and stream.source['type'] != 'webcam':
                        self.active_streams.discard(source_id)

            # Process frames in parallel
            futures = {}
            for source_id in self.active_streams:
                frame = frames.get(source_id)
                if frame is not None:
                    futures[self.executor.submit(
                        process_single_frame,
                        self.config_dict,
                        frame,
                        source_id,
                        self.streams[source_id].frame_idx,
                        self.outputs[source_id],
                        self.pose_trackers[source_id],
                        self.config_dict['project'].get('multi_person'),
                        'to_video' in self.config_dict['project'].get('save_video', []),
                        'to_images' in self.config_dict['project'].get('save_video', []),
                        self.config_dict['pose'].get('show_realtime_results', False),
                        self.out_videos.get(source_id),
                        self.config_dict['pose'].get('output_format', 'openpose')
                    )] = source_id

            if not futures:
                if not any(stream.source['type'] == 'webcam' for stream in self.streams.values()):
                    # No frames to process and no webcams running, stop processing
                    logging.info("All streams have finished processing.")
                    self.stopped = True
                    break
                else:
                    # Wait for frames from webcams
                    time.sleep(0.1)
                    continue

            processed_frames = {}
            for future in concurrent.futures.as_completed(futures):
                source_id = futures[future]
                try:
                    source_id, img_show, out_vid = future.result()
                    processed_frames[source_id] = img_show
                    # Update out_videos dictionary with the video writer returned from process_single_frame
                    if out_vid is not None:
                        self.out_videos[source_id] = out_vid
                except Exception as e:
                    logging.error(f"Error processing frame from source {source_id}: {e}")
                    processed_frames[source_id] = self.get_placeholder_frame(source_id, 'Error')

            # Put processed frames into display queue
            if processed_frames:
                self.display_queue.put(processed_frames)
            else:
                # Sleep briefly to prevent high CPU usage when no frames are available
                time.sleep(0.01)

    def stop(self):
        self.stopped = True
        for stream in self.streams.values():
            stream.stop()
        self.executor.shutdown()
        # Release video writers
        for out_vid in self.out_videos.values():
            if out_vid is not None:
                out_vid.release()

    def get_placeholder_frame(self, source_id, message):
        input_size = self.config_dict['pose'].get('input_size', (640, 480))
        img_placeholder = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        cv2.putText(img_placeholder, f'Source {source_id} {message}', (10, input_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return img_placeholder


class GenericStream(threading.Thread):
    def __init__(self, source, input_size=(640, 480), frame_ranges=[]):
        super().__init__()
        self.source = source
        self.input_size = input_size
        self.frame_ranges = frame_ranges
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.daemon = True
        self.frame_idx = 0
        self.total_frames = None
        self.pbar = None

    def run(self):
        source_type = self.source['type']
        if source_type == 'webcam':
            self.cap = cv2.VideoCapture(int(self.source['id']))
            if not self.cap.isOpened():
                logging.error(f"Cannot open webcam {self.source['id']}")
                self.stopped = True
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
            time.sleep(1)  # Give some time for the webcam to initialize

        elif source_type == 'video':
            self.cap = cv2.VideoCapture(self.source['path'])
            if not self.cap.isOpened():
                logging.error(f"Cannot open video file {self.source['path']}")
                self.stopped = True
                return
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.setup_progress_bar()

        elif source_type == 'images':
            image_files = glob.glob(os.path.join(self.source['path'], '*' + self.source['path'].split('.')[-1]))
            self.image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            self.total_frames = len(self.image_files)
            self.image_index = 0
            self.setup_progress_bar()
        else:
            logging.error(f"Unknown source type: {source_type}")
            self.stopped = True
            return

        while not self.stopped:
            if source_type in ['webcam', 'video']:
                ret, frame = self.cap.read()
                if not ret:
                    if source_type == 'video':
                        self.stopped = True
                        if self.pbar is not None:
                            self.pbar.close()
                        break
                    else:
                        logging.error(f"Failed to read frame from {self.source['path']}")
                        time.sleep(0.1)
                        continue
            elif source_type == 'images':
                if self.image_index >= len(self.image_files):
                    self.stopped = True
                    if self.pbar is not None:
                        self.pbar.close()
                    break
                frame = cv2.imread(self.image_files[self.image_index])
                self.image_index += 1
            else:
                frame = None

            if frame is not None:
                frame = cv2.resize(frame, self.input_size)
                with self.lock:
                    self.frame = frame.copy()
                self.frame_idx += 1
                if self.pbar is not None:
                    self.pbar.update(1)
                self.new_frame_event.set()
            else:
                with self.lock:
                    self.frame = None
                time.sleep(0.1)

    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame

    def stop(self):
        self.stopped = True
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        if self.pbar is not None:
            self.pbar.close()

    def setup_progress_bar(self):
        self.pbar = tqdm(total=self.total_frames, desc=f'Processing {os.path.basename(str(self.source["path"]))}', position=self.source['id'], leave=False)

def setup_capture_directories(source_path, output_dir, save_images):
    '''
    Set up output directories for saving images and JSON files.

    Returns:
        tuple: (output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path)
    '''
    if isinstance(source_path, int):
        # Handle webcam source
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'webcam{source_path}_{current_date}'
    elif str(source_path).startswith("webcam"):
        # Handle string starting with 'webcam'
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f'{source_path}_{current_date}'
    else:
        output_dir_name = os.path.basename(os.path.splitext(str(source_path))[0])

    # Define the full path for the output directory
    output_dir_full = os.path.abspath(os.path.join(output_dir, "pose"))

    # Create output directories if they do not exist
    if not os.path.isdir(output_dir_full):
        os.makedirs(output_dir_full)

    # Prepare directories for images and JSON outputs
    img_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_img')
    json_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_json')
    if save_images and not os.path.isdir(img_output_dir):
        os.makedirs(img_output_dir)
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir)

    # Define the path for the output video file
    output_video_path = os.path.join(output_dir_full, f'{output_dir_name}_pose.mp4')

    return output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path
