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
    output_dir = config_dict['project']['project_dir']
    frame_range = config_dict['project'].get('frame_range', [])
    multi_person = config_dict['project'].get('multi_person', False)
    video_dir = os.path.join(output_dir, 'videos')
    pose_dir = os.path.join(output_dir, 'pose')

    show_realtime_results = config_dict['pose'].get('show_realtime_results', False)

    vid_img_extension = config_dict['pose']['vid_img_extension']
    webcam_ids = config_dict['pose'].get('webcam_ids', [])

    overwrite_pose = config_dict['pose'].get('overwrite_pose', False)

    # Check if pose estimation has already been done
    if os.path.exists(pose_dir) and not overwrite_pose:
        logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
        return
    elif overwrite_pose:
        logging.info('Overwriting previous pose estimation.')

    logging.info('Estimating pose...')

    # Prepare list of sources (webcams, videos, image folders)
    sources = []

    if vid_img_extension == 'webcam':
        sources.extend({'type': 'webcam', 'id': cam_id, 'path': cam_id} for cam_id in (webcam_ids if isinstance(webcam_ids, list) else [webcam_ids]))
    else:
        video_paths = [str(f) for f in Path(video_dir).rglob('*' + vid_img_extension) if f.is_file()]
        sources.extend({'type': 'video', 'id': idx, 'path': video_path} for idx, video_path in enumerate(video_paths))
        image_folders = [str(f) for f in Path(video_dir).iterdir() if f.is_dir()]
        sources.extend({'type': 'images', 'id': idx, 'path': folder} for idx, folder in enumerate(image_folders, start=len(video_paths)))


    if not sources:
        raise FileNotFoundError(f'No video files, image folders, or webcams found in {video_dir}.')
    
    process_functions = {}
    for source in sources:
        if source['type'] == 'webcam':
            process_functions[source['id']] = process_single_frame
        else:
            process_functions[source['id']] = process_single_frame

    logging.info(f'Processing sources: {sources}')

    # Initialize pose trackers
    pose_trackers = {source['id']: setup_pose_tracker(config_dict['pose']['det_frequency'], config_dict['pose']['mode'], config_dict['pose']['pose_model']) for source in sources}

    # Create display queue
    display_queue = queue.Queue()

    # Initialize streams
    stream_manager = StreamManager(sources, config_dict, pose_trackers, display_queue, output_dir, frame_range, process_functions)
    stream_manager.start()

    # Start display thread only if show_realtime_results is True
    if show_realtime_results:
        # Create and start display thread
        input_size = config_dict['pose'].get('input_size', (640, 480))
        display_thread = CombinedDisplayThread(sources, input_size, display_queue)
        display_thread.start()
    else:
        display_thread = None

    try:
        while not stream_manager.stopped:
            if display_thread and display_thread.stopped:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    finally:
        stream_manager.stop()
        if display_thread:
            display_thread.stop()
            display_thread.join()

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

    logging.info(f"Processing frame {frame_idx} from source {source_id}")

    try:
        output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = output_dirs
        
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
            if out_vid is not None:
                out_vid.write(img_show)

        if save_images:
            cv2.imwrite(os.path.join(img_output_dir, f'{frame_idx:06d}.jpg'), img_show)


        return source_id, img_show, out_vid

    except Exception as e:
        logging.error(f"Error processing frame from source {source_id}: {e}")
        img_placeholder = get_placeholder_frame(config_dict['pose']['input_size'], source_id, 'Error')
        return source_id, img_placeholder, out_vid

def get_placeholder_frame(input_size, source_id, message):
    img_placeholder = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    cv2.putText(img_placeholder, f'Source {source_id}: {message}', (50, input_size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return img_placeholder

class CombinedDisplayThread(threading.Thread):
    '''
    Thread for displaying combined images to avoid thread-safety issues with OpenCV.
    '''
    def __init__(self, sources, input_size, display_queue):
        super().__init__(daemon=True)
        self.display_queue = display_queue
        self.stopped = False
        self.sources = sources
        self.input_size = input_size
        self.window_name = "Combined Feeds"
        self.grid_size = self.calculate_grid_size(len(sources))
        self.black_frame = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        self.frames = {}  # Store the latest frame from each source

        # Initialize placeholders for sources
        for source in self.sources:
            self.frames[source['id']] = get_placeholder_frame(self.input_size, source['id'], 'Not Connected')

        # Create a consistent order of source IDs for display
        self.source_ids = [source['id'] for source in self.sources]

    def run(self):
        while not self.stopped:
            try:
                frames_dict = self.display_queue.get(timeout=0.1)
                for source_id in self.source_ids:
                    self.frames[source_id] = frames_dict.get(source_id, self.frames[source_id])
                self.display_combined_image()
            except queue.Empty:
                continue

    def display_combined_image(self):
        combined_image = self.combine_frames()
        if combined_image is not None:
            cv2.imshow(self.window_name, combined_image)
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                logging.info("Display window closed by user.")
                self.stopped = True

    def combine_frames(self):
        # Use the consistent order of source IDs
        frames_list = []
        for source_id in self.source_ids:
            frame = self.frames.get(source_id, self.black_frame)
            frames_list.append(frame)

        if not frames_list:
            return None

        # Resize frames for display
        resized_frames = []
        for frame in frames_list:
            if frame.shape[1] != self.input_size[0] or frame.shape[0] != self.input_size[1]:
                resized_frame = cv2.resize(frame, self.input_size)
            else:
                resized_frame = frame
            resized_frames.append(resized_frame)

        rows = []
        for i in range(0, len(resized_frames), self.grid_size[1]):
            row_frames = resized_frames[i:i + self.grid_size[1]]
            rows.append(np.hstack(row_frames))

        return np.vstack(rows)

    def calculate_grid_size(self, num_sources):
        cols = int(np.ceil(np.sqrt(num_sources)))
        rows = int(np.ceil(num_sources / cols))
        return (rows, cols)

    def stop(self):
        self.stopped = True

    def __del__(self):
        cv2.destroyAllWindows()


class StreamManager:
    def __init__(self, sources, config_dict, pose_trackers, display_queue, output_dir, frame_range, process_functions):
        self.sources = sources
        self.config_dict = config_dict
        self.pose_trackers = pose_trackers
        self.display_queue = display_queue
        self.output_dir = output_dir
        self.executor = ThreadPoolExecutor(max_workers=len(sources))
        self.active_streams = set()
        self.process_functions = process_functions
        self.frame_ranges = frame_range or []
        self.stopped = False
        self.streams, self.outputs, self.out_videos = {}, {}, {}

        self.initialize_streams_and_outputs()

    def initialize_streams_and_outputs(self):
        for source in self.sources:
            stream = GenericStream(source, self.config_dict, self.frame_ranges)
            self.streams[source['id']] = stream
            self.outputs[source['id']] = setup_capture_directories(
                source['path'], self.output_dir, 'to_images' in self.config_dict['project'].get('save_video', [])
            )
            self.out_videos[source['id']] = None
            self.active_streams.add(source['id'])

    def start(self):
        for stream in self.streams.values():
            stream.start()
        threading.Thread(target=self.process_streams, daemon=True).start()

    def process_streams(self):
        logging.info("Début du traitement des flux")
        while not self.stopped and self.active_streams:
            frames = {}
            for source_id, stream in self.streams.items():
                if not stream.stopped:
                    frame = stream.read()
                    if frame is not None:
                        frames[source_id] = frame
                        logging.info(f"Frame {stream.frame_idx} lue depuis la source {source_id}")
                    else:
                        logging.info(f"Aucune frame lue depuis la source {source_id}")
                else:
                    logging.info(f"Le flux {source_id} est arrêté")
                    if source_id in self.active_streams:
                        self.active_streams.discard(source_id)
                        logging.info(f"Flux {source_id} retiré des flux actifs.")

            if not frames:
                if not self.active_streams:
                    logging.info("Tous les flux ont terminé le traitement.")
                    self.stopped = True
                    break
                else:
                    logging.info("Aucune frame disponible actuellement, attente...")
                    time.sleep(0.1)
                    continue

            logging.info(f"Soumission de {len(frames)} frames pour le traitement")
            futures = {self.executor.submit(self.process_frame, source_id, frame): source_id
                    for source_id, frame in frames.items() if frame is not None}

            self.handle_future_results(futures)

    def process_frame(self, source_id, frame):
        process_function = self.process_functions.get(source_id)
        return process_function(self.config_dict,
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
                        self.config_dict['pose'].get('output_format', 'openpose'))

    def handle_future_results(self, futures):
        for future in futures:
            source_id = futures[future]
            try:
                result = future.result()
                source_id, processed_frame, out_vid = result
                self.display_queue.put({source_id: processed_frame})
            except Exception as e:
                logging.error(f"Error processing frame from source {source_id}: {e}")

    def initialize_video_writer(self, img_show, source_id):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        H, W = img_show.shape[:2]
        fps = self.config_dict['pose'].get('fps', 30)
        output_video_path = f"{self.outputs[source_id]}/{source_id}_output.mp4"
        return cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))

    def stop(self):
        self.stopped = True
        for stream in self.streams.values():
            stream.stop()
        self.executor.shutdown()
        for out_vid in self.out_videos.values():
            if out_vid is not None:
                out_vid.release()

class GenericStream(threading.Thread):
    def __init__(self, source, config_dict, frame_ranges=None):
        super().__init__()
        self.source = source
        self.input_size = config_dict['pose'].get('input_size', (640, 480))
        self.image_extension = config_dict['pose']['vid_img_extension']
        self.frame_ranges = frame_ranges
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.daemon = True
        self.frame_idx = 0
        self.total_frames = None
        self.cap = None
        self.image_files = []
        self.image_index = 0
        self.pbar = None

    def run(self):
        if self.source['type'] == 'webcam':
            self.setup_webcam()
        elif self.source['type'] == 'video':
            self.open_video()
        elif self.source['type'] == 'images':
            self.load_images()
        else:
            logging.error(f"Unknown source type: {self.source['type']}")
            self.stopped = True
            return

        while not self.stopped:
            self.process_frame()

    def setup_webcam(self):
        self.open_webcam()
        time.sleep(1)  # Give time for the webcam to initialize

    def open_video(self):
        self.cap = cv2.VideoCapture(self.source['path'])
        if not self.cap.isOpened():
            logging.error(f"Cannot open video file {self.source['path']}")
            self.stopped = True
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.setup_progress_bar()

    def load_images(self):
        image_files = glob.glob(os.path.join(self.source['path'], f'*{self.image_extension}'))
        self.image_files = sorted(image_files)
        self.total_frames = len(self.image_files)
        self.setup_progress_bar()

    def process_frame(self):
        if self.source['type'] == 'webcam':
            frame = self.read_webcam_frame()
        elif self.source['type'] == 'video':
            if not self.frame_ranges or self.frame_idx in self.frame_ranges:
                ret, frame = self.cap.read()
                if not ret:
                    logging.info(f"End of video {self.source['path']}")
                    self.stopped = True
                    if self.pbar:
                        self.pbar.close()
                    return
                frame = cv2.resize(frame, self.input_size)
                with self.lock:
                    self.frame = frame
                self.frame_idx += 1
                if self.pbar:
                    self.pbar.update(1)
        elif self.source['type'] == 'images':
            frame = cv2.imread(self.image_files[self.image_index]) if self.image_index < len(self.image_files) else None
            self.image_index += 1

        if frame is not None:
            frame = cv2.resize(frame, self.input_size)
            with self.lock:
                self.frame = frame
            self.frame_idx += 1
            if self.pbar:
                self.pbar.update(1)
        else:
            time.sleep(0.1)

    def open_webcam(self):
        self.connected = False  # Add this attribute
        try:
            self.cap = cv2.VideoCapture(int(self.source['id']), cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_size[1])
                logging.info(f"Webcam {self.source['id']} opened.")
                self.connected = True
            else:
                logging.error(f"Cannot open webcam {self.source['id']}.")
                self.cap = None
        except Exception as e:
            logging.error(f"Exception occurred while opening webcam {self.source['id']}: {e}")
            self.cap = None

    def read_webcam_frame(self):
        if self.cap is None or not self.cap.isOpened():
            logging.warning(f"Webcam {self.source['id']} not opened. Attempting to open...")
            self.open_webcam()
            if self.cap is None or not self.cap.isOpened():
                with self.lock:
                    self.frame = None
                return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.warning(f"Failed to read frame from webcam {self.source['id']}.")
            self.cap.release()
            self.cap = None
            with self.lock:
                self.frame = None
            return None
        return frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()
        if self.pbar:
            self.pbar.close()

    def setup_progress_bar(self):
        self.pbar = tqdm(total=self.total_frames, desc=f'Processing {os.path.basename(str(self.source["path"]))}', position=self.source['id'])


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
    detections = []
    for i in range(nb_detections):  # Number of detected people
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
    if not os.path.isdir(json_output_dir):
        os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)
