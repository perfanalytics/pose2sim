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
import queue
import multiprocessing

import itertools as it
import numpy as np

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, draw_skeleton
from scipy.optimize import linear_sum_assignment

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
    source_dir = os.path.join(output_dir, 'videos')
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
        video_files = [str(f) for f in Path(source_dir).rglob('*' + vid_img_extension) if f.is_file()]
        sources.extend({'type': 'video', 'id': idx, 'path': video_path} for idx, video_path in enumerate(video_files))
        image_dirs = [str(f) for f in Path(source_dir).iterdir() if f.is_dir()]
        sources.extend({'type': 'images', 'id': idx, 'path': folder} for idx, folder in enumerate(image_dirs, start=len(video_files)))

    if not sources:
        raise FileNotFoundError(f'No Webcams or no media files found in {source_dir}.')

    process_functions = {}
    for source in sources:
        if source['type'] == 'webcam':
            process_functions[source['id']] = process_single_frame
        else:
            process_functions[source['id']] = process_single_frame

    logging.info(f'Processing sources: {sources}')

    # Create display queue
    display_queue = multiprocessing.Queue()

    pose_tracker_settings = determine_tracker_settings(config_dict)

    # Create a global frame queue
    frame_queue = multiprocessing.Queue(maxsize=100)

    # Initialize shared memory buffers
    expected_frame_size = config_dict['pose'].get('input_size', (640, 480))
    frame_width, frame_height = expected_frame_size
    expected_frame_size_bytes = frame_width * frame_height * 3  # Assuming 3 channels
    num_buffers = 10  # Adjust as needed

    # Preallocate shared memory buffers
    shared_buffers = {}
    available_buffers = multiprocessing.Queue()
    for i in range(num_buffers):
        unique_name = f"frame_buffer_{i}"
        shm = multiprocessing.shared_memory.SharedMemory(name=unique_name, create=True, size=expected_frame_size_bytes)
        shared_buffers[unique_name] = shm
        available_buffers.put(unique_name)

    # Create dictionaries to hold outputs for each source
    source_outputs = {}

    # Start reading processes for each source
    reading_processes = []
    shared_counts = {}
    for source in sources:
        shared_counts[source['id']] = {
            'queued': multiprocessing.Value('i', 0),
            'processed': multiprocessing.Value('i', 0)
        }
        # Initialize outputs for each source
        outputs = setup_capture_directories(
            source['path'], pose_dir, 'to_images' in config_dict['project'].get('save_video', [])
        )
        # Correctly store the outputs in the dictionary
        source_outputs[source['id']] = outputs

        process = SourceProcess(
            source,
            config_dict,
            frame_queue,
            available_buffers,
            shared_buffers,
            shared_counts,
            expected_frame_size
        )
        process.start()
        reading_processes.append(process)


    # Compute the maximum number of worker processes
    cpu_count = multiprocessing.cpu_count()
    num_sources = len(sources)
    max_workers = max(1, cpu_count - num_sources - 2)
    logging.info(f"Starting {max_workers} worker processes.")

    # Start worker processes
    worker_processes = []
    for _ in range(max_workers):
        worker = WorkerProcess(
            config_dict,
            frame_queue,
            available_buffers,
            shared_buffers,
            shared_counts,
            pose_tracker_settings,
            display_queue,
            source_outputs
        )
        worker.start()
        worker_processes.append(worker)

    # Start display thread only if show_realtime_results is True
    display_thread = None
    if show_realtime_results:
        input_size = config_dict['pose'].get('input_size', (640, 480))
        display_thread = DisplayProcess(sources, input_size, display_queue)
        display_thread.start()

    # Initialize progress bars
    progress_bars = {}
    for idx, source in enumerate(sources):
        source_id = source['id']
        desc = f"Source {source_id}"
        progress_bars[source_id] = tqdm(total=0, desc=desc, position=idx, leave=True)

    try:
        while any(process.is_alive() for process in reading_processes) or not frame_queue.empty():
            for source_id, pb in progress_bars.items():
                counts = shared_counts[source_id]
                queued = counts['queued'].value
                processed = counts['processed'].value
                pb.total = queued
                pb.n = processed
                pb.refresh()
            if display_thread and display_thread.stopped:
                break
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    finally:
        # Wait for reading processes to finish
        for process in reading_processes:
            process.join()
        # Signal worker processes to stop
        for _ in worker_processes:
            frame_queue.put(None)
        # Wait for worker processes to finish
        for worker in worker_processes:
            worker.join()

        # Clean up shared memory
        for shm in shared_buffers.values():
            shm.close()
            shm.unlink()

        if display_thread:
            display_thread.stop()
            display_thread.join()
        for pb in progress_bars.values():
            pb.close()
        logging.shutdown()


def process_single_frame(config_dict, frame, source_id, frame_idx, output_dirs, pose_tracker, multi_person, save_video, save_images, show_realtime_results, output_format, out_vid, prev_keypoints):
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
        tuple: (source_id, img_show)
    '''
    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = output_dirs

    # Perform pose estimation on the frame
    keypoints, scores = pose_tracker(frame)

    # Tracking people IDs across frames (if needed)
    keypoints, scores, prev_keypoints = track_people(
        keypoints, scores, multi_person, None, prev_keypoints, pose_tracker
    )

    if 'openpose' in output_format:
        json_file_path = os.path.join(json_output_dir, f'{output_dir_name}_{frame_idx:06d}.json')
        save_to_openpose(json_file_path, keypoints, scores)

    # Draw skeleton on the frame
    img_show = draw_skeleton(frame, keypoints, scores, kpt_thr=0.1)

    # Save video and images
    if save_video and out_vid is not None:
        out_vid.write(img_show)

    if save_images:
        cv2.imwrite(os.path.join(img_output_dir, f'{output_dir_name}_{frame_idx:06d}.jpg'), img_show)

    return (source_id, img_show), keypoints


class DisplayProcess(multiprocessing.Process):
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
        self.window_size = (1920, 1080)
        self.grid_size = self.calculate_grid_size(len(sources))
        self.img_placeholder = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        self.frames = {source['id']: self.get_placeholder_frame(source['id'], 'Not Connected') for source in sources}
        self.source_ids = [source['id'] for source in self.sources]

    def run(self):
        while not self.stopped:
            try:
                frames_dict = self.display_queue.get(timeout=0.1)
                if frames_dict:
                    self.frames.update(frames_dict)
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
        window_width, window_height = self.window_size
        rows, cols = self.grid_size
        frame_width = window_width // cols
        frame_height = window_height // rows
        frame_size = (frame_width, frame_height)
        
        combined_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < len(self.source_ids):
                    source_id = self.source_ids[idx]
                    frame = self.frames.get(source_id, self.img_placeholder)
                    resized_frame = cv2.resize(frame, frame_size)
                    y1 = i * frame_height
                    y2 = y1 + frame_height
                    x1 = j * frame_width
                    x2 = x1 + frame_width
                    combined_image[y1:y2, x1:x2] = resized_frame
                    idx += 1
                else:
                    continue
        return combined_image

    def calculate_grid_size(self, num_sources):
        cols = int(np.ceil(np.sqrt(num_sources)))
        rows = int(np.ceil(num_sources / cols))
        return (rows, cols)

    def stop(self):
        self.stopped = True

    def get_placeholder_frame(self, source_id, message):
        return cv2.putText(self.img_placeholder.copy(), f'Source {source_id}: {message}', (50, self.input_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def __del__(self):
        cv2.destroyAllWindows()


class WorkerProcess(multiprocessing.Process):
    def __init__(self, config_dict, frame_queue, available_buffers, shared_buffers, shared_counts, pose_tracker_settings, display_queue, source_outputs):
        super().__init__()
        self.config_dict = config_dict
        self.frame_queue = frame_queue
        self.available_buffers = available_buffers
        self.shared_buffers = shared_buffers
        self.shared_counts = shared_counts
        self.pose_tracker_settings = pose_tracker_settings
        self.display_queue = display_queue
        self.stopped = False
        self.source_outputs = source_outputs
        self.prev_keypoints = {}

    def run(self):
        try:
            # Initialize the pose tracker
            pose_tracker = setup_pose_tracker(self.pose_tracker_settings)
        
            # Prepare other necessary parameters
            multi_person = self.config_dict['project'].get('multi_person', False)
            save_video = 'to_video' in self.config_dict['project'].get('save_video', [])
            save_images = 'to_images' in self.config_dict['project'].get('save_video', [])
            show_realtime_results = self.config_dict['pose'].get('show_realtime_results', False)
            output_format = self.config_dict['project'].get('output_format', 'openpose')

            # Initialize a dictionary to store out_vids for each source inside the process
            local_out_vids = {}

            while True:
                item = self.frame_queue.get()
                if item is None:
                    # No more frames to process
                    break
                buffer_name, frame_shape, frame_dtype_str, source_id, frame_idx = item
                shm = self.shared_buffers[buffer_name]
                frame = np.ndarray(frame_shape, dtype=np.dtype(frame_dtype_str), buffer=shm.buf)

                # Get the outputs for this source
                outputs = self.source_outputs[source_id]

                # Create out_vid for this source if needed and not already created
                if save_video and source_id not in local_out_vids:
                    # Create VideoWriter for this source
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = self.config_dict['pose'].get('fps', 30)
                    input_size = self.config_dict['pose'].get('input_size', (640, 480))
                    W, H = input_size
                    output_video_path = outputs[4]
                    out_vid = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H))
                    local_out_vids[source_id] = out_vid
                else:
                    out_vid = local_out_vids.get(source_id)

                prev_keypoints = self.prev_keypoints.get(source_id, None)

                # Process the frame
                result, current_keypoints = process_single_frame(
                    self.config_dict,
                    frame,
                    source_id,
                    frame_idx,
                    outputs,
                    pose_tracker,
                    multi_person,
                    save_video,
                    save_images,
                    show_realtime_results,
                    output_format,
                    out_vid,
                    prev_keypoints
                )

                self.prev_keypoints[source_id] = current_keypoints

                # Update shared counts
                with self.shared_counts[source_id]['processed'].get_lock():
                    self.shared_counts[source_id]['processed'].value += 1

                # Release the buffer back to available_buffers
                self.available_buffers.put(buffer_name)

                # Handle display
                if self.display_queue:
                    self.display_queue.put({source_id: result[1]})

            # Release VideoWriters
            for out_vid in local_out_vids.values():
                out_vid.release()

        except Exception as e:
            logging.error(f"Error in WorkerProcess: {e}")
            self.stopped = True


class SourceProcess(multiprocessing.Process):
    def __init__(self, source, config_dict, frame_queue, available_buffers, shared_buffers, shared_counts, frame_size):
        super().__init__()
        self.source = source
        self.config_dict = config_dict
        self.frame_queue = frame_queue
        self.image_extension = config_dict['pose']['vid_img_extension']
        self.stopped = False
        self.frame_idx = 0
        self.total_frames = 0
        self.cap = None
        self.image_files = []
        self.image_index = 0
        self.pbar = None
        self.frame_ranges = None
        self.available_buffers = available_buffers
        self.shared_buffers = shared_buffers
        self.shared_counts = shared_counts
        self.frame_size = frame_size

    def parse_frame_ranges(self, frame_ranges):
        if self.source['type'] != 'webcam':
            if len(frame_ranges) == 2 and all(isinstance(x, int) for x in frame_ranges):
                start_frame, end_frame = frame_ranges
                return set(range(start_frame, end_frame + 1))
            elif len(frame_ranges) == 0:
                return None
            else:
                return set(frame_ranges)
        else:
            return None

    def run(self):
        try:
            if self.source['type'] == 'webcam':
                self.setup_webcam()
                time.sleep(1)
            elif self.source['type'] == 'video':
                self.open_video()
                self.frame_ranges = self.parse_frame_ranges(self.config_dict['project'].get('frame_range', []))
                if self.frame_ranges:
                    self.total_frames = len(self.frame_ranges)
                else:
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.setup_progress_bar()
            elif self.source['type'] == 'images':
                self.load_images()
            else:
                logging.error(f"Unknown source type: {self.source['type']}")
                self.stopped = True
                return
            while not self.stopped:
                frame = self.capture_frame()
                if frame is not None:
                    # Get a buffer from available_buffers
                    buffer_name = self.available_buffers.get()
                    shm = self.shared_buffers[buffer_name]
                    np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
                    np.copyto(np_frame, frame)
                    # Put the buffer info into frame_queue
                    item = (buffer_name, frame.shape, frame.dtype.str, self.source['id'], self.frame_idx)
                    self.frame_queue.put(item)
                    # Update shared_counts
                    with self.shared_counts[self.source['id']]['queued'].get_lock():
                        self.shared_counts[self.source['id']]['queued'].value += 1
                    self.frame_idx += 1
                else:
                    break
        except Exception as e:
            logging.error(f"Error in SourceProcess: {e}")
            self.stopped = True

    def shared_counts_lock(self):
        return multiprocessing.Lock()

    def cleanup_shared_memory(self):
        for shm in self.shm_list:
            shm.close()
            shm.unlink()
        self.shm_list.clear()

    def setup_webcam(self):
        self.open_webcam()
        time.sleep(1)

    def open_video(self):
        self.cap = cv2.VideoCapture(self.source['path'])
        if not self.cap.isOpened():
            logging.error(f"Cannot open video file {self.source['path']}")
            self.stopped = True
            return

    def load_images(self):
        path_pattern = os.path.join(self.source['path'], f'*{self.image_extension}')
        self.image_files = sorted(glob.glob(path_pattern))
        self.total_frames = len(self.image_files)
        self.setup_progress_bar()

    def capture_frame(self):
        frame = None
        if self.source['type'] == 'webcam':
            frame = self.read_webcam_frame()
            if frame is not None:
                self.frame_idx += 1
        elif self.source['type'] == 'video':
            ret, frame = self.cap.read()
            if not ret:
                logging.info(f"End of video {self.source['path']}")
                self.stopped = True
                if self.pbar:
                    self.pbar.close()
                return None
            if self.frame_ranges and self.frame_idx not in self.frame_ranges:
                logging.debug(f"Skipping frame {self.frame_idx} as it's not in the specified frame range.")
                self.frame_idx += 1
                return self.capture_frame()
            else:
                logging.debug(f"Reading frame {self.frame_idx} from video {self.source['path']}.")
        elif self.source['type'] == 'images':
            if self.image_index < len(self.image_files):
                frame = cv2.imread(self.image_files[self.image_index])
                self.image_index += 1
                self.frame_idx += 1
            else:
                self.stopped = True
                return None

        if frame is not None:
            # Avoid unnecessary resizing if frame already matches expected size
            if frame.shape[1] != self.frame_size[0] or frame.shape[0] != self.frame_size[1]:
                frame = cv2.resize(frame, (self.frame_size[0], self.frame_size[1]))
            return frame
        return None

    def open_webcam(self):
        self.connected = False
        try:
            self.cap = cv2.VideoCapture(int(self.source['id']), cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
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
                return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            logging.warning(f"Failed to read frame from webcam {self.source['id']}.")
            self.cap.release()
            self.cap = None
            return None
        return frame

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
    os.makedirs(output_dir_full, exist_ok=True)

    # Prepare directories for images and JSON outputs
    img_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_img')
    json_output_dir = os.path.join(output_dir_full, f'{output_dir_name}_json')
    if save_images:
        os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

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
    keypoints_with_confidence = np.concatenate([keypoints, scores[:, :, np.newaxis]], axis=2)
    keypoints_with_confidence_flat = keypoints_with_confidence.reshape((keypoints.shape[0], -1))

    detections = [
        {
            "person_id": [-1],
            "pose_keypoints_2d": person_keypoints.tolist(),
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }
        for person_keypoints in keypoints_with_confidence_flat
    ]

    # Create JSON output structure
    json_output = {"version": 1.3, "people": detections}

    # Save JSON output for each frame
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def determine_tracker_settings(config_dict):
    det_frequency = config_dict['pose']['det_frequency']
    mode = config_dict['pose']['mode']
    pose_model = config_dict['pose']['pose_model']

    try:
        import torch
        import onnxruntime as ort
        if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
            device = 'cuda'
            backend = 'onnxruntime'
            logging.info(f"Valid CUDA installation found: using ONNXRuntime backend with GPU.")
        elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
            device = 'rocm'
            backend = 'onnxruntime'
            logging.info(f"Valid ROCM installation found: using ONNXRuntime backend with GPU.")
        else:
            raise
    except:
        try:
            import onnxruntime as ort
            if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                device = 'mps'
                backend = 'onnxruntime'
                logging.info(f"Valid MPS installation found: using ONNXRuntime backend with GPU.")
            else:
                raise
        except:
            device = 'cpu'
            backend = 'openvino'
            logging.info(f"No valid CUDA installation found: using OpenVINO backend with CPU.")

        logging.info(f"Using device: {device}, backend: {backend}")

    if det_frequency > 1:
        logging.info(f'Inference run every {det_frequency} frames. In between, pose estimation tracks previously detected points.')
    elif det_frequency == 1:
        logging.info(f'Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")

    # Select the appropriate model based on the model_type
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_class = BodyWithFeet
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY'):
        model_class = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_class = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    else:
        raise ValueError(f"Invalid model_type: {pose_model}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'.")

    logging.info(f'Mode: {mode}.')

    return (model_class, det_frequency, mode, backend, device)


def setup_pose_tracker(pose_tracker_settings):
    # Initialize the pose tracker with the selected model
    model_class, det_frequency, mode, backend, device = pose_tracker_settings
    
    pose_tracker = PoseTracker(
        model_class,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=False,
        to_openpose=False)

    return pose_tracker


def track_people(keypoints, scores, multi_person, tracking_mode, prev_keypoints, pose_tracker=None):
    return_first_only = not multi_person

    if tracking_mode == 'rtmlib':
        keypoints, scores = sort_people_rtmlib(pose_tracker, keypoints, scores, return_first_only)
    else:
        if prev_keypoints is None:
            prev_keypoints = keypoints
        prev_keypoints, keypoints, scores = sort_people_hungarian(prev_keypoints, keypoints, scores, return_first_only)

    return keypoints, scores, prev_keypoints


def sort_people_rtmlib(pose_tracker, keypoints, scores, return_first_only=False):
    '''
    Associate persons across frames (RTMLib method)

    INPUTS:
    - pose_tracker: PoseTracker. The initialized RTMLib pose tracker object
    - keypoints: array of shape K, L, M with K the number of detected persons,
    L the number of detected keypoints, M their 2D coordinates
    - scores: array of shape K, L with K the number of detected persons,
    L the confidence of detected keypoints

    OUTPUT:
    - sorted_keypoints: array with reordered persons
    - sorted_scores: array with reordered scores
    '''
    
    try:
        desired_size = max(pose_tracker.track_ids_last_frame)+1
        sorted_keypoints = np.full((desired_size, keypoints.shape[1], 2), np.nan)
        sorted_keypoints[pose_tracker.track_ids_last_frame] = keypoints[:len(pose_tracker.track_ids_last_frame), :, :]
        sorted_scores = np.full((desired_size, scores.shape[1]), np.nan)
        sorted_scores[pose_tracker.track_ids_last_frame] = scores[:len(pose_tracker.track_ids_last_frame), :]
    except:
        sorted_keypoints, sorted_scores = keypoints, scores

    if return_first_only and sorted_keypoints.shape[0] > 0:
        sorted_keypoints = np.array([sorted_keypoints[0]])
        sorted_scores = np.array([sorted_scores[0]])

    return sorted_keypoints, sorted_scores


def sort_people_hungarian(keyptpre, keypt, scores, return_first_only=False):
    '''
    Matches people across frames using the Hungarian algorithm.

    INPUTS:
    - keyptpre: array of shape (K_prev, L, 2)
    - keypt: array of shape (K_curr, L, 2)
    - scores: array of shape (K_curr, L)

    OUTPUTS:
    - sorted_prev_keypoints: array with reordered people with the values from the previous frame if the current is empty
    - sorted_keypoints: array with reordered people
    - sorted_scores: array with reordered scores
    '''

    K_prev = len(keyptpre)
    K_curr = len(keypt)

    # Calculate the cost matrix
    cost_matrix = np.zeros((K_prev, K_curr))

    for i in range(K_prev):
        for j in range(K_curr):
            # Calculate the average Euclidean distance between keypoints
            dist = np.linalg.norm(keyptpre[i] - keypt[j], axis=1)
            valid = ~np.isnan(dist)
            if np.any(valid):
                cost_matrix[i, j] = np.nanmean(dist[valid])
            else:
                cost_matrix[i, j] = np.inf

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Initialize the sorted arrays
    sorted_keypoints = np.full_like(keyptpre, np.nan)
    sorted_scores = np.full((K_prev, keypt.shape[1]), np.nan)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] != np.inf:
            sorted_keypoints[r] = keypt[c]
            sorted_scores[r] = scores[c]
        else:
            pass  # No valid assignment

    # Keep previous keypoints if no new ones are assigned
    sorted_prev_keypoints = np.where(np.isnan(sorted_keypoints), keyptpre, sorted_keypoints)

    if return_first_only and sorted_keypoints.shape[0] > 0:
        sorted_keypoints = np.array([sorted_keypoints[0]])
        sorted_scores = np.array([sorted_scores[0]])

    return sorted_prev_keypoints, sorted_keypoints, sorted_scores
