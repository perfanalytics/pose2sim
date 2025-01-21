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
import time
import math
import re
import json
import glob
import ast
import logging
import queue
import multiprocessing
import psutil
import cv2
import numpy as np

from datetime import datetime
from pathlib import Path
from functools import partial
from multiprocessing import shared_memory
from tqdm import tqdm
from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton

from Pose2Sim.common import natural_sort_key, sort_people_sports2d
from Pose2Sim.skeletons import *


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
def init_pose_tracker(pose_tracker_settings):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - ModelClass: class. The RTMlib model class to use for pose detection (Body, BodyWithFeet, Wholebody)
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker
    - backend: str. The backend to use for pose detection (onnxruntime, openvino, opencv)
    - device: str. The device to use for pose detection (cpu, cuda, rocm, mps)

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    ModelClass, det_frequency, mode, tracking, backend, device = pose_tracker_settings
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False)
    return pose_tracker


def init_backend_device(backend='auto', device='auto'):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device == 'auto' or backend == 'auto':
        if device != 'auto' or backend != 'auto':
            logging.warning("If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                logging.info("Valid CUDA installation found: using ONNXRuntime backend with GPU.")
                return 'onnxruntime', 'cuda'
            elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
                logging.info("Valid ROCM installation found: using ONNXRuntime backend with GPU.")
                return 'onnxruntime', 'rocm'
            else:
                raise
        except:
            try:
                import onnxruntime as ort
                if ('MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers()):
                    logging.info("Valid MPS installation found: using ONNXRuntime backend with GPU.")
                    return 'onnxruntime', 'mps'
                else:
                    raise
            except:
                logging.info("No valid CUDA installation found: using OpenVINO backend with CPU.")
                return 'openvino', 'cpu'

    else:
        return backend.lower(), device.lower()


def save_keypoints_to_openpose(json_file_path, all_keypoints, all_scores):
    '''
    Save the keypoints and scores to a JSON file in the OpenPose format

    INPUTS:
    - json_file_path: Path to save the JSON file
    - keypoints: Detected keypoints
    - scores: Confidence scores for each keypoint

    OUTPUTS:
    - JSON file with the detected keypoints and confidence scores in the OpenPose format
    '''

    detections = []

    for idx_person in range(len(all_keypoints)):
        keypoints_with_confidence_i = []
        for (kp, score) in zip(all_keypoints[idx_person], all_scores[idx_person]):
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
    with open(json_file_path, 'w') as outfile:
        json.dump(json_output, outfile)


def process_frame(vid_img_extension, frame, source_id, frame_idx, output_dirs, pose_tracker,
                  multi_person, save_video, save_images, display_detection,
                  output_format, video_writer, previous_keypoints, timestamp=None):
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
        display_detection (bool): Whether to display results in real time.
        out_vid (cv2.VideoWriter): Video writer object.

    Returns:
        tuple: (source_id, img_show)
    '''
    output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path = output_dirs

    # Pose estimation
    keypoints, scores = pose_tracker(frame)

    # Sort persons consistently if multiple persons
    if multi_person:
        if previous_keypoints is None:
            previous_keypoints = keypoints
        previous_keypoints, keypoints, scores = sort_people_sports2d(previous_keypoints, keypoints, scores=scores)

    if vid_img_extension == 'webcam':
        file_name = f"{output_dir_name}_{timestamp}"
    else:
        file_name = f"{output_dir_name}_{frame_idx:06d}"

    if 'openpose' in output_format.lower():
        json_path = os.path.join(json_output_dir, f"{file_name}.json")
        save_keypoints_to_openpose(json_path, keypoints, scores)

    # Draw skeleton if needed
    annotated_frame = None
    if display_detection or save_video or save_images:
        annotated_frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.1)

    if save_video and video_writer:
        video_writer.write(annotated_frame)

    if save_images:
        cv2.imwrite(os.path.join(img_output_dir, f"{file_name}.jpg"), annotated_frame)

    return (source_id, annotated_frame), keypoints


class FrameSynchronizer(multiprocessing.Process):
    '''
    Class to display frames from multiple sources in a synchronized grid.
    '''

    def __init__(self, sources, display_queue):
        super().__init__(daemon=True)
        self.num_sources = len(sources)
        self.display_queue = display_queue
        self.stopped = False
        self.frame_data = {}

    def run(self):
        while not self.stopped:
            try:
                source_id, frame_idx, final_frame = self.display_queue.get(timeout=0.1)
                if final_frame is None:
                    continue

                if frame_idx not in self.frame_data:
                    self.frame_data[frame_idx] = {}
                self.frame_data[frame_idx][source_id] = final_frame

                # Once we have frames from all sources for this frame_idx, display them
                if len(self.frame_data[frame_idx]) == self.num_sources:
                    self.show_synchronized(frame_idx)
                    del self.frame_data[frame_idx]
            except queue.Empty:
                pass

    def show_synchronized(self, frame_idx):
        frames_dict = self.frame_data.get(frame_idx, {})
        if not frames_dict:
            return

        # Sort frames by source id
        frames_list = [frames_dict[sid] for sid in sorted(frames_dict.keys())]
        n_sources = len(frames_list)
        if n_sources == 0:
            return

        # Arrange frames in a grid (square-ish)
        cols = int(math.ceil(math.sqrt(n_sources)))
        rows = int(math.ceil(n_sources / cols))

        # Window size
        window_width, window_height = 1280, 720
        frame_width = window_width // cols
        frame_height = window_height // rows

        combined_image = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n_sources:
                    break
                frame = frames_list[idx]
                frame_resized = cv2.resize(frame, (frame_width, frame_height))
                y1, x1 = r * frame_height, c * frame_width
                combined_image[y1:y1+frame_height, x1:x1+frame_width] = frame_resized
                idx += 1

        cv2.imshow("Synchronized Display", combined_image)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stopped = True
            cv2.destroyWindow("Synchronized Display")

    def stop(self):
        self.stopped = True

    def __del__(self):
        cv2.destroyAllWindows()


class PoseEstimatorWorker(multiprocessing.Process):
    '''
    Worker process that pulls frames from the frame_queue, runs pose estimation,
    and returns results.
    '''

    def __init__(
        self,
        frame_queue,
        available_buffers,
        shared_buffers,
        shared_counts,
        pose_tracker_settings,
        source_outputs,
        active_sources,
        multi_person,
        save_video,
        save_images,
        display_detection,
        output_format,
        vid_img_extension,
        fps,
        size,
        display_queue=None
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.available_buffers = available_buffers
        self.shared_buffers = shared_buffers
        self.shared_counts = shared_counts
        self.pose_tracker_settings = pose_tracker_settings
        self.source_outputs = source_outputs
        self.active_sources = active_sources
        self.display_queue = display_queue
        self.multi_person = multi_person
        self.save_video = save_video
        self.save_images = save_images
        self.display_detection = display_detection
        self.output_format = output_format
        self.vid_img_extension = vid_img_extension
        self.fps = fps
        self.W, self.H = size

        self.prev_keypoints_map = {}
        self.stopped = False

    def run(self):
        try:
            # Initialize pose tracker
            pose_tracker = init_pose_tracker(self.pose_tracker_settings)

            # Local storage for each source's video writer
            local_video_writers = {}

            while True:
                # If no new frames and no active sources remain, stop
                if self.frame_queue.empty() and self.active_sources.value == 0:
                    logging.info(f"PoseEstimatorWorker {self.pid} stopping.")
                    break

                try:
                    item = self.frame_queue.get(timeout=0.1)
                    if item is None:
                        break

                    # Unpack frame info
                    buffer_name, frame_shape, frame_dtype_str, sid, frame_idx, *others = item
                    timestamp = others[0] if others else None

                    # Convert shared memory to numpy
                    shm = self.shared_buffers[buffer_name]
                    frame_np = np.ndarray(frame_shape, dtype=np.dtype(frame_dtype_str), buffer=shm.buf)

                    # Prepare the video writer if needed
                    if self.save_video and sid not in local_video_writers:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_video_path = self.source_outputs[sid][4]
                        local_video_writers[sid] = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.W, self.H))

                    video_writer = local_video_writers.get(sid)
                    prev_kpts = self.prev_keypoints_map.get(sid)

                    # Process frame
                    (sid, final_frame), curr_kpts = process_frame(
                        self.vid_img_extension,
                        frame_np,
                        sid,
                        frame_idx,
                        self.source_outputs[sid],
                        pose_tracker,
                        self.multi_person,
                        self.save_video,
                        self.save_images,
                        self.display_detection,
                        self.output_format,
                        video_writer,
                        prev_kpts,
                        timestamp
                    )
                    self.prev_keypoints_map[sid] = curr_kpts

                    # Send frames for display if needed
                    if self.display_queue and self.display_detection and final_frame is not None:
                        self.display_queue.put((sid, frame_idx, final_frame))

                    # Increase count and return buffer
                    with self.shared_counts[sid]['processed'].get_lock():
                        self.shared_counts[sid]['processed'].value += 1
                    self.available_buffers.put(buffer_name)

                except queue.Empty:
                    pass

            # Release any local video writers
            for vw in local_video_writers.values():
                vw.release()

        except Exception as err:
            logging.error(f"PoseEstimatorWorker error: {err}")
            self.stopped = True


class CaptureCoordinator(multiprocessing.Process):
    '''
    Orchestrates reading frames from multiple media sources in a certain mode
    ('continuous' or 'alternating'), controlling the pace of capture.
    '''

    def __init__(self,
                 sources,
                 command_queues,
                 available_buffers,
                 shared_counts,
                 source_ended,
                 frame_limit=10,
                 mode='continuous'):
        super().__init__(daemon=True)
        self.sources = sources
        self.command_queues = command_queues
        self.available_buffers = available_buffers
        self.shared_counts = shared_counts
        self.source_ended = source_ended
        self.mode = mode

        self.min_interval = 1.0 / frame_limit if frame_limit > 0 else 0.0
        self.stopped = multiprocessing.Value('b', False)

    def run(self):
        last_capture_time = time.time()
        while not self.stopped.value:
            if all(self.source_ended[s['id']] for s in self.sources):
                break

            now = time.time()
            elapsed = now - last_capture_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            if self.mode == 'continuous':
                active_srcs = [s for s in self.sources if not self.source_ended[s['id']]]
                if self.available_buffers.qsize() >= len(active_srcs):
                    buffs = [self.available_buffers.get() for _ in active_srcs]
                    for i, src in enumerate(active_srcs):
                        sid = src['id']
                        self.command_queues[sid].put(("CAPTURE_FRAME", buffs[i]))
                    last_capture_time = time.time()
                else:
                    time.sleep(0.005)

            elif self.mode == 'alternating':
                chunk = self.available_buffers.qsize()
                if chunk <= 0:
                    time.sleep(0.005)
                    continue

                frames_sent = 0
                src_index = 0
                sources_requested = []

                while frames_sent < chunk and not all(self.source_ended[s['id']] for s in self.sources):
                    src = self.sources[src_index]
                    sid = src['id']
                    if not self.source_ended[sid]:
                        buf_name = self.available_buffers.get()
                        self.command_queues[sid].put(("CAPTURE_FRAME", buf_name))
                        sources_requested.append(sid)
                        frames_sent += 1
                    src_index = (src_index + 1) % len(self.sources)

                last_capture_time = time.time()
                if frames_sent > 0:
                    self.wait_until_processed(frames_sent, sources_requested)

            else:
                time.sleep(0.01)

        self.stop()

    def wait_until_processed(self, total_frames, sources_list):
        needed_counts = {}
        old_counts = {}

        for sid in sources_list:
            needed_counts[sid] = needed_counts.get(sid, 0) + 1

        for sid in needed_counts.keys():
            old_counts[sid] = self.shared_counts[sid]['processed'].value

        while not self.stopped.value:
            done = 0
            for sid, needed_val in needed_counts.items():
                current_processed = self.shared_counts[sid]['processed'].value
                if current_processed >= old_counts[sid] + needed_val:
                    done += needed_val
            if done >= total_frames:
                break
            time.sleep(0.01)

    def stop(self):
        self.stopped.value = True
        for src in self.sources:
            self.command_queues[src['id']].put(None)


class MediaSource(multiprocessing.Process):
    '''
    Reads frames from a single source: webcam, video file, or image directory.
    Sends frames to the frame queue when commanded by the CaptureCoordinator.
    '''

    def __init__(
        self,
        source,
        frame_queue,
        shared_buffers,
        shared_counts,
        frame_size,
        active_sources,
        command_queue,
        vid_img_extension,
        frame_ranges,
        webcam_ready=None,
        source_ended=None,
    ):
        super().__init__()
        self.source = source
        self.frame_queue = frame_queue
        self.shared_buffers = shared_buffers
        self.shared_counts = shared_counts
        self.frame_size = frame_size
        self.active_sources = active_sources
        self.command_queue = command_queue
        self.vid_img_extension = vid_img_extension
        self.frame_ranges = frame_ranges
        self.webcam_ready = webcam_ready
        self.source_ended = source_ended

        self.frame_idx = 0
        self.total_frames = 0
        self.cap = None
        self.image_files = []
        self.img_idx = 0
        self.stopped = False

    def run(self):
        try:
            # Open the specific source
            if self.source['type'] == 'webcam':
                self.open_webcam()
                self.shared_counts[self.source['id']]['total'].value = 0
                if self.webcam_ready is not None:
                    self.webcam_ready[self.source['id']] = True

            elif self.source['type'] == 'video':
                self.open_video()
                if self.frame_ranges:
                    self.total_frames = len(self.frame_ranges)
                else:
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.shared_counts[self.source['id']]['total'].value = self.total_frames

            elif self.source['type'] == 'images':
                self.load_images()
                self.shared_counts[self.source['id']]['total'].value = self.total_frames

            while not self.stopped:
                try:
                    cmd = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if cmd is None:
                    break

                if isinstance(cmd, tuple) and cmd[0] == "CAPTURE_FRAME":
                    buf_name = cmd[1]
                    # Stop if we've reached total frames for video/images

                    if self.source['type'] in ('video', 'images'):
                        if self.frame_idx >= self.total_frames:
                            self.stop_source()
                            break

                    # Read frame and send it
                    frame = self.read_frame()
                    if frame is None:
                        self.stop_source()
                        break

                    self.send_frame(frame, buf_name)
                else:
                    pass

            logging.info(f"MediaSource {self.source['id']} ended.")

            with self.active_sources.get_lock():
                self.active_sources.value -= 1

            if self.source_ended is not None:
                self.source_ended[self.source['id']] = True

        except Exception as err:
            logging.error(f"MediaSource {self.source['id']} error: {err}")
            self.stopped = True
            with self.active_sources.get_lock():
                self.active_sources.value -= 1
            if self.source_ended is not None:
                self.source_ended[self.source['id']] = True

    def open_webcam(self):
        self.cap = cv2.VideoCapture(int(self.source['id']), cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            logging.error(f"Unable to open webcam {self.source['id']}.")
            self.cap = None
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            logging.info(f"Webcam {self.source['id']} ready.")
        time.sleep(1)

    def open_video(self):
        self.cap = cv2.VideoCapture(self.source['path'])
        if not self.cap.isOpened():
            logging.error(f"Cannot open video file: {self.source['path']}")
            self.stopped = True

    def load_images(self):
        pattern = os.path.join(self.source['path'], f"*{self.extension}")
        self.image_files = sorted(glob.glob(pattern), key=natural_sort_key)
        self.total_frames = len(self.image_files)

    def read_frame(self):
        if self.source['type'] == 'video':
            if not self.cap:
                return None
            if self.frame_ranges and self.frame_idx not in self.frame_ranges:
                self.frame_idx += 1
                return self.read_frame()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.info(f"Video finished: {self.source['path']}")
                self.stopped = True
                return None
            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            return frame

        elif self.source['type'] == 'images':
            if self.img_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.img_idx])
                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
                self.img_idx += 1
                return frame
            else:
                self.stopped = True
                return None

        elif self.source['type'] == 'webcam':
            if not self.cap or not self.cap.isOpened():
                logging.warning(f"Webcam {self.source['id']} closed. Reopening...")
                self.open_webcam()
                if not self.cap or not self.cap.isOpened():
                    return None
            ret, frm = self.cap.read()
            if not ret or frm is None:
                logging.warning(f"Failed to read from webcam {self.source['id']}.")
                self.cap.release()
                self.cap = None
                return None
            return frm

        return None

    def send_frame(self, frame, buffer_name):
        shm = self.shared_buffers[buffer_name]
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np_frame[:] = frame

        timestamp = f"{time.time():.5f}"
        item = (buffer_name, frame.shape, frame.dtype.str, self.source['id'], self.frame_idx, timestamp)
        self.frame_queue.put(item)

        with self.shared_counts[self.source['id']]['queued'].get_lock():
            self.shared_counts[self.source['id']]['queued'].value += 1

        self.frame_idx += 1

    def stop_source(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

    def stop(self):
        self.stop_source()


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
    overwrite_pose = config_dict['pose'].get('overwrite_pose', False)

    # Check for existing results
    if os.path.exists(pose_dir) and not overwrite_pose:
        logging.info('Skipping pose estimation as it has already been done. '
                     'Set overwrite_pose to true in Config.toml if you want to run it again.')
        return
    elif overwrite_pose:
        logging.info("Overwriting existing pose estimation results...")

    logging.info("Starting pose estimation...")
    display_detection = config_dict['pose'].get('display_detection', False)
    vid_img_extension = config_dict['pose']['vid_img_extension']
    webcam_ids = config_dict['pose'].get('webcam_ids', [])
    capture_mode = config_dict['pose'].get('capture_mode', 'continuous')
    save_images = 'to_images' in config_dict['project'].get('save_video', [])

    # Gather sources
    sources = []
    if vid_img_extension == 'webcam':
        sources.extend({
            'type': 'webcam',
            'id': cam_id,
            'path': cam_id
        } for cam_id in (webcam_ids if isinstance(webcam_ids, list) else [webcam_ids]))
    else:
        video_files = sorted([str(f) for f in Path(source_dir).rglob('*' + vid_img_extension) if f.is_file()])
        sources.extend({
            'type': 'video',
            'id': idx,
            'path': video_path
        } for idx, video_path in enumerate(video_files))

        image_dirs = sorted([str(f) for f in Path(source_dir).iterdir() if f.is_dir()])
        sources.extend({
            'type': 'images',
            'id': idx,
            'path': folder
        } for idx, folder in enumerate(image_dirs, start=len(video_files)))

    if not sources:
        raise FileNotFoundError(f"\nNo sources found in {source_dir} matching extension '{vid_img_extension}'.")

    logging.info(f"Sources: {sources}")

    # Pose tracker settings
    pose_tracker_settings = determine_tracker_settings(config_dict)

    # Handle frame_range
    def parse_frame_ranges(frame_range):
        if not frame_range:
            return None
        # If user gave [start, end]
        if len(frame_range) == 2 and all(isinstance(x, int) for x in frame_range):
            start, end = frame_range
            return set(range(start, end + 1))
        return set(frame_range)

    frame_range = None
    if vid_img_extension != 'webcam':
        frame_range = parse_frame_ranges(config_dict['project'].get('frame_range', []))

    # Decide on input_size
    requested_size = config_dict['pose'].get('input_size', None)
    if not requested_size:
        fw, fh = find_largest_frame_size(sources, vid_img_extension)
        logging.info(f"Auto-detected largest frame size: {fw}x{fh}")
    else:
        fw, fh = requested_size
        logging.info(f"Using user-defined frame size: {fw}x{fh}")

    # Determine FPS if 'auto'
    frame_rate = config_dict['project'].get('frame_rate', 30)
    if str(frame_rate).lower() == 'auto':
        frame_rate = find_largest_fps(sources)
        logging.info(f"Auto-detected largest frame rate: {frame_rate} fps")

    # Prepare shared memory
    available_memory = psutil.virtual_memory().available
    frame_bytes = fw * fh * 3
    n_buffers = int((available_memory / 2) / frame_bytes)
    logging.info(f"Allocating {n_buffers} shared frame buffers...")

    frame_queue = multiprocessing.Queue(maxsize=n_buffers)
    shared_buffers = {}
    available_buffers = multiprocessing.Queue()

    for i in range(n_buffers):
        unique_name = f"frame_buffer_{i}"
        # Create the buffer
        shm = shared_memory.SharedMemory(name=unique_name, create=True, size=frame_bytes)
        shared_buffers[unique_name] = shm
        available_buffers.put(unique_name)

    # Prepare per-source counters
    source_outputs = {}
    shared_counts = {}
    for s in sources:
        shared_counts[s['id']] = {
            'queued': multiprocessing.Value('i', 0),
            'processed': multiprocessing.Value('i', 0),
            'total': multiprocessing.Value('i', 0)
        }

    # Real-time display
    display_queue = None
    sync_process = None
    if display_detection:
        display_queue = multiprocessing.Queue()

    if display_detection or vid_img_extension == 'webcam':
        sync_process = FrameSynchronizer(sources, display_queue)
        sync_process.start()

    active_sources = multiprocessing.Value('i', len(sources))
    manager = multiprocessing.Manager()
    webcam_ready = manager.dict()
    for s in sources:
        if s['type'] == 'webcam':
            webcam_ready[s['id']] = False

    source_ended = manager.dict()
    for s in sources:
        source_ended[s['id']] = False

    command_queues = {s['id']: manager.Queue() for s in sources}

    # Create media sources
    media_sources = []
    for s in sources:
        out_dirs = create_output_folders(s['path'], pose_dir, save_images)
        source_outputs[s['id']] = out_dirs

        ms = MediaSource(
            source=s,
            frame_queue=frame_queue,
            shared_buffers=shared_buffers,
            shared_counts=shared_counts,
            frame_size=(fw, fh),
            active_sources=active_sources,
            command_queue=command_queues[s['id']],
            vid_img_extension=vid_img_extension,
            frame_ranges=frame_range,
            webcam_ready=webcam_ready,
            source_ended=source_ended
        )
        ms.start()
        media_sources.append(ms)

    # Wait for webcams
    if vid_img_extension == 'webcam' and webcam_ids:
        while True:
            if all(webcam_ready[w] for w in webcam_ready.keys()):
                break
            time.sleep(0.1)
        logging.info("All webcams ready.")

    # Decide how many workers to start initially
    cpu_count = multiprocessing.cpu_count()
    
    if display_detection or vid_img_extension == 'webcam':
        initial_workers = max(1, cpu_count - len(sources) - 2)
    else:
        initial_workers = max(1, cpu_count - len(sources) - 1)

    if initial_workers < 1:
        initial_workers = 1

    logging.info(f"Starting {initial_workers} workers.")

    def spawn_new_worker():
        w = PoseEstimatorWorker(
            frame_queue,
            available_buffers,
            shared_buffers,
            shared_counts,
            pose_tracker_settings,
            source_outputs,
            active_sources,
            config_dict['project'].get('multi_person', False),
            ('to_video' in config_dict['project'].get('save_video', [])),
            save_images,
            display_detection,
            config_dict['project'].get('output_format', 'openpose'),
            vid_img_extension,
            frame_rate,
            (fw, fh),
            display_queue
        )
        w.start()
        return w

    workers = []
    for _ in range(initial_workers):
        workers.append(spawn_new_worker())

    # Start capture coordinator
    capture_coordinator = CaptureCoordinator(
        sources=sources,
        command_queues=command_queues,
        available_buffers=available_buffers,
        shared_counts=shared_counts,
        source_ended=source_ended,
        frame_limit=frame_rate if vid_img_extension == 'webcam' else 0,
        mode=capture_mode
    )
    capture_coordinator.start()

    # Setup progress bars
    progress_bars = {}
    bar_ended_state = {}
    bar_pos = 0

    for s in sources:
        sid = s['id']
        if s['type'] != 'webcam':
            pb = tqdm(
                total=0,
                desc=f"\033[32mSource {sid}\033[0m",
                position=bar_pos,
                leave=True
            )
            progress_bars[sid] = pb
            bar_ended_state[sid] = False
            bar_pos += 1

    buffer_bar = tqdm(
        total=n_buffers,
        desc='Buffers Free',
        position=bar_pos,
        leave=True,
        colour='blue'
    )
    bar_pos += 1

    worker_bar = tqdm(
        total=len(workers),
        desc='Active Workers',
        position=bar_pos,
        leave=True,
        colour='blue'
    )
    bar_pos += 1

    previous_ended_count = 0
    try:
        while True:
            # Update progress bars
            for sid, pb in progress_bars.items():
                cnts = shared_counts[sid]
                tval = cnts['total'].value
                pval = cnts['processed'].value
                if tval > 0:
                    pb.total = tval
                    pb.n = pval
                    pb.refresh()

                if source_ended[sid] and not bar_ended_state[sid]:
                    bar_ended_state[sid] = True
                    pb.set_description_str(f"\033[31mSource {sid} (Ended)\033[0m")
                    pb.refresh()

            buffer_bar.n = available_buffers.qsize()
            buffer_bar.refresh()

            alive_workers = sum(w.is_alive() for w in workers)
            worker_bar.n = alive_workers
            worker_bar.refresh()

            # Check if user closed the display
            if sync_process and sync_process.stopped:
                logging.info("\nUser closed display. Stopping all streams.")
                capture_coordinator.stop()
                for ms_proc in media_sources:
                    ms_proc.stop_source()
                break

            # Possibly spawn new worker(s) if new sources ended
            current_ended_count = sum(1 for s in sources if source_ended[s['id']])
            ended_delta = current_ended_count - previous_ended_count
            if ended_delta > 0:
                for _ in range(ended_delta):
                    logging.info("Spawning a new PoseEstimatorWorker.")
                    new_w = spawn_new_worker()
                    workers.append(new_w)
                    worker_bar.total = len(workers)
                previous_ended_count = current_ended_count

            # If all sources ended, queue empty, and no alive workers => done
            all_ended = all(source_ended[s['id']] for s in sources)
            if all_ended and frame_queue.empty() and alive_workers == 0:
                logging.info("All sources ended, queue empty, and workers finished. Exiting loop.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("User interrupted pose estimation.")
        capture_coordinator.stop()
        for ms_proc in media_sources:
            ms_proc.stop_source()

    finally:
        # Stop capture coordinator
        capture_coordinator.stop()
        capture_coordinator.join(timeout=2)
        if capture_coordinator.is_alive():
            capture_coordinator.terminate()

        # Stop media sources
        for s in sources:
            command_queues[s['id']].put(None)
        for rp in media_sources:
            rp.join(timeout=2)
            if rp.is_alive():
                rp.terminate()

        # Stop workers
        for w in workers:
            w.join(timeout=2)
            if w.is_alive():
                logging.warning(f"Forcibly terminating worker {w.pid}")
                w.terminate()

        # Free shared memory
        for shm in shared_buffers.values():
            shm.close()
            shm.unlink()

        # Stop sync display
        if sync_process:
            sync_process.stop()
            sync_process.join(timeout=2)
            if sync_process.is_alive():
                sync_process.terminate()

        # Final bar updates
        buffer_bar.n = available_buffers.qsize()
        buffer_bar.refresh()

        for sid, pb in progress_bars.items():
            pb.close()
        buffer_bar.close()
        worker_bar.close()

        logging.info("Pose estimation done. Exiting now.")
        logging.shutdown()

        if display_detection:
            cv2.destroyAllWindows()


def create_output_folders(source_path, output_dir, save_images):
    '''
    Set up output directories for saving images and JSON files.

    Returns:
        tuple: (output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path)
    '''

    if isinstance(source_path, int):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_name = f"webcam{source_path}_{now}"
    else:
        output_dir_name = os.path.basename(os.path.splitext(str(source_path))[0])

    os.makedirs(output_dir, exist_ok=True)
    img_output_dir = os.path.join(output_dir, f"{output_dir_name}_img")
    json_output_dir = os.path.join(output_dir, f"{output_dir_name}_json")

    if save_images:
        os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    output_video_path = os.path.join(output_dir, f"{output_dir_name}_pose.mp4")
    return output_dir, output_dir_name, img_output_dir, json_output_dir, output_video_path


def determine_tracker_settings(config_dict):
    det_frequency = config_dict['pose']['det_frequency']
    mode = config_dict['pose']['mode']
    pose_model = config_dict['pose']['pose_model']
    backend = config_dict['pose']['backend']
    device = config_dict['pose']['device']

    # Select the appropriate model based on the model_type
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    elif pose_model.upper() == 'HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info(f"Using FACE_106 model for pose estimation.")
    elif pose_model.upper() == 'ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info(f"Using ANIMAL2D_17 model for pose estimation.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation.")
    pose_model_name = pose_model
    try:
        pose_model = eval(model_name)
    except:
        raise ValueError(f"Pose model '{model_name}' not supported yet.")

    # Select device and backend
    backend, device = init_backend_device(backend=backend, device=device)

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
            try:
                mode = ast.literal_eval(mode)
            except:  # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/', ':/').replace('":"\\', ':\\')
                mode = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', mode) # changes "[640", "640]" to [640,640]
                mode = json.loads(mode)
            det_class = mode.get('det_class')
            det = mode.get('det_model')
            det_input_size = mode.get('det_input_size')
            pose_class = mode.get('pose_class')
            pose = mode.get('pose_model')
            pose_input_size = mode.get('pose_input_size')

            ModelClass = partial(Custom,
                                 det_class=det_class, det=det, det_input_size=det_input_size,
                                 pose_class=pose_class, pose=pose, pose_input_size=pose_input_size,
                                 backend=backend, device=device)

        except (json.JSONDecodeError, TypeError):
            logging.warning("\nInvalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'

    logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
    logging.info(f'Mode: {mode}.')

    return (ModelClass, det_frequency, mode, False, backend, device)


def find_largest_frame_size(sources, vid_img_extension):
    '''
    If input_size is not specified, find the maximum (width, height) 
    among all sources (videos, images, webcams).
    '''
    max_w, max_h = 0, 0
    for s in sources:
        # Handle each source type
        if s['type'] == 'webcam':
            cap = cv2.VideoCapture(int(s['id']))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)
            cap.release()

        elif s['type'] == 'video':
            cap = cv2.VideoCapture(s['path'])
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                max_w = max(max_w, width)
                max_h = max(max_h, height)
            cap.release()

        elif s['type'] == 'images':
            all_exts = ("*.jpg", "*.png", "*.jpeg", "*.bmp")
            found = []
            for ext in all_exts:
                found.extend(glob.glob(os.path.join(s['path'], ext)))
            found = sorted(found, key=natural_sort_key)
            if found:
                # Just read the first image for dimension
                im = cv2.imread(found[0])
                if im is not None:
                    h, w = im.shape[:2]
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)

    # If none found, default to (640,480)
    if max_w == 0 or max_h == 0:
        max_w, max_h = 640, 480
    return max_w, max_h


def find_largest_fps(sources):
    '''
    Auto-detect the largest FPS among all video/webcam sources.
    If none found or invalid, default to 30.
    '''
    max_fps = 0
    for s in sources:
        if s['type'] in ('video', 'webcam'):
            if s['type'] == 'webcam':
                cap = cv2.VideoCapture(int(s['id']))
            else:
                cap = cv2.VideoCapture(s['path'])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                # Some webcams give 0 or -1
                if fps <= 0 or math.isnan(fps):
                    fps = 30
                max_fps = max(max_fps, fps)
            cap.release()
    return int(math.ceil(max_fps)) if max_fps > 0 else 30
