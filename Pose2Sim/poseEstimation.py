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
    need another detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and in-between tracks points (faster but less accurate).

    If a valid CUDA installation is detected, uses the GPU with the ONNXRuntime backend.
    Otherwise, uses the CPU with the OpenVINO backend.

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
def get_formatted_timestamp():
    dt = datetime.now()
    ms = dt.microsecond // 1000
    return dt.strftime("%Y%m%d_%H%M%S_") + f"{ms:03d}"


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
        display_queue,
        output_format,
        save_images,
        vid_img_extension
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
        self.output_format = output_format
        self.save_images = save_images
        self.vid_img_extension = vid_img_extension

        self.prev_keypoints_map = {}
        self.stopped = False

    def run(self):
        try:
            # Initialize pose tracker
            pose_tracker = init_pose_tracker(self.pose_tracker_settings)

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
                    buffer_name, frame_shape, frame_dtype_str, sid, frame_idx, is_placeholder, timestamp = item

                    # Convert shared memory to numpy
                    shm = self.shared_buffers[buffer_name]
                    frame_np = np.ndarray(frame_shape, dtype=np.dtype(frame_dtype_str), buffer=shm.buf)

                    if is_placeholder:
                        if self.display_queue:
                            self.display_queue.put((sid, frame_idx, frame_np, True))
                        self.available_buffers.put(buffer_name)
                        continue

                    # Pose
                    keypoints, scores = pose_tracker(frame_np)

                    # Tri multi-person
                    previous_keypoints = self.prev_keypoints_map.get(sid)
                    if self.multi_person:
                        if previous_keypoints is None:
                            previous_keypoints = keypoints
                        previous_keypoints, keypoints, scores = sort_people_sports2d(previous_keypoints, keypoints, scores=scores)
                        self.prev_keypoints_map[sid] = previous_keypoints

                    out_dirs = self.source_outputs[sid]
                    (output_dir_name, img_output_dir, json_output_dir, _ ) = out_dirs
                    if self.vid_img_extension == 'webcam':
                        file_name = f"{output_dir_name}_{timestamp}"
                    else:
                        file_name = f"{output_dir_name}_{frame_idx:06d}"

                    if 'openpose' in self.output_format:
                        json_file_name = f"{file_name}.json"
                        json_path = os.path.join(json_output_dir, json_file_name)
                        save_keypoints_to_openpose(json_path, keypoints, scores)

                    if self.save_images:
                        img_file_name = f"{file_name}.jpg"
                        cv2.imwrite(os.path.join(img_output_dir, img_file_name), frame_np)

                    # Draw skeleton if needed
                    if self.display_queue:
                        annotated_frame = draw_skeleton(frame_np, keypoints, scores, kpt_thr=0.1)
                        self.display_queue.put((sid, frame_idx, annotated_frame, False))

                    # Increase count and return buffer
                    with self.shared_counts[sid]['processed'].get_lock():
                        self.shared_counts[sid]['processed'].value += 1
                    self.available_buffers.put(buffer_name)

                except queue.Empty:
                    pass

        except Exception as err:
            logging.error(f"PoseEstimatorWorker error: {err}")
            self.stopped = True


class FrameSynchronizer(multiprocessing.Process):
    '''
    Affichage unifié, deux modes possibles :
      - mode webcams  => live (placeholder "Not connected")
      - mode vidéos/images => synchro par frame_idx
    Jamais un mélange des deux en même temps.

    On enregistre la vidéo si besoin (self.save_video).
    '''

    def __init__(self,
                 sources,
                 display_queue,
                 vid_img_extension,
                 save_video=False,
                 source_outputs=None,
                 frame_width=1280,
                 frame_height=720,
                 fps=30,
                 orientation='landscape'):
        super().__init__(daemon=True)
        self.sources = sources
        self.display_queue = display_queue
        self.vid_img_extension = vid_img_extension
        self.save_video = save_video
        self.source_outputs = source_outputs

        self.last_frames   = {}
        self.placeholder_map = {}

        self.sync_data = {}

        self.stopped = False
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.orientation = orientation

        self.video_writers = {}

    def run(self):
        if self.save_video and self.source_outputs:
            for s in self.sources:
                sid = s['id']
                out_dirs = self.source_outputs[sid]
                output_video_path = out_dirs[-1]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writers[sid] = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    self.fps,
                    (self.frame_width, self.frame_height)
                )

        while not self.stopped:
            try:
                sid, frame_idx, frame, is_placeholder = self.display_queue.get(timeout=0.1)
                if frame is None:
                    continue

                if self.vid_img_extension == 'webcam':
                    self.last_frames[sid] = frame
                    self.placeholder_map[sid] = is_placeholder
                else:
                    if frame_idx not in self.sync_data:
                        self.sync_data[frame_idx] = {}
                    self.sync_data[frame_idx][sid] = frame

                if not is_placeholder and sid in self.video_writers:
                    frm_resized = self.safe_resize(frame, self.frame_width, self.frame_height)
                    self.video_writers[sid].write(frm_resized)

                self.show_unified()

            except queue.Empty:
                pass

        for vw in self.video_writers.values():
            vw.release()
        cv2.destroyAllWindows()

    def safe_resize(self, frame, target_w, target_h):
        h, w = frame.shape[:2]
        if w <= target_w and h <= target_h:
            return frame
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def show_unified(self):
        frames_list = self.get_frames_list()

        if not frames_list:
            return

        mosaic = self.build_mosaic(frames_list)

        cv2.imshow("Display", mosaic)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stopped = True

    def get_frames_list(self):
        if self.vid_img_extension == 'webcam':
            frames_list = []
            for s in self.sources:
                sid = s['id']
                frm = self.last_frames.get(sid, None)
                if frm is None:
                    black = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
                    cv2.putText(black, "No frames yet", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
                    frames_list.append(black)
                else:
                    if self.placeholder_map.get(sid, False):
                        frm_small = self.safe_resize(frm, self.frame_width, self.frame_height)
                        cv2.putText(frm_small, "Not connected", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
                        frames_list.append(frm_small)
                    else:
                        frames_list.append(frm)
            return frames_list
        else:
            complete_idxs = []
            for f_idx, frames_dict in self.sync_data.items():
                if len(frames_dict) == len(self.sources):
                    complete_idxs.append(f_idx)
            if not complete_idxs:
                return []

            min_idx = min(complete_idxs)
            frames_dict = self.sync_data[min_idx]
            del self.sync_data[min_idx]

            frames_list = []
            for s in self.sources:
                sid = s['id']
                frames_list.append(frames_dict[sid])
            return frames_list

    def build_mosaic(self, frames_list):
        n = len(frames_list)
        if n == 0:
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        if self.orientation == 'portrait':
            window_w, window_h = 720, 1280
        else:
            window_w, window_h = 1280, 720

        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        sub_w = window_w // cols
        sub_h = window_h // rows

        mosaic = np.zeros((window_h, window_w, 3), dtype=np.uint8)

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                frm = frames_list[idx]
                frm_small = self.safe_resize(frm, sub_w, sub_h)

                h2, w2 = frm_small.shape[:2]
                if w2 != sub_w or h2 != sub_h:
                    frm_small = cv2.resize(frm_small, (sub_w, sub_h))

                y1, x1 = r*sub_h, c*sub_w
                mosaic[y1:y1+sub_h, x1:x1+sub_w] = frm_small
                idx += 1

        return mosaic

    def stop(self):
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
                 webcam_ready,
                 frame_limit=10,
                 mode='continuous'):
        super().__init__(daemon=True)
        self.sources = sources
        self.command_queues = command_queues
        self.available_buffers = available_buffers
        self.shared_counts = shared_counts
        self.source_ended = source_ended
        self.webcam_ready = webcam_ready
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

            def is_ready_source(s):
                sid = s['id']
                if self.source_ended[sid]:
                    return False
                if s['type'] == 'webcam':
                    return bool(self.webcam_ready[sid])
                return True

            ready_srcs = [s for s in self.sources if is_ready_source(s)]

            if self.mode == 'continuous':
                if self.available_buffers.qsize() >= len(ready_srcs):
                    buffs = [self.available_buffers.get() for _ in ready_srcs]
                    for i, src in enumerate(ready_srcs):
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

                ready_count = len(ready_srcs)
                while frames_sent < chunk and not all(self.source_ended[s['id']] for s in self.sources):
                    if ready_count == 0:
                        break
                    src = ready_srcs[src_index]
                    sid = src['id']
                    buf_name = self.available_buffers.get()
                    self.command_queues[sid].put(("CAPTURE_FRAME", buf_name))
                    sources_requested.append(sid)
                    frames_sent += 1

                    src_index = (src_index + 1) % ready_count

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
        orientation='landscape'
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
        self.orientation = orientation

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
                    self.webcam_ready[self.source['id']] = (self.cap is not None and self.cap.isOpened())

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

                if isinstance(cmd, tuple):
                    if cmd[0] == "CAPTURE_FRAME":
                        buf_name = cmd[1]
                        # Stop if we've reached total frames for video/images
                        if self.source['type'] in ('video', 'images'):
                            if self.frame_idx >= self.total_frames:
                                self.stop_source()
                                break

                        # Read frame and send it
                        frame, is_placeholder = self.read_frame()
                        if frame is None:
                            self.stop_source()
                            break

                        self.send_frame(frame, buf_name, is_placeholder)

                    elif cmd[0] == "STOP_CAPTURE":
                        self.stop_source()
                        self.stopped = True
                        break
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
            logging.warning(f"Unable to open webcam {self.source['id']}.")
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
        pattern = os.path.join(self.source['path'], f"*{self.vid_img_extension}")
        self.image_files = sorted(glob.glob(pattern), key=natural_sort_key)
        self.total_frames = len(self.image_files)

    def read_frame(self):
        if self.source['type'] == 'video':
            if not self.cap:
                return None, False
            if self.frame_ranges and self.frame_idx not in self.frame_ranges:
                self.frame_idx += 1
                return self.read_frame()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.info(f"Video finished: {self.source['path']}")
                self.stopped = True
                return None, False

            if self.orientation == 'portrait':
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
            return frame, False

        elif self.source['type'] == 'images':
            if self.img_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.img_idx])
                if frame is None:
                    return None, False

                if self.orientation == 'portrait':
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
                self.img_idx += 1
                return frame, False
            else:
                self.stopped = True
                return None, False

        elif self.source['type'] == 'webcam':
            if self.cap is None or not self.cap.isOpened():
                if self.webcam_ready is not None:
                    self.webcam_ready[self.source['id']] = False

                self.open_webcam()

                if self.cap is None or not self.cap.isOpened():
                    placeholder_frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                    return placeholder_frame, True

            ret, frm = self.cap.read()
            if not ret or frm is None:
                logging.warning(f"Failed to read from webcam {self.source['id']}. Reconnecting loop.")
                self.cap.release()
                self.cap = None
                placeholder_frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                return placeholder_frame, True

            if self.webcam_ready is not None:
                self.webcam_ready[self.source['id']] = True

            if self.orientation == 'portrait':
                frm = cv2.rotate(frm, cv2.ROTATE_90_CLOCKWISE)

            frm = cv2.resize(frm, self.frame_size, interpolation=cv2.INTER_AREA)
            return frm, False

        return None, False

    def send_frame(self, frame, buffer_name, is_placeholder):
        shm = self.shared_buffers[buffer_name]
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np_frame[:] = frame

        timestamp = get_formatted_timestamp()

        item = (buffer_name, frame.shape, frame.dtype.str, self.source['id'], self.frame_idx, is_placeholder, timestamp)
        self.frame_queue.put(item)

        if not is_placeholder:
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
    save_files = config_dict['pose'].get('save_video', [])
    save_images = ('to_images' in save_files)
    save_video = ('to_video' in save_files)
    orientation = config_dict['pose'].get('orientation', 'landscape')

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
        frame_rate = find_lowest_fps(sources)
        logging.info(f"Auto-detected lowest frame rate: {frame_rate} fps")
    else:
        logging.info(f"Using user-defined frame rate: {frame_rate} fps")

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

    active_sources = multiprocessing.Value('i', len(sources))
    manager = multiprocessing.Manager()
    webcam_ready = manager.dict()
    source_ended = manager.dict()
    command_queues = {}

    media_sources = []
    for s in sources:
        source_ended[s['id']] = False
        command_queues[s['id']] = manager.Queue()
        out_dirs = create_output_folders(s['path'], pose_dir, save_images)
        source_outputs[s['id']] = out_dirs

        if s['type'] == 'webcam':
            webcam_ready[s['id']] = False

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
            source_ended=source_ended,
            orientation=orientation
        )
        ms.start()
        media_sources.append(ms)

    # Decide how many workers to start
    cpu_count = multiprocessing.cpu_count()

    # Real-time display
    display_queue = None
    sync_process = None
    if display_detection or vid_img_extension == 'webcam' or save_video:
        display_queue = multiprocessing.Queue()
        sync_process = FrameSynchronizer(
            sources=sources,
            display_queue=display_queue,
            vid_img_extension=vid_img_extension,
            save_video=save_video,
            source_outputs=source_outputs,
            frame_width=fw,
            frame_height=fh,
            fps=frame_rate,
            orientation=orientation
        )
        sync_process.start()
        initial_workers = max(1, cpu_count - len(sources) - 2)
    else:
        initial_workers = max(1, cpu_count - len(sources) - 1)

    logging.info(f"Starting {initial_workers} workers.")

    def spawn_new_worker():
        w = PoseEstimatorWorker(
            frame_queue=frame_queue,
            available_buffers=available_buffers,
            shared_buffers=shared_buffers,
            shared_counts=shared_counts,
            pose_tracker_settings=pose_tracker_settings,
            source_outputs=source_outputs,
            active_sources=active_sources,
            multi_person=config_dict['project'].get('multi_person', False),
            display_queue=display_queue,
            output_format=config_dict['project'].get('output_format', 'openpose'),
            save_images=save_images,
            vid_img_extension=vid_img_extension
        )
        w.start()
        return w

    workers = [spawn_new_worker() for _ in range(initial_workers)]

    # Start capture coordinator
    capture_coordinator = CaptureCoordinator(
        sources=sources,
        command_queues=command_queues,
        available_buffers=available_buffers,
        shared_counts=shared_counts,
        source_ended=source_ended,
        webcam_ready=webcam_ready,
        frame_limit=frame_rate if vid_img_extension == 'webcam' else 0,
        mode=capture_mode
    )
    capture_coordinator.start()

    # Setup progress bars
    progress_bars = {}
    bar_ended_state = {}
    bar_pos = 0

    source_type_map = {s['id']: s['type'] for s in sources}

    if vid_img_extension == 'webcam':
        cv2.namedWindow("StopWindow", cv2.WINDOW_NORMAL)

    for s in sources:
        sid = s['id']
        if s['type'] == 'webcam':
            pb = tqdm(
                total=0,
                desc=f"\033[32mWebcam {sid} (not connected)\033[0m",
                position=bar_pos,
                leave=True,
                bar_format="{desc}"
            )
            progress_bars[sid] = pb
            bar_ended_state[sid] = False
        else:
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
                s_type = source_type_map[sid]
                cnts = shared_counts[sid]
                pval = cnts['processed'].value
                qval = cnts['queued'].value
                tval = cnts['total'].value

                if s_type == 'webcam':
                    connected = webcam_ready[sid]
                    if connected:
                        pb.set_description_str(
                            f"\033[32mWebcam {sid}\033[0m : {pval}/{qval} processed/read"
                        )
                    else:
                        pb.set_description_str(
                            f"\033[31mWebcam {sid} (not connected)\033[0m : {pval}/{qval}"
                        )
                    pb.refresh()

                    if source_ended[sid] and not bar_ended_state[sid]:
                        bar_ended_state[sid] = True
                        pb.set_description_str(
                            f"\033[31mWebcam {sid} (Ended)\033[0m : {pval}/{qval}"
                        )
                        pb.refresh()

                    if check_stop_window_open():
                        stop_all_cameras_immediately(sources, command_queues)
                        break

                else:
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

    output_video_path = os.path.join(output_dir, f"{output_dir_name}_pose.avi")
    return output_dir_name, img_output_dir, json_output_dir, output_video_path


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
        logging.info("Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info("Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info("Using COCO_17 model (body) for pose estimation.")
    elif pose_model.upper() == 'HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info("Using HAND_21 model for pose estimation.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info("Using FACE_106 model for pose estimation.")
    elif pose_model.upper() == 'ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info("Using ANIMAL2D_17 model for pose estimation.")
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
            cap_test = cv2.VideoCapture(int(s['id']), cv2.CAP_DSHOW)
            if cap_test.isOpened():
                cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                time.sleep(0.2)
                ret, frame_test = cap_test.read()
                if ret and frame_test is not None:
                    h, w = frame_test.shape[:2]
                    max_w = max(max_w, w)
                    max_h = max(max_h, h)
            cap_test.release()

        elif s['type'] == 'video':
            cap_test = cv2.VideoCapture(s['path'])
            if cap_test.isOpened():
                w = int(cap_test.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT))
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            cap_test.release()

        elif s['type'] == 'images':
            pattern = os.path.join(sources['path'], f"*{vid_img_extension}")
            found = sorted(glob.glob(pattern), key=natural_sort_key)
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


def measure_webcam_fps(cam_index, warmup_frames=20, measure_frames=50):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return 30
    
    for _ in range(warmup_frames):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            return 30
    
    import time
    start = time.time()
    count = 0
    for _ in range(measure_frames):
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    
    end = time.time()
    cap.release()
    if count < 1:
        return 30

    fps_measured = count / (end - start)
    return fps_measured


def find_lowest_fps(sources):
    '''
    Auto-detect the largest FPS among all video/webcam sources.
    If none found or invalid, default to 30.
    '''

    min_fps = 9999

    for s in sources:
        if s['type'] == 'video':
            cap = cv2.VideoCapture(s['path'])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or math.isnan(fps):
                    fps = 30
                min_fps = min(min_fps, fps)
            cap.release()

        elif s['type'] == 'webcam':
            fps = measure_webcam_fps(int(s['id']))
            min_fps = min(min_fps, fps)
    
    return int(math.ceil(min_fps)) if min_fps < 9999 else 30


def check_stop_window_open():
    if cv2.getWindowProperty("StopWindow", cv2.WND_PROP_VISIBLE) < 1:
        return True
    key = cv2.waitKey(1)
    if key == 27:
        return True
    return False


def stop_all_cameras_immediately(sources, command_queues):
    for s in sources:
        if s['type'] == 'webcam':
            command_queues[s['id']].put(("STOP_CAPTURE", None))
