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
from anytree.importer import DictImporter

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton
from Pose2Sim.common import natural_sort_key, sort_people_sports2d, colors, thickness, draw_bounding_box, draw_keypts, draw_skel
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


def safe_resize(frame, desired_w, desired_h):
    h, w = frame.shape[:2]
    scale = min(desired_w / w, desired_h / h)
    if scale != 0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame

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

    ModelClass, det_frequency, mode, tracking, backend, device, *others = pose_tracker_settings
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
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.pose_tracker = None
        self.stopped = False

    def run(self):
        # Initialize pose tracker
        self.pose_tracker = init_pose_tracker(self.pose_tracker_settings)
        while not self.stopped:
            # If no new frames and no active sources remain, stop
            if self.queue.empty() and self.active_sources.value == 0:
                logging.info("Stopping worker as no active sources or items are in the queue.")
                self.stop()
                break

            try:
                item = self.queue.get_nowait()
                self.process_frame(item)
            except queue.Empty:
                time.sleep(0.005)

    def process_frame(self, item):
        # Unpack frame info
        buffer_name, idx, frame_shape, frame_dtype, *others = item
        # Convert shared memory to numpy
        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=self.buffers[buffer_name].buf)
        
        keypoints, scores = self.pose_tracker(frame) if not others[1] else (None, None)

        result = (buffer_name, idx, frame_shape, frame_dtype, others[0], others[1], keypoints, scores)
        self.result_queue.put(result)

    def stop(self):
        self.stopped = True


class BaseSynchronizer:
    def __init__(self,
                 sources,
                 frame_buffers,
                 pose_buffers,
                 available_frame_buffers,
                 available_pose_buffers,
                 vid_img_extension,
                 source_outputs,
                 shared_counts,
                 save_images,
                 save_video,
                 frame_size,
                 frame_rate,
                 orientation,
                 combined_frames,
                 multi_person,
                 output_format,
                 display_detection):
        self.sources = sources
        self.frame_buffers = frame_buffers
        self.pose_buffers = pose_buffers
        self.available_frame_buffers = available_frame_buffers
        self.available_pose_buffers = available_pose_buffers

        self.vid_img_extension = vid_img_extension
        self.source_outputs = source_outputs
        self.shared_counts = shared_counts
        self.save_images = save_images
        self.save_video = save_video
        self.frame_size = frame_size
        self.frame_rate = frame_rate
        self.orientation = orientation
        self.combined_frames = combined_frames
        self.multi_person = multi_person
        self.output_format = output_format
        self.display_detection = display_detection

        self.sync_data = {}
        self.placeholder_map = {}

        self.video_writers = {}
        self.stopped = False

    def get_frames_list(self):
        if not self.sync_data:
            return None

        for idx in sorted(self.sync_data.keys()):
            group = self.sync_data[idx]
            if len(group) == len(self.sources):
                complete_group = self.sync_data.pop(idx)
                frames_list = []
                for source in self.sources:
                    sid = source['id']
                    buffer_name, frame_shape, frame_dtype, keypoints, scores, is_placeholder = complete_group[sid]
                    frame = np.ndarray(frame_shape, dtype=np.dtype(frame_dtype), buffer=self.frame_buffers[buffer_name].buf).copy()
                    self.available_frame_buffers.put(buffer_name)
                    orig_w, orig_h = frame_shape[1], frame_shape[0]
                    if is_placeholder:
                        cv2.putText(frame, "Not connected", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    frames_list.append((frame, sid, orig_w, orig_h, keypoints, scores))
                return frames_list
        return None

    def build_mosaic(self, frames_list):
        target_w, target_h = self.frame_size
        if not frames_list:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8), {}
        n = len(frames_list)
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        mosaic_h = rows * target_h
        mosaic_w = cols * target_w
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        subinfo = {}
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                frame, *others = frames_list[idx]
                x_off = c * target_w
                y_off = r * target_h
                mosaic[y_off:y_off+target_h, x_off:x_off+target_w] = frame
                if others is not None:
                    subinfo[others[0]] = {
                        "x_offset": x_off,
                        "y_offset": y_off,
                        "scaled_w": target_w,
                        "scaled_h": target_h,
                        "orig_w": others[1],
                        "orig_h": others[2]
                    }
                idx += 1
        return mosaic, subinfo

    def read_mosaic(self, mosaic_np, subinfo, keypoints, scores):
        frames = {}
        recovered_keypoints = {}
        recovered_scores = {}

        for s in self.sources:
            sid = s['id']
            if sid not in subinfo:
                continue
            info = subinfo[sid]
            x_off = info["x_offset"]
            y_off = info["y_offset"]
            sc_w  = info["scaled_w"]
            sc_h  = info["scaled_h"]
            orig_w = info["orig_w"]
            orig_h = info["orig_h"]

            frame_region = mosaic_np[y_off:y_off+sc_h, x_off:x_off+sc_w].copy()
            frames[sid] = frame_region

            rec_kpts = []
            rec_scores = []
            for p in range(len(keypoints)):
                kp_person = keypoints[p]
                sc_person = scores[p]

                center_x = np.mean([x for (x, y) in kp_person])
                center_y = np.mean([y for (x, y) in kp_person])

                if x_off <= center_x <= x_off + sc_w and y_off <= center_y <= y_off + sc_h:
                    local_kpts = []
                    local_scores = []
                    
                    for (xk, yk), scv in zip(kp_person, sc_person):
                        x_local = (xk - x_off) * (orig_w / float(sc_w))
                        y_local = (yk - y_off) * (orig_h / float(sc_h))
                        local_kpts.append([x_local, y_local])
                        local_scores.append(scv)
                    rec_kpts.append(np.array(local_kpts))
                    rec_scores.append(np.array(local_scores))
            recovered_keypoints[sid] = rec_kpts
            recovered_scores[sid] = rec_scores

        return frames, recovered_keypoints, recovered_scores

    def stop(self):
        self.stopped = True


class FrameQueueProcessor(multiprocessing.Process, BaseSynchronizer):
    def __init__(self, frame_queue, pose_queue, **kwargs):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, **kwargs)
        self.frame_queue = frame_queue
        self.pose_queue = pose_queue

    def run(self):
        while not self.stopped:
            try:
                buffer_name, idx, frame_shape, frame_dtype, sid, is_placeholder = self.frame_queue.get_nowait()
                if idx not in self.sync_data:
                    self.sync_data[idx] = {}
                self.sync_data[idx][sid] = (buffer_name, frame_shape, frame_dtype, None, None, is_placeholder)
                
                if frames_list := self.get_frames_list():
                    mosaic, subinfo = self.build_mosaic(frames_list)
                    pose_buf_name = self.available_pose_buffers.get_nowait()
                    np.ndarray(mosaic.shape, dtype=mosaic.dtype, buffer=self.pose_buffers[pose_buf_name].buf)[:] = mosaic
                    self.pose_queue.put((pose_buf_name, idx, mosaic.shape, mosaic.dtype.str, subinfo, False))
            except queue.Empty:
                time.sleep(0.005)


class ResultQueueProcessor(multiprocessing.Process, BaseSynchronizer):
    def __init__(self, result_queue, **kwargs):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, **kwargs)
        self.result_queue = result_queue
        self.prev_keypoints = {}

    def run(self):
        self.init_video_writers()

        while not self.stopped:
            try:
                result_item = self.result_queue.get_nowait()
                self.process_result_item(result_item)
            except queue.Empty:
                time.sleep(0.005)

        for vw in self.video_writers.values():
            vw.release()
        cv2.destroyAllWindows()

    def init_video_writers(self):
        if self.save_video and self.source_outputs:
            for s in self.sources:
                sid = s['id']
                out_dirs = self.source_outputs[sid]
                output_video_path = out_dirs[-1]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writers[sid] = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    self.frame_rate,
                    (self.frame_size[0], self.frame_size[1])
                )

    def process_result_item(self, result_item):
        buffer_name, idx, frame_shape, frame_dtype, info, is_placeholder, keypoints, scores = result_item
        if self.combined_frames:
            self.handle_combined_frames(buffer_name, frame_shape, frame_dtype, keypoints, scores, info, idx)
        else:
            self.handle_individual_frames(buffer_name, frame_shape, frame_dtype, info, is_placeholder, keypoints, scores, idx)

    def handle_combined_frames(self, buffer_name, frame_shape, frame_dtype, keypoints, scores, subinfo, idx):
        shm = self.pose_buffers[buffer_name]
        mosaic = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)

        if self.save_images or self.save_video or self.display_detection:
            mosaic = draw_skeleton(mosaic, keypoints, scores)
            if self.display_detection:
                self.show_mosaic(mosaic)

        frames, recovered_keypoints, recovered_scores = self.read_mosaic(mosaic, subinfo, keypoints, scores)
        self.available_pose_buffers.put(buffer_name)

        for sid in frames:
            self.handle_output(sid, frames[sid], recovered_keypoints[sid], recovered_scores[sid], idx)

    def handle_individual_frames(self, buffer_name, frame_shape, frame_dtype, sid, is_placeholder, keypoints, scores, idx):
        if idx not in self.sync_data:
            self.sync_data[idx] = {}
        self.sync_data[idx][sid] = (buffer_name, frame_shape, frame_dtype, keypoints, scores, is_placeholder)
        frames_list = self.get_frames_list()
        if frames_list:
            annotated_frames = []
            for (frame, sid, orig_w, orig_h, *others) in frames_list:
                if others[0] is not None:
                    if self.save_images or self.save_video or self.display_detection:
                        frame = draw_skeleton(frame, others[0], others[1])
                    annotated_frames.append((frame, sid, orig_w, orig_h))
                    self.handle_output(sid, frame, others[0], others[1], idx)
                
            if self.display_detection:
                mosaic, _ = self.build_mosaic(annotated_frames)
                self.show_mosaic(mosaic)

    def draw_skeleton(self, frame, keypoints, scores):
        # try:
        #     # MMPose skeleton
        #     frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
        # except:
            # Sports2D skeleton
            valid_X, valid_Y, valid_scores = [], [], []
            for person_keypoints, person_scores in zip(keypoints, scores):
                person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                valid_X.append(person_X)
                valid_Y.append(person_Y)
                valid_scores.append(person_scores)
            if self.multi_person: frame = draw_bounding_box(frame, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
            frame = draw_keypts(frame, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
            frame = draw_skel(frame, valid_X, valid_Y, self.pose_model)
            return frame

    def show_mosaic(self, mosaic):          
        desired_w, desired_h = 1280, 720
        mosaic = safe_resize(mosaic, desired_w, desired_h)
        cv2.imshow("Display", mosaic)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stop()

    def handle_output(self, sid, frame, keypoints, scores, idx):
        out_dirs = self.source_outputs[sid]
        file_name = f"{out_dirs[0]}_{idx}"

        # Tracking people IDs across frames
        if self.multi_person:
            if sid not in self.prev_keypoints:
                self.prev_keypoints[sid] = keypoints
            self.prev_keypoints[sid], keypoints, scores = sort_people_sports2d(self.prev_keypoints[sid], keypoints, scores=scores)

        # Save to json
        if 'openpose' in self.output_format:
            json_path = os.path.join(out_dirs[2], f"{file_name}.json")
            save_keypoints_to_openpose(json_path, keypoints, scores)
        if self.save_images:
            cv2.imwrite(os.path.join(out_dirs[1], f"{file_name}.jpg"), frame)
        if self.save_video:
            # frame = safe_resize(frame, self.frame_size[0], self.frame_size[1])
            if sid in self.video_writers:
                self.video_writers[sid].write(frame)


class CaptureCoordinator(multiprocessing.Process):
    def __init__(self,
                 sources,
                 command_queues,
                 available_frame_buffers,
                 shared_counts,
                 source_ended,
                 webcam_ready,
                 frame_rate,
                 mode):
        super().__init__(daemon=True)
        self.sources = sources
        self.command_queues = command_queues
        self.available_frame_buffers = available_frame_buffers
        self.shared_counts = shared_counts
        self.source_ended = source_ended
        self.webcam_ready = webcam_ready
        self.mode = mode

        self.min_interval = 1.0 / frame_rate if frame_rate > 0 else 0.0
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
                if self.available_frame_buffers.qsize() >= len(ready_srcs):
                    buffs = [self.available_frame_buffers.get() for _ in ready_srcs]
                    for i, src in enumerate(ready_srcs):
                        sid = src['id']
                        self.command_queues[sid].put(("CAPTURE_FRAME", buffs[i]))
                    last_capture_time = time.time()
                else:
                    time.sleep(0.005)

            elif self.mode == 'alternating':
                chunk = self.available_frame_buffers.qsize()
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
                    buf_name = self.available_frame_buffers.get()
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
    def __init__(
        self,
        source,
        frame_queue,
        frame_buffers,
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
        self.frame_buffers = frame_buffers
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
                        buffer_name = cmd[1]
                        # Stop if we've reached total frames for video/images
                        if self.source['type'] in ('video', 'images'):
                            if self.frame_idx >= self.total_frames:
                                self.stop()
                                break

                        # Read frame and send it
                        frame, is_placeholder = self.read_frame()
                        if frame is None:
                            self.stop()
                            break

                        with self.shared_counts[self.source['id']]['processed'].get_lock():
                            self.shared_counts[self.source['id']]['processed'].value += 1

                        if self.orientation == 'portrait':
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                        frame = safe_resize(frame, self.frame_size[0], self.frame_size[1])

                        self.send_frame(frame, buffer_name, is_placeholder)

                    elif cmd[0] == "STOP_CAPTURE":
                        self.stop()
                        break
                    else:
                        pass

            logging.info(f"MediaSource {self.source['id']} ended.")

            with self.active_sources.get_lock():
                self.active_sources.value -= 1

            if self.source_ended is not None:
                self.source_ended[self.source['id']] = True

        except Exception as e:
            logging.error(f"MediaSource {self.source['id']} error: {e}")
            self.stop()
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
            self.stop()

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
                self.stop()
                return None, False

            return frame, False

        elif self.source['type'] == 'images':
            if self.frame_idx < len(self.image_files):
                frame = cv2.imread(self.image_files[self.frame_idx])
                if frame is None:
                    return None, False

                self.frame_idx += 1
                return frame, False
            else:
                self.stop()
                return None, False

        elif self.source['type'] == 'webcam':
            if self.cap is None or not self.cap.isOpened():
                if self.webcam_ready is not None:
                    self.webcam_ready[self.source['id']] = False

                self.open_webcam()

                if self.cap is None or not self.cap.isOpened():
                    placeholder_frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                    return placeholder_frame, True

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logging.warning(f"Failed to read from webcam {self.source['id']}. Reconnecting loop.")
                self.cap.release()
                self.cap = None
                placeholder_frame = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
                return placeholder_frame, True

            if self.webcam_ready is not None:
                self.webcam_ready[self.source['id']] = True

            return frame, False

        return None, False

    def send_frame(self, frame, buffer_name, is_placeholder):
        shm = self.frame_buffers[buffer_name]
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np_frame[:] = frame

        item = (
            buffer_name,
            self.frame_idx if not self.source['type'] == 'webcam' else get_formatted_timestamp(),
            frame.shape,
            frame.dtype.str,
            self.source['id'],
            is_placeholder
        )
        self.frame_queue.put(item)

        if not is_placeholder:
            with self.shared_counts[self.source['id']]['queued'].get_lock():
                self.shared_counts[self.source['id']]['queued'].value += 1
            self.frame_idx += 1

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


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
    combined_frames = config_dict['pose'].get('combined_frames', False)
    multi_workers = config_dict['pose'].get('multi_workers', False)
    multi_person = config_dict['project'].get('multi_person', False)
    output_format = config_dict['project'].get('output_format', 'openpose')

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
    num_sources = len(sources)
    available_memory = psutil.virtual_memory().available
    frame_bytes = fw * fh * 3
    n_buffers_total = int((available_memory / 2) / (frame_bytes * (num_sources / (num_sources + 1))))

    frame_buffer_count = 0

    if not combined_frames:
        frame_buffer_count = n_buffers_total

        logging.info(f"Allocating {frame_buffer_count} buffers.")
    else:
        frame_buffer_count = num_sources * 3

        logging.info(f"Allocating {frame_buffer_count} frame buffers.")

        pose_buffer_count = n_buffers_total - frame_buffer_count

        logging.info(f"Allocating {pose_buffer_count} pose buffers.")

    frame_queue = multiprocessing.Queue(maxsize=frame_buffer_count)
    pose_queue = multiprocessing.Queue()
    if combined_frames:
        pose_queue = multiprocessing.Queue(maxsize=pose_buffer_count)
    result_queue = multiprocessing.Queue()

    frame_buffers = {}
    available_frame_buffers = multiprocessing.Queue()

    for i in range(frame_buffer_count):
        buf_name = f"frame_{i}"
        shm = shared_memory.SharedMemory(name=buf_name, create=True, size=frame_bytes)
        frame_buffers[buf_name] = shm
        available_frame_buffers.put(buf_name)

    pose_buffers = {}
    available_pose_buffers = {}

    if combined_frames:
        available_pose_buffers = multiprocessing.Queue()
        for i in range(pose_buffer_count):
            buf_name = f"pose_buffer_{i}"
            shm = shared_memory.SharedMemory(name=buf_name, create=True, size=frame_bytes * num_sources)
            pose_buffers[buf_name] = shm
            available_pose_buffers.put(buf_name)

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
            source = s,
            frame_queue = frame_queue,
            frame_buffers = frame_buffers,
            shared_counts = shared_counts,
            frame_size = (fw, fh),
            active_sources = active_sources,
            command_queue = command_queues[s['id']],
            vid_img_extension = vid_img_extension,
            frame_ranges = frame_range,
            webcam_ready = webcam_ready,
            source_ended = source_ended,
            orientation = orientation
        )
        ms.start()
        media_sources.append(ms)

    result_processor = ResultQueueProcessor(
        result_queue = result_queue,
        sources = sources,
        frame_buffers = frame_buffers, 
        pose_buffers = pose_buffers,
        available_frame_buffers = available_frame_buffers,
        available_pose_buffers = available_pose_buffers,
        vid_img_extension= vid_img_extension,
        source_outputs = source_outputs,
        shared_counts = shared_counts,
        save_images = save_images,
        save_video = save_video,
        frame_size = (fw, fh),
        frame_rate = frame_rate,
        orientation = orientation,
        combined_frames = combined_frames,
        multi_person = multi_person,
        output_format = output_format,
        display_detection= display_detection,
        pose_model=pose_tracker_settings[-1]
    )
    result_processor.start()

    # Decide how many workers to start
    cpu_count = multiprocessing.cpu_count()

    if combined_frames:
        frame_processor = FrameQueueProcessor(
            frame_queue = frame_queue,
            pose_queue = pose_queue,
            sources = sources,
            frame_buffers = frame_buffers,
            pose_buffers = pose_buffers,
            available_frame_buffers = available_frame_buffers, 
            available_pose_buffers = available_pose_buffers,
            vid_img_extension = vid_img_extension,
            source_outputs = source_outputs,
            shared_counts = shared_counts,
            save_images = save_images, 
            save_video = save_video, 
            frame_size = (fw, fh), 
            frame_rate = frame_rate,
            orientation = orientation,
            combined_frames = combined_frames,
            multi_person = multi_person, 
            output_format = output_format, 
            display_detection = display_detection,
            pose_model=pose_tracker_settings[-1]
        )
        frame_processor.start()

        initial_workers = max(1, cpu_count - len(sources) - 3)
    else:
        initial_workers = max(1, cpu_count - len(sources) - 2)

    if not multi_workers:
        initial_workers = 1

    logging.info(f"Starting {initial_workers} workers.")

    def spawn_new_worker():
        worker = PoseEstimatorWorker(
            queue=pose_queue if combined_frames else frame_queue,
            result_queue=result_queue,
            shared_counts=shared_counts,
            pose_tracker_settings=pose_tracker_settings,
            buffers=pose_buffers if combined_frames else frame_buffers,
            available_buffers=available_pose_buffers if combined_frames else available_frame_buffers,
            active_sources=active_sources,
            combined_frames=combined_frames
        )
        worker.start()
        return worker


    workers = [spawn_new_worker() for _ in range(initial_workers)]

    # Start capture coordinator
    capture_coordinator = CaptureCoordinator(
        sources=sources,
        command_queues=command_queues,
        available_frame_buffers=available_frame_buffers,
        shared_counts=shared_counts,
        source_ended=source_ended,
        webcam_ready=webcam_ready,
        frame_rate=frame_rate if vid_img_extension == 'webcam' else 0,
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

    frame_buffer_bar = tqdm(
        total=frame_buffer_count,
        desc='Frame Buffers Free',
        position=bar_pos,
        leave=True,
        colour='blue'
    )
    bar_pos += 1

    if combined_frames:
        pose_buffer_bar = tqdm(
            total=pose_buffer_count,
            desc='Pose Buffers Free',
            position=bar_pos,
            leave=True,
            colour='blue'
        )
        bar_pos += 1

    if multi_workers:
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

            frame_buffer_bar.n = available_frame_buffers.qsize()
            frame_buffer_bar.refresh()

            if combined_frames:
                pose_buffer_bar.n = available_pose_buffers.qsize()
                pose_buffer_bar.refresh()

            # Check if user closed the display
            if result_processor and result_processor.stopped:
                logging.info("\nUser closed display. Stopping all streams.")
                capture_coordinator.stop()
                for ms_proc in media_sources:
                    ms_proc.stop()
                break

            alive_workers = sum(w.is_alive() for w in workers)

            if multi_workers:
                worker_bar.n = alive_workers
                worker_bar.refresh()

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
            if all_ended and frame_queue.empty() and pose_queue.empty() and alive_workers == 0:
                logging.info("All sources ended, queues empty, and worker finished. Exiting loop.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("User interrupted pose estimation.")
        capture_coordinator.stop()
        for ms_proc in media_sources:
            ms_proc.stop()

    finally:
        # Stop capture coordinator
        capture_coordinator.stop()
        capture_coordinator.join()
        if capture_coordinator.is_alive():
            capture_coordinator.terminate()

        # Stop media sources
        for s in sources:
            command_queues[s['id']].put(None)

        for ms in media_sources:
            ms.join(timeout=2)
            if ms.is_alive():
                ms.terminate()

        # Stop workers
        for w in workers:
            w.join(timeout=2)
            if w.is_alive():
                logging.warning(f"Forcibly terminating worker {w.pid}")
                w.terminate()

        # Free shared memory
        for shm in frame_buffers.values():
            shm.close()
            shm.unlink()
        for shm in pose_buffers.values():
            shm.close()
            shm.unlink()

        if combined_frames:
            frame_processor.stop()
            frame_processor.join(timeout=2)
            if frame_processor.is_alive():
                frame_processor.terminate()

        result_processor.stop()
        result_processor.join(timeout=2)
        if result_processor.is_alive():
            result_processor.terminate()

        # Final bar updates
        frame_buffer_bar.n = available_frame_buffers.qsize()
        frame_buffer_bar.refresh()
        if combined_frames:
            pose_buffer_bar.n = available_pose_buffers.qsize()
            pose_buffer_bar.refresh()

        for sid, pb in progress_bars.items():
            pb.close()
        frame_buffer_bar.close()
        if combined_frames:
            pose_buffer_bar.close()
        if multi_workers:
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
    json_output_dir = os.path.join(output_dir, f"{output_dir_name}_json")
    os.makedirs(json_output_dir, exist_ok=True)

    if save_images:
        img_output_dir = os.path.join(output_dir, f"{output_dir_name}_img")
        os.makedirs(img_output_dir, exist_ok=True)

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
        try: # from Config.toml
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if pose_model.id == 'None':
                pose_model.id = None
        except:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Select device and backend
    backend, device = init_backend_device(backend=backend, device=device)

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
            try:
                mode = ast.literal_eval(mode)
            except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
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

    return (ModelClass, det_frequency, mode, False, backend, device, pose_model)


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
                fps = round(cap.get(cv2.CAP_PROP_FPS))
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
