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


# INIT
import os
import time
import math
import glob
import json
import logging
import queue
import multiprocessing
import psutil
import cv2
import numpy as np

from datetime import datetime, timezone, timedelta
from multiprocessing import shared_memory
from tqdm import tqdm

from rtmlib import draw_skeleton

from Pose2Sim.source import WebcamSource, ImageSource, VideoSource
from deep_sort_realtime.deepsort_tracker import DeepSort
from Pose2Sim.common import (
    natural_sort_key,
    sort_people_sports2d,
    sort_people_deepsort,
    colors,
    thickness,
    draw_bounding_box,
    draw_keypts,
    draw_skel,
)


# AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# CLASSES
class PoseEstimatorWorker(multiprocessing.Process):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.stopped = False
        self.prev_keypoints = {}
        self.prev_bboxes = {}

    def run(self):
        model = self.ModelClass(mode=self.config.pose_model.mode,
            to_openpose=self.to_openpose,
            backend=self.config.pose_model.backend,
            device=self.config.pose_model.device)

        try:
            self.det_model = model.det_model
        except AttributeError:  # rtmo
            self.det_model = None
        self.pose_model = model.pose_model

        self.tracker_ready_event.set()

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
        buffer_name, timestamp, idx, frame_shape, frame_dtype, *others = item
        # Convert shared memory to numpy
        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=self.buffers[buffer_name].buf)

        if not others[1]:
            if self.det_model is not None:
                if idx % self.det_frequency == 0:
                    bboxes = self.det_model(frame)
                    if others[0] != -1:
                        self.prev_bboxes[others[0]] = bboxes
                    else:
                        self.prev_bboxes = bboxes
                else:
                    if others[0] != -1:
                        bboxes = self.prev_bboxes.get(others[0], None)
                        keypoints, scores = self.pose_model(frame, bboxes=bboxes)
                    else:
                        bboxes = self.prev_bboxes
                keypoints, scores = self.pose_model(frame, bboxes=bboxes)
            else:  # rtmo
                keypoints, scores = self.pose_model(frame)

            if self.multi_person:
                if self.tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, self.deepsort_tracker, frame, idx)
                if self.tracking_mode == 'sports2d':
                    if others[0] is not None:
                        prev_kpts = self.prev_keypoints.get(others[0], None)
                        updated_prev, keypoints, scores = sort_people_sports2d(prev_kpts, keypoints, scores=scores)
                        self.prev_keypoints[others[0]] = updated_prev
                    else:
                        self.prev_keypoints, keypoints, scores = sort_people_sports2d(
                            self.prev_keypoints,
                            keypoints,
                            scores=scores
                            )
        else:
            keypoints, scores = None, None

        result = (buffer_name, timestamp, idx, frame_shape, frame_dtype, others[0], others[1], others[2], keypoints, scores)
        self.result_queue.put(result)

    def stop(self):
        self.stopped = True


class BaseSynchronizer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

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
                    buffer_name, frame_shape, frame_dtype, keypoints, scores, is_placeholder, transform_info = (
                        complete_group[sid]
                    )
                    frame = np.ndarray(
                        frame_shape,
                        dtype=np.dtype(frame_dtype),
                        buffer=self.frame_buffers[buffer_name].buf,
                    ).copy()
                    self.available_frame_buffers.put(buffer_name)
                    orig_w, orig_h = frame_shape[1], frame_shape[0]
                    if is_placeholder:
                        cv2.putText(frame, "Not connected", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    frames_list.append((frame, sid, orig_w, orig_h, transform_info, keypoints, scores))
                return frames_list
        return None

    def build_mosaic(self, frames_list):
        target_w, target_h = self.frame_size
        mosaic = np.zeros((self.mosaic_rows * target_h, self.mosaic_cols * target_w, 3), dtype=np.uint8)
        subinfo = {}
        for frame_tuple in frames_list:
            frame, *others = frame_tuple
            info = self.mosaic_subinfo[others[0]]
            x_off = info["x_offset"]
            y_off = info["y_offset"]
            mosaic[y_off:y_off+target_h, x_off:x_off+target_w] = frame
            subinfo[others[0]] = {
                "x_offset": x_off,
                "y_offset": y_off,
                "scaled_w": target_w,
                "scaled_h": target_h,
                "orig_w": others[1],
                "orig_h": others[2],
                "transform_info": others[3]
            }
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
            sc_w = info["scaled_w"]
            sc_h = info["scaled_h"]
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
                buffer_name, timestamp, idx, frame_shape, frame_dtype, sid, is_placeholder, _ = self.frame_queue.get_nowait()
                if idx not in self.sync_data:
                    self.sync_data[idx] = {}
                self.sync_data[idx][sid] = (buffer_name, frame_shape, frame_dtype, None, None, is_placeholder, None)

                if frames_list := self.get_frames_list():
                    mosaic, subinfo = self.build_mosaic(frames_list)
                    pose_buf_name = self.available_pose_buffers.get_nowait()
                    np.ndarray(mosaic.shape, dtype=mosaic.dtype, buffer=self.pose_buffers[pose_buf_name].buf)[:] = mosaic
                    self.pose_queue.put((pose_buf_name, timestamp, idx, mosaic.shape, mosaic.dtype.str, -1, False, subinfo))
            except queue.Empty:
                time.sleep(0.005)


class ResultQueueProcessor(multiprocessing.Process, BaseSynchronizer):
    def __init__(self, result_queue, pose_model, **kwargs):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, **kwargs)
        self.result_queue = result_queue
        self.pose_model = pose_model

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
                output_video_path = out_dirs[-2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writers[sid] = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    self.frame_rate,
                    (self.frame_size[0], self.frame_size[1])
                )

    def process_result_item(self, result_item):
        buffer_name, timestamp, idx, frame_shape, frame_dtype, sid, is_placeholder, info, keypoints, scores = result_item
        if self.combined_frames:
            self.handle_combined_frames(buffer_name, frame_shape, frame_dtype, keypoints, scores, info, timestamp, idx)
        else:
            self.handle_individual_frames(buffer_name, frame_shape, frame_dtype, sid, is_placeholder, keypoints, scores, timestamp, idx)

    def handle_combined_frames(self, buffer_name, frame_shape, frame_dtype, keypoints, scores, subinfo, timestamp, idx):
        shm = self.pose_buffers[buffer_name]
        mosaic = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf)

        if self.save_images or self.save_video or self.display_detection:
            mosaic = draw_skeleton(mosaic, keypoints, scores)
            if self.display_detection:
                self.show_mosaic(mosaic)

        frames, recovered_keypoints, recovered_scores = self.read_mosaic(mosaic, subinfo, keypoints, scores)
        self.available_pose_buffers.put(buffer_name)

        for sid in frames:
            trans_info = subinfo[sid].get("transform_info", None)
            self.handle_output(sid, frames[sid], recovered_keypoints[sid], recovered_scores[sid], timestamp, idx, trans_info)

    def handle_individual_frames(self, buffer_name, frame_shape, frame_dtype, sid, is_placeholder, keypoints, scores, timestamp, idx):
        if idx not in self.sync_data:
            self.sync_data[idx] = {}
        self.sync_data[idx][sid] = (buffer_name, frame_shape, frame_dtype, keypoints, scores, is_placeholder, None)
        frames_list = self.get_frames_list()
        if frames_list:
            annotated_frames = []
            for (frame, sid, orig_w, orig_h, transform_info, *others) in frames_list:
                if others[0] is not None:
                    if self.save_images or self.save_video or self.display_detection:
                        frame = draw_skeleton(frame, others[0], others[1])
                    annotated_frames.append((frame, sid, orig_w, orig_h, transform_info))
                    self.handle_output(sid, frame, others[0], others[1], timestamp, idx, transform_info)

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
            if self.multi_person: 
                frame = draw_bounding_box(frame, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
            frame = draw_keypts(frame, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
            frame = draw_skel(frame, valid_X, valid_Y, self.pose_model)
            return frame

    def show_mosaic(self, mosaic):   
        desired_w, desired_h = 1280, 720
        mosaic = transform(mosaic, desired_w, desired_h, False)[0]
        cv2.imshow("Display", mosaic)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stop()

    def handle_output(self, sid, frame, keypoints, scores, timestamp, idx, transform_info):
        out_dirs = self.source_outputs[sid]
        file_name = f"{out_dirs[0]}_{timestamp}_{idx:06d}"

        # Save to json
        if transform_info is not None:
            keypoints = inverse_transform_keypoints(keypoints, transform_info)
        if 'openpose' in self.output_format:
            json_path = os.path.join(out_dirs[2], f"{file_name}.json")
            save_keypoints_to_openpose(json_path, keypoints, scores)
        if self.save_images:
            cv2.imwrite(os.path.join(out_dirs[1], f"{file_name}.jpg"), frame)
        if self.save_video:
            if sid in self.video_writers:
                self.video_writers[sid].write(frame)


class CaptureCoordinator(multiprocessing.Process):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.min_interval = 1.0 / self.frame_rate if self.frame_rate > 0 else 0.0
        self.stopped = multiprocessing.Value('b', False)

    def run(self):
        self.tracker_ready_event.wait()
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
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

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

                if self.webcam_recording and self.cap is not None and self.cap.isOpened() and self.output_raw_video_path:
                    raw_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    raw_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.raw_writer = cv2.VideoWriter(self.source_outputs[-1], fourcc, 30, (raw_width, raw_height))

            elif self.source['type'] == 'video':
                self.open_video()
                self.start_timestamp = get_file_utc_timestamp(self.source['path'])
                if self.frame_ranges:
                    self.total_frames = len(self.frame_ranges)
                else:
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.shared_counts[self.source['id']]['total'].value = self.total_frames

            elif self.source['type'] == 'images':
                self.load_images()
                self.start_timestamp = get_file_utc_timestamp(self.image_files[0])
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

                        frame, transform_info = transform(frame, self.frame_size[0], self.frame_size[1], True, self.rotation)

                        self.send_frame(frame, buffer_name, is_placeholder, transform_info)

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

    def send_frame(self, frame, buffer_name, is_placeholder, transform_info):
        shm = self.frame_buffers[buffer_name]
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np_frame[:] = frame

        if self.source['type'] in ('video', 'images'):
            timestamp_str = get_frame_utc_timestamp(self.start_timestamp, self.frame_idx, self.frame_rate)
        else:
            timestamp_str = get_formatted_utc_timestamp()

        item = (
            buffer_name,
            timestamp_str,
            self.frame_idx,
            frame.shape,
            frame.dtype.str,
            self.source['id'],
            is_placeholder,
            transform_info
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


## FUNCTIONS
def estimate_pose_all(config):
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

    deepsort_tracker = None
    if tracking_mode == 'deepsort' and multi_person:
        deepsort_tracker = DeepSort(**config.get_deepsort_params())

    logging.info(f"Model input size: {config.pose_model.det_input_size[0]}x{config.pose_model.det_input_size[1]}")

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

    fw, fh = find_largest_frame_size(sources, vid_img_extension)

    num_sources = len(config.sources)

    if fw >= fh:
        mosaic_cols = math.ceil(math.sqrt(num_sources))
        mosaic_rows = math.ceil(num_sources / mosaic_cols)
    else:
        mosaic_rows = math.ceil(math.sqrt(num_sources))
        mosaic_cols = math.ceil(num_sources / mosaic_rows)

    if combined_frames:
        cell_w = frame_size[0] // mosaic_cols
        cell_h = frame_size[1] // mosaic_rows

        logging.info(f"Combined frames: {mosaic_rows} rows & {mosaic_cols} cols")
        logging.info(f"Frame input size: {cell_w}x{cell_h}")

        frame_size = (cell_w, cell_h)
    else:
        frame_size = frame_size

    mosaic_subinfo = {}
    for i, source in enumerate(config.sources):
        r = i // mosaic_cols
        c = i % mosaic_cols
        if combined_frames:
            x_off = c * cell_w
            y_off = r * cell_h
            source.mosaic_subinfo = {"x_offset": x_off, "y_offset": y_off, "scaled_w": cell_w, "scaled_h": cell_h}
        else:
            x_off = c * frame_size[0]
            y_off = r * frame_size[1]
            source.mosaic_subinfo = {"x_offset": x_off, "y_offset": y_off, "scaled_w": frame_size[0], "scaled_h": frame_size[1]}

    available_memory = psutil.virtual_memory().available
    frame_bytes = frame_size[0] * frame_size[1] * 3
    n_buffers_total = int((available_memory / 2) / (frame_bytes * (num_sources / (num_sources + 1))))

    frame_buffer_count = 0

    if not combined_frames:
        frame_buffer_count = n_buffers_total

        logging.info(f"Allocating {frame_buffer_count} buffers.")
    else:
        frame_buffer_count = num_sources * 3

        logging.info(f"Allocating {frame_buffer_count} frame buffers.")

        pose_buffer_count = min(n_buffers_total - frame_buffer_count, 10000)

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
    for source in config.sources:
        shared_counts[source.name] = {
            'queued': multiprocessing.Value('i', 0),
            'processed': multiprocessing.Value('i', 0),
            'total': multiprocessing.Value('i', 0)
        }

    # Decide how many workers to start
    cpu_count = multiprocessing.cpu_count()

    active_sources = multiprocessing.Value('i', num_sources)

    if combined_frames:
        initial_workers = max(1, cpu_count - num_sources - 3)
    else:
        initial_workers = max(1, cpu_count - num_sources - 2)

    if not multi_workers:
        initial_workers = 1

    logging.info(f"Starting {initial_workers} workers.")

    tracker_ready_event = multiprocessing.Event()

    def spawn_new_worker():
        worker = PoseEstimatorWorker(
            queue=pose_queue if combined_frames else frame_queue,
            result_queue=result_queue,
            shared_counts=shared_counts,
            ModelClass=ModelClass,
            det_frequency=det_frequency,
            to_openpose=to_openpose,
            backend=backend,
            device=device,
            buffers=pose_buffers if combined_frames else frame_buffers,
            available_buffers=available_pose_buffers if combined_frames else available_frame_buffers,
            active_sources=active_sources,
            combined_frames=combined_frames,
            tracker_ready_event=tracker_ready_event,
            multi_person=multi_person,
            tracking_mode=tracking_mode,
            deepsort_tracker=deepsort_tracker
        )
        worker.start()
        return worker


    workers = [spawn_new_worker() for _ in range(initial_workers)]

    manager = multiprocessing.Manager()
    webcam_ready = manager.dict()
    source_ended = manager.dict()
    command_queues = {}

    media_sources = []
    for source in counfig.sources:
        source.ended = False
        source.command_queues = manager.Queue()
        source.output = create_output_folders(s['path'], pose_dir, save_images, webcam_recording)

        if isinstance(source, WebcamSource):
            source.ready = False

        ms = MediaSource(
            source=s,
            frame_queue=frame_queue,
            frame_buffers=frame_buffers,
            shared_counts=shared_counts,
            frame_size=frame_size,
            frame_rate=frame_rate,
            active_sources=active_sources,
            command_queue=command_queues[s['id']],
            vid_img_extension=vid_img_extension,
            frame_ranges=frame_range,
            webcam_ready=webcam_ready,
            source_ended=source_ended,
            rotation=rotation,
            webcam_recording=webcam_recording,
        )
        ms.start()
        media_sources.append(ms)

    result_processor = ResultQueueProcessor(
        result_queue=result_queue,
        pose_model=pose_model,
        sources=sources,
        frame_buffers=frame_buffers,
        pose_buffers=pose_buffers,
        available_frame_buffers=available_frame_buffers,
        available_pose_buffers=available_pose_buffers,
        vid_img_extension=vid_img_extension,
        source_outputs=source_outputs,
        shared_counts=shared_counts,
        save_images=save_images,
        save_video=save_video,
        frame_size=frame_size,
        frame_rate=frame_rate,
        combined_frames=combined_frames,
        multi_person=multi_person,
        output_format=output_format,
        display_detection=display_detection,
        mosaic_cols=mosaic_cols,
        mosaic_rows=mosaic_rows,
        mosaic_subinfo=mosaic_subinfo
    )
    result_processor.start()

    if combined_frames:
        frame_processor = FrameQueueProcessor(
            frame_queue=frame_queue,
            pose_queue=pose_queue,
            sources=sources,
            frame_buffers=frame_buffers,
            pose_buffers=pose_buffers,
            available_frame_buffers=available_frame_buffers,
            available_pose_buffers=available_pose_buffers,
            vid_img_extension=vid_img_extension,
            source_outputs=source_outputs,
            shared_counts=shared_counts,
            save_images=save_images,
            save_video=save_video,
            frame_size=frame_size,
            frame_rate=frame_rate,
            combined_frames=combined_frames,
            multi_person=multi_person,
            output_format=output_format,
            display_detection=display_detection,
            mosaic_cols=mosaic_cols,
            mosaic_rows=mosaic_rows,
            mosaic_subinfo=mosaic_subinfo
        )
        frame_processor.start()

    # Start capture coordinator
    capture_coordinator = CaptureCoordinator(
        sources=sources,
        command_queues=command_queues,
        available_frame_buffers=available_frame_buffers,
        shared_counts=shared_counts,
        source_ended=source_ended,
        webcam_ready=webcam_ready,
        frame_rate=frame_rate if vid_img_extension == 'webcam' else 0,
        mode=capture_mode,
        tracker_ready_event=tracker_ready_event
    )
    capture_coordinator.start()

    # Setup progress bars
    progress_bars = {}
    bar_ended_state = {}
    bar_pos = 0

    for source in config.sources:
        if isinstance(source, WebcamSource):
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
            for source, pb in progress_bars.items():
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
        for source in config.sources:
            source.command_queue.put(None)

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


def transform(frame, desired_w, desired_h, full, rotation=0):
    rotation = rotation % 360
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    rotated_h, rotated_w = frame.shape[:2]

    scale = min(desired_w / rotated_w, desired_h / rotated_h)
    new_w = int(rotated_w * scale)
    new_h = int(rotated_h * scale)
    if scale != 0:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if full:
        canvas = np.zeros((desired_h, desired_w, 3), dtype=np.uint8)
        x_offset = (desired_w - new_w) // 2
        y_offset = (desired_h - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
        bottom_left_offset_y = desired_h - (y_offset + new_h)
        transform_info = {
            'rotation': rotation,
            'scale': scale,
            'x_offset': x_offset,
            'y_offset': bottom_left_offset_y,
            'rotated_size': (rotated_w, rotated_h),
            'canvas_size': (desired_w, desired_h)
        }
        return canvas, transform_info
    else:
        return frame, None


def inverse_transform_keypoints(keypoints, transform_info):
    desired_w, desired_h = transform_info['canvas_size']
    scale = transform_info['scale']
    x_offset = transform_info['x_offset']
    y_offset = transform_info['y_offset']
    rotation = transform_info['rotation']
    rotated_size = transform_info['rotated_size']
    new_keypoints = []
    for person in keypoints:
        new_person = []
        for (x, y) in person:

            y_bl = desired_h - y
            x_bl = desired_w - x

            X = (x_bl - x_offset) / scale
            Y = (y_bl - y_offset) / scale

            if rotation % 360 == 0:
                orig_x, orig_y = X, Y
            elif rotation % 360 == 90:
                orig_x = rotated_size[0] - Y
                orig_y = X
            elif rotation % 360 == 180:
                orig_x = rotated_size[0] - X
                orig_y = rotated_size[1] - Y
            elif rotation % 360 == 270:
                orig_x = Y
                orig_y = rotated_size[1] - X
            else:
                orig_x, orig_y = X, Y
            new_person.append([orig_x, orig_y])
        new_keypoints.append(np.array(new_person))
    return new_keypoints


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

def get_formatted_utc_timestamp():
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%S") + f"{dt.microsecond:06d}"


def get_file_utc_timestamp(file_path):
    ts = os.path.getmtime(file_path)
    return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=timezone.utc)


def get_frame_utc_timestamp(start_timestamp, frame_idx, fps):
    frame_time = start_timestamp + timedelta(seconds=(frame_idx / fps))
    return frame_time.strftime("%Y%m%dT%H%M%S") + f"{frame_time.microsecond:06d}"