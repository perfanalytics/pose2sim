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
import json
import logging
import queue
import math
import multiprocessing
import psutil
import cv2
import uuid
import numpy as np
import pandas as pd
import re
import ast
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone

from multiprocessing import shared_memory
from tqdm import tqdm
from typing import Any

from rtmlib import draw_skeleton

from Pose2Sim.source import FrameData
from deep_sort_realtime.deepsort_tracker import DeepSort
from Pose2Sim.stages.base import BaseStage
from Pose2Sim.source import BaseSource, WebcamSource
from Pose2Sim.model import PoseModel
from Pose2Sim.common import (
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
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


@dataclass
class PoseEstimationSettings:
    output_format: str
    webcam_recording: bool
    save_video: bool
    save_images: bool
    tracking_mode: str
    display_detection: bool
    multi_person: bool
    combined_frames: bool
    multi_workers: int
    overwrite_pose: bool
    deepsort_params: dict[str, Any]

    # Fabrication à partir du dictionnaire complet de config ---------------- #
    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "PoseEstimationSettings":
        pe = cfg.get("poseEstimation")
        save_flags = pe.get("save_video")
        return cls(
            output_format           = pe.get("output_format"),
            webcam_recording        = pe.get("webcam_recording"),
            save_video              = "to_video"  in save_flags,
            save_images             = "to_images" in save_flags,
            tracking_mode           = pe.get("tracking_mode"),
            display_detection       = pe.get("display_detection"),
            multi_person            = pe.get("multi_person"),
            combined_frames         = pe.get("combined_frames"),
            multi_workers           = pe.get("multi_workers"),
            overwrite_pose          = pe.get("overwrite_pose"),
            deepsort_params         = pe.get("get_deepsort_params"),
        )


# CLASSES
class PoseEstimatorWorker(multiprocessing.Process):
    def __init__(self, sources, pose_model, input_queue, output_queue, tracker_ready_event, deepsort_params=None):
        super().__init__(daemon=True)
        self.sources = sources
        self.pose_model = pose_model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tracker_ready_event = tracker_ready_event
        self.deepsort_params = deepsort_params
        self.deepsort_tracker = None

        self.ended = True
        self.prev_keypoints = {}
        self.prev_bboxes = {}

    def run(self):
        model = self.pose_model.model_class(mode=self.pose_model.mode,
            to_openpose=self.pose_model.to_openpose,
            backend=self.pose_model.backend,
            device=self.pose_model.device)

        try:
            self.det_model = model.det_model
        except AttributeError:  # rtmo
            self.det_model = None
        self.model = model.pose_model

        self.tracker_ready_event.set()

        while not self.ended:
            if all(source.ended for source in self.sources) and self.input_queue.isEmpty():
                self.ended = True
                break

            frame_data = self.input_queue.get()
            frame = np.ndarray(frame_data.shape, dtype=frame_data.dtype, buffer=frame_data.buffer.buf)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if not frame_data.placeholder:
                if self.det_model is not None:
                    if frame_data.idx % self.pose_model.det_frequency == 0:
                        bboxes = self.det_model(frame)
                        if frame_data.source is not None:
                            self.prev_bboxes[frame_data.source.name] = bboxes
                        else:
                            self.prev_bboxes = bboxes
                    else:
                        if frame_data.source is not None:
                            bboxes = self.prev_bboxes.get(frame_data.source.name, None)
                        else:
                            bboxes = self.prev_bboxes
                    frame_data.keypoints, frame_data.scores = self.model(frame, bboxes=bboxes)
                else:  # rtmo
                    frame_data.keypoints, frame_data.scores = self.model(frame)

                if self.multi_person:
                    if self.tracking_mode == 'deepsort':
                        if self.deepsort_tracker == None:
                            self.deepsort_tracker = DeepSort(**self.deepsort_params)
                        frame_data.keypoints, frame_data.scores = sort_people_deepsort(frame_data.keypoints, frame_data.scores, self.deepsort_tracker, frame, frame_data.idx)
                    if self.tracking_mode == 'sports2d':
                        if frame_data.source is not None:
                            prev_kpts = self.prev_keypoints.get(frame_data.source.name, None)
                            updated_prev, frame_data.keypoints, frame_data.scores = sort_people_sports2d(prev_kpts, frame_data.keypoints, frame_data.scores)
                            self.prev_keypoints[frame_data.source.name] = updated_prev
                        else:
                            self.prev_keypoints, frame_data.keypoints, frame_data.scores = sort_people_sports2d(
                                self.prev_keypoints,
                                frame_data.keypoints,
                                frame_data.scores
                                )
            else:
                frame_data.keypoints, frame_data.scores = None, None

            self.output_queue.put(frame_data)

    def stop(self):
        self.stopped = True


class BaseProcessor:
    def __init__(self, sources, queue, combined_frames, available_buffers, buffers, frame_size):
        self.sources = sources
        self.sync_data = {}
        self.combined_frames = combined_frames
        self.queue = queue
        self.available_buffers = available_buffers
        self.buffers = buffers
        self.frame_size = frame_size

    def get_frames_list(self):
        frames_list = []
        for idx in sorted(self.sync_data.keys()):
            group = self.sync_data[idx]
            if len(group) == len(self.sources):
                complete_group = self.sync_data.pop(idx)
                for source in self.sources:
                    frame_data = (complete_group[source.name])
                    frames_list.append(frame_data)
                return frames_list


class OutputQueueProcessor(BaseProcessor):
    def __init__(self, sources, combined_frames, queue, available_buffers, buffers, frame_size, display_detection, save_video, save_images, skeleton, downstream_queue):
        BaseProcessor.__init__(self, sources, queue, combined_frames, available_buffers, buffers, frame_size)
        self.display_detection = display_detection
        self.save_video = save_video
        self.save_images = save_images
        self.skeleton = skeleton
        self.downstream_queue = downstream_queue

    def run(self):
        try:
            frame_data = self.queue.get()
            self.process_result_item(frame_data)
        except queue.Empty:
            time.sleep(0.005)

    def process_result_item(self, frame_data):
        if self.combined_frames:
            self.handle_combined_frames(frame_data)
        else:
            self.handle_individual_frames(frame_data)

    def handle_combined_frames(self, mosaic_data):
        mosaic = np.ndarray(
            mosaic_data.shape,
            dtype=np.dtype(mosaic_data.dtype),
            buffer=mosaic_data.buffer.buf,
        )
        self.available_buffers.put(mosaic_data.buffer.name)

        if self.save_images or self.save_video or self.display_detection:
            mosaic = draw_skeleton(mosaic, mosaic_data.keypoints, mosaic_data.scores)
            if self.display_detection:
                self.show_mosaic(mosaic)

        for source in self.sources:
            rec_kpts = []
            rec_scores = []

            x0, y0 = source.x_offset, source.y_offset
            x1, y1 = x0 + source.width, y0 + source.height

            for kp_person, sc_person in zip(mosaic_data.keypoints, mosaic_data.scores):
                local_kpts = []
                local_scores = []

                for (xk, yk), scv in zip(kp_person, sc_person):
                    if x0 <= xk <= x1 and y0 <= yk <= y1:
                        local_kpts.append([xk - x0, yk - y0])
                        local_scores.append(scv)

                if local_kpts:
                    rec_kpts.append(np.asarray(local_kpts, dtype=np.float32))
                    rec_scores.append(np.asarray(local_scores, dtype=np.float32))

            frame_data = FrameData(
                source     = source,
                timestamp  = mosaic_data.timestamp,
                idx        = mosaic_data.idx,
                keypoints  = rec_kpts,
                scores     = rec_scores,
                file_name = f"{source.name}_{mosaic_data.timestamp}_{mosaic_data.idx:06d}"
            )

            frame = mosaic[y0:y1, x0:x1],

            if self.settings.save_files[1]:
                cv2.imwrite(os.path.join(frame_data.source.img_output_dir, f"{frame_data.file_name}.jpg"), frame)
            if self.settings.save_files[0]:
                frame_data.source.video_writer.write(frame)

            self.downstream_queue.put(frame_data)

    def handle_individual_frames(self, frame_data):
        if frame_data.idx not in self.sync_data:
            self.sync_data[frame_data.idx] = {}
        self.sync_data[frame_data.idx][frame_data.source.name] = frame_data
        if frames_list := self.get_frames_list():
            if self.display_detection:
                mosaic = np.zeros(
                    self.frame_size,
                    dtype=np.uint8
                )
            for frame_data in frames_list:
                if not frame_data.placeholder:
                    frame = np.ndarray(
                        frame_data.shape,
                        dtype=np.dtype(frame_data.dtype),
                        buffer=frame_data.buffer.buf,
                    )
                    self.available_buffers.put(frame_data.buffer.name)
                    if self.save_images or self.save_video or self.display_detection:
                        frame = draw_skeleton(frame, frame_data.keypoints, frame_data.scores)

                    frame_data.file_name = f"{frame_data.source.name}_{frame_data.timestamp}_{frame_data.idx:06d}"

                    if self.settings.save_files[1]:
                        cv2.imwrite(os.path.join(frame_data.source.img_output_dir, f"{frame_data.file_name}.jpg"), frame)
                    if self.settings.save_files[0]:
                        frame_data.source.video_writer.write(frame)

                    self.downstream_queue.put(frame_data)

                    if mosaic:
                        source = frame_data.source
                        mosaic[source.y_offset:source.y_offset + source.height,
                            source.x_offset:source.x_offset + source.width] = frame

            if mosaic:
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
            frame = draw_skel(frame, valid_X, valid_Y, self.skeleton)
            return frame

    def show_mosaic(self, mosaic):
        cv2.imshow("Display", mosaic)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stop()


class InputQueueProcessor(BaseProcessor):
    def __init__(self, sources, queue, interval, combined_frames, available_buffers, buffers, buffers_count, frame_size):
        BaseProcessor.__init__(self, sources, queue, combined_frames, available_buffers, buffers, frame_size)
        self.last_capture_time = time.time()
        self.interval = interval
        self.buffers_count = buffers_count

    def run(self):
        if self.combined_frames:
            for source in self.sources:
                frame_data = source.frame_queue.get()
                if frame_data.idx not in self.sync_data:
                    self.sync_data[frame_data.idx] = {}
                self.sync_data[frame_data.idx][frame_data.source.name] = frame_data

            if frames_list := self.get_frames_list():
                mosaic = np.zeros(
                    self.frame_size,
                    dtype=np.uint8
                )
                timestamps = []
                idx = None

                for frame_data in frames_list:
                    if not frame_data.placeholder:
                        timestamps.append(frame_data.timestamp)
                        idx = frame_data.idx
                        frame = np.ndarray(
                            frame_data.shape,
                            dtype=np.dtype(frame_data.dtype),
                            buffer=frame_data.buffer.buf,
                        )
                        self.available_buffers.put(frame_data.buffer.name)

                        source = frame_data.source
                        mosaic[source.y_offset:source.y_offset + source.height,
                            source.x_offset:source.x_offset + source.width] = frame

                avg_timestamp = np.mean(timestamps)
                mosaic_data = FrameData(None, avg_timestamp, idx)
                buffer = self.available_buffers.get()
                shm = self.buffers[buffer]
                mosaic_data.shape = mosaic.shape
                mosaic_data.dtype = mosaic.dtype.str
                mosaic_data.buffer = shm
                np_mosaic = np.ndarray(mosaic_data.shape, dtype=mosaic_data.dtype, buffer=shm.buf)
                np_mosaic[:] = mosaic

                self.queue.put(mosaic_data)
        else:
            for source in self.sources:
                self.queue.put(source.frame_queue.get())

        elapsed = time.time() - self.last_capture_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)

        if self.available_buffers.qsize() == self.buffers_count:
            empty = True

        if self.mode == 'continuous' or (self.mode == 'alternating' and empty):
            if self.available_frame_buffers.qsize() >= len(self.sources):
                for source in self.sources:
                    shared_memory = self.buffers[self.available_buffers.get()]
                    self.source.command_queues.put(("CAPTURE_FRAME", shared_memory))
                self.last_capture_time = time.time()
            else:
                empty = False

    def stop(self):
        self.ended = True
        for source in self.sources:
            self.source.command_queues.put(None)


class PoseEstimationSession:
    def __init__(self, settings, sources, pose_model, downstream_queue):
        self.settings = settings
        self.sources = sources
        self.pose_model = pose_model
        self.capture_coord = None
        self.frame_proc = None
        self.result_proc = None
        self.workers = []
        self.downstream_queue = downstream_queue

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        overwrite_pose = self.pose_estimation.get('overwrite_pose')

        for source in self.sources:
            if not isinstance(source, WebcamSource):
                if os.path.exists(os.path.join(self.pose_dir, source.name)) and not overwrite_pose:
                    logging.info(f'[{source.name} - pose estimation] Skipping as it has already been done.'
                                'To recalculate, set overwrite_pose to true in Config.toml.')
                    return
                elif os.path.exists(os.path.join(self.pose_dir, source.name)) and overwrite_pose:
                    logging.info(f'[{source.name} - pose estimation] Overwriting estimation results.')

        mosaic_rows = 0
        mosaic_cols = 0

        if self.settings.combined_frames:
            mosaic_cols = math.ceil(math.sqrt(len(self.sources)))
            mosaic_rows = mosaic_cols
            cell_w = self.pose_model.det_input_size[0] // mosaic_cols
            cell_h = self.pose_model.det_input_size[1] // mosaic_rows

            logging.info(f"Combined frames: {mosaic_rows} rows & {mosaic_cols} cols")

            for i, source in enumerate(self.sources):
                r = i // mosaic_cols
                c = i % mosaic_cols
                x_off = c * cell_w
                y_off = r * cell_h
                source.x_offset = x_off
                source.y_offset = y_off
                source.desired_width = cell_w
                source.desired_height = cell_h

            frame_size = (cell_w, cell_h)
        else:
            frame_size = self.pose_model.det_input_size
            logging.info(f"Frame input size: {frame_size[0]}x{frame_size[1]}")
            for source in self.sources:
                source.desired_width = frame_size[0]
                source.desired_height = frame_size[1]

        available_memory = psutil.virtual_memory().available
        self.frame_size = self.pose_model.det_input_size[0] * self.pose_model.det_input_size[1] * 3
        self.buffers_count = min(
            int((available_memory / 2) / (self.frame_size * (len(self.sources) / (len(self.sources) + 1)))),
            10000
        )

        self.buffers = {}
        self.available_buffers = multiprocessing.Queue(maxsize=self.buffers_count)

        for _ in range(self.buffers_count):
            buf_uuid = str(uuid.uuid4())
            shm = shared_memory.SharedMemory(name=buf_uuid, create=True, size=self.frame_size)
            self.buffers[buf_uuid] = shm
            self.available_buffers.put(buf_uuid)

        for source in self.sources:
            source.start_in_process()
            source.capture_ready_event.wait()
            if self.settings.save_files[0]:
                source.intit_video_writer()

        self.fps = min(source.fps for source in self.sources)
        logging.info(f'[Pose estimation] capture frame rate set to: {self.fps}.')

        tracker_ready = multiprocessing.Event()

        self.input_queue  = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        self.input_proc = InputQueueProcessor(
            sources=self.sources,
            queue=self.input_queue,
            interval=1.0/self.fps,
            combined_frames=self.settings.combined_frames,
            available_buffers = self.available_buffers,
            buffers = self.buffers,
            buffers_count = self.buffers_count,
            frame_size=self.frame_size,
        )
        self.output_proc = OutputQueueProcessor(
            sources=self.sources,
            combined_frames=self.settings.combined_frames,
            queue=self.output_queue,
            available_buffers = self.available_buffers,
            buffers = self.buffers,
            frame_size=self.frame_size,
            downstream_queue=self.downstream_queue,
        )
        
        tracker_ready = multiprocessing.Event()

        cpu = multiprocessing.cpu_count()

        n_workers = 1

        deepsort_params = None
        if self.settings.tracking_mode == 'deepsort':
            try:
                deepsort_params = ast.literal_eval(self.setting.deepsort_params)
            except:  # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/', ':/').replace('":"\\', ':\\')
                deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params)  # changes "[640", "640]" to [640,640]
                deepsort_params = json.loads(deepsort_params)

        if self.settings.multi_workers:
            n_workers = max(1, cpu - len(self.sources) - 4)
            logging.info(f"Starting {n_workers} workers.")

        def new_worker():
            w = PoseEstimatorWorker(
                sources=self.sources,
                pose_model=self.pose_model,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                tracker_ready_event=tracker_ready,
                deepsort_params=deepsort_params,
            )
            w.start()
            return w

        self.workers = [new_worker() for _ in range(n_workers)]

        self.tracker_ready_event.wait()

        self.bars = {}
        bar_pos = 0
        for src in self.sources:
            src.progress_bar.position = bar_pos
            bar_pos += 1

        self.bars["buffers"] = tqdm(
            total=self.buffer_manager.buffers_count,
            desc="Buffers Free", position=bar_pos, leave=True
        )
        bar_pos += 1


        if self.settings.multi_workers:
            self.bars["workers"] = tqdm(
                total=len(self.workers),
                desc="Active Workers", position=bar_pos, leave=True
            )

        logging.info("PoseEstimationSession started.")

    def run(self):
        try:
            prev_ended = 0
            
            logging.info(f"Allocating {self.buffers_count} buffers.")

            while not self.ended:
                if not all(not sources.ended for sources in self.sources):
                    self.capture_coord.run()
                    self.input_proc.run()
                
                self.output_proc.run()

                self.bars["buffers"].n = self.available_buffers.qsize()
                self.bars["buffers"].refresh()

                alive = sum(not w.ended for w in self.workers)

                if self.settings.multi_workers:
                    self.bars["workers"].n = alive
                    self.bars["workers"].refresh()

                if alive == 0:
                    break

                if cv2.waitKey(1) & 0xFF == ord(" "):
                    logging.info("Space bar pressed: stopping sources.")
                    for s in self.sources:
                        s.stop()

                time.sleep(0.05)

        except (KeyboardInterrupt, SystemExit):
            logging.info("User interrupted pose estimation.")
        finally:
            self.stop()

    def stop(self):
        self.ended = True

        if self.capture_coord:
            self.capture_coord.stop()
            self.capture_coord.join(timeout=2)

        for s in self.sources:
            s.stop()
            s.progress_bar.close()

        for w in self.workers:
            w.join(timeout=2)
            if w.is_alive():
                w.terminate()

        for p in (self.frame_proc, self.result_proc):
            if p:
                p.stop()
                p.join(timeout=2)
                if p.is_alive():
                    p.terminate()

        for shm in self.buffers:
            shm.close()
            shm.unlink()

        for b in self.bars.values():
            b.close()
        if self.settings.display_detection:
            cv2.destroyAllWindows()

        logging.info("PoseEstimationSession stopped.")


class PoseEstimationStage(BaseStage):
    name = "pose_estimation"
    stream = True

    def __init__(self, settings: PoseEstimationSettings, sources: list[BaseSource], session_dir: Path, pose_model: PoseModel):
        self.settings    = settings
        self.sources     = sources
        self.session_dir = Path(session_dir)
        self.pose_model  = pose_model
        self.pose_dir = os.path.join(self.session_dir, 'pose')

    def run(self, in_q, out_q, stop_evt):
        os.makedirs(self.pose_dir, exist_ok=True)

        for source in self.sources:
            if isinstance(self, WebcamSource):
                now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                output_dir_name = f"{source.name}_{now}"

                if self.settings.webcam_recording:
                    source.output_video_path = os.path.join(self.pose_dir, f"{output_dir_name}_record.avi")
                    source.csv_file = os.path.join(self.pose_dir, f"{output_dir_name}_timestamps.csv")

            else:
                output_dir_name = self.name
                if 'openpose' in self.settings.output_format:
                    source.output_dir = os.path.join(self.pose_dir, f"{output_dir_name}_output")
                    os.makedirs(source.output_dir, exist_ok=True)

            if self.settings.save_images:
                source.img_output_dir = os.path.join(self.pose_dir, f"{output_dir_name}_img")
                os.makedirs(source.img_output_dir, exist_ok=True)

            if self.settings.save_video:
                source.output_video_path = os.path.join(self.pose_dir, f"{output_dir_name}_pose.avi")  


        with PoseEstimationSession(self.settings, self.sources,
                                   self.pose_model, out_q) as sess:
            while not stop_evt.is_set():
                sess.run()
        out_q.put(BaseStage.sentinel)

    def from_config(self, config):
        self.settings = PoseEstimationSettings.from_config(config)

    def save_data(self, data_out):
        if 'openpose' in self.settings.output_format:
            json_path = os.path.join(data_out.source.output_dir, f"{data_out.file_name}.json")
            save_keypoints_to_openpose(json_path, data_out.keypoints, data_out.scores)
        # if 'deeplabcut' in self.settings.output_format:
        # if 'pose2sim' in self.settings.output_format:


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


def save_keypoints_to_deeplabcut(csv_file_path, all_keypoints, all_scores, timestamp):
    """
    Append keypoints and scores to a CSV file in the DeepLabCut format.
    This function supports dynamically adding new persons if more are detected over time.
    The CSV header is a MultiIndex for keypoint columns with a final timestamp column.
    
    INPUTS:
    - csv_file_path: path to the CSV file (accumulated over time)
    - all_keypoints: list of detected keypoints for each person (each person is a list of keypoints)
    - all_scores: list of confidence scores for each keypoint for each person
    - timestamp: timestamp of the frame (appended as the last column)
    """
    # Determine current detection dimensions
    new_num_persons = len(all_keypoints)  # may be 0 if no detections
    new_num_keypoints = len(all_keypoints[0]) if new_num_persons > 0 else 0

    # Helper: Build a MultiIndex header given max persons and keypoints per person.
    def build_multiindex(max_persons, num_keypoints):
        cols = []
        # Create keypoint columns for each person
        for person_idx in range(1, max_persons+1):
            for kp_idx in range(num_keypoints):
                for coord in ["x", "y", "likelihood"]:
                    cols.append(("deeplabcut", f"person_{person_idx}", f"part_{kp_idx}", coord))
        # Append the timestamp column at the end.
        cols.append(("timestamp", "", "", ""))
        return pd.MultiIndex.from_tuples(cols, names=["scorer", "individual", "bodypart", "coord"])

    # If the CSV file exists, read its header to determine the current dimensions.
    if os.path.exists(csv_file_path):
        try:
            # Read existing CSV with a 4-level header.
            existing_df = pd.read_csv(csv_file_path, header=[0,1,2,3])
            existing_persons = set()
            existing_keyparts = set()
            for col in existing_df.columns:
                if col[0] != "timestamp":
                    existing_persons.add(col[1])
                    existing_keyparts.add(col[2])
            if existing_persons:
                current_max_persons = max([int(p.split('_')[1]) for p in existing_persons])
            else:
                current_max_persons = 0
            if existing_keyparts:
                # Assumes keypoint parts are named as "part_0", "part_1", etc.
                current_num_keypoints = max([int(part.split('_')[1]) for part in existing_keyparts]) + 1
            else:
                current_num_keypoints = 0
        except Exception:
            current_max_persons = 0
            current_num_keypoints = 0
    else:
        current_max_persons = 0
        current_num_keypoints = 0

    # Determine updated dimensions: use the maximum of existing and current detection.
    max_persons = max(current_max_persons, new_num_persons)
    # For keypoints, if current detection provides any, use that; otherwise fall back on existing.
    num_keypoints = new_num_keypoints if new_num_keypoints > 0 else current_num_keypoints

    # Build the new header.
    new_header = build_multiindex(max_persons, num_keypoints)

    # If the file exists and the header dimensions have increased, update the entire CSV.
    if os.path.exists(csv_file_path) and (max_persons > current_max_persons or num_keypoints > current_num_keypoints):
        df_existing = pd.read_csv(csv_file_path, header=[0,1,2,3])
        # Reindex to add new columns (filling with NaN).
        df_existing = df_existing.reindex(columns=new_header)
        df_existing.to_csv(csv_file_path, index=False)

    # Build a new row that matches the header.
    # For each person from 1 to max_persons, use detection data if available; otherwise fill with NaN.
    row_data = []
    for person_idx in range(1, max_persons+1):
        if person_idx <= new_num_persons:
            keypoints_person = all_keypoints[person_idx - 1]
            scores_person = all_scores[person_idx - 1]
            for kp_idx in range(num_keypoints):
                if kp_idx < len(keypoints_person):
                    kp = keypoints_person[kp_idx]
                    score = scores_person[kp_idx]
                    row_data.extend([kp[0].item(), kp[1].item(), score.item()])
                else:
                    row_data.extend([float('nan'), float('nan'), float('nan')])
        else:
            for kp_idx in range(num_keypoints):
                row_data.extend([float('nan'), float('nan'), float('nan')])
    # Append the timestamp at the end.
    row_data.append(timestamp)

    # Create a DataFrame for the new row and append it.
    new_row_df = pd.DataFrame([row_data], columns=new_header)
    new_row_df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
