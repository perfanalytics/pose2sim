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
import multiprocessing
import psutil
import cv2
import uuid
import numpy as np

from multiprocessing import shared_memory
from tqdm import tqdm

from rtmlib import draw_skeleton

from Pose2Sim.source import FrameData
from deep_sort_realtime.deepsort_tracker import DeepSort
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


# CLASSES
class BufferManager:
    def __init__(self, config):
        available_memory = psutil.virtual_memory().available
        buffer_size = config.pose_model.det_input_size[0] * config.pose_model.det_input_size[1] * 3
        self.buffers_count = min(int((available_memory / 2) / (buffer_size * (len(config.sources) / (len(config.sources) + 1)))), 10000)

        logging.info(f"Allocating {self.buffers_count} buffers.")

        self.buffers = {}
        self.available_buffers = multiprocessing.Queue(maxsize=self.buffers_count)

        for _ in range(self.buffers_count):
            buf_uuid = str(uuid.uuid4())
            shm = shared_memory.SharedMemory(name=buf_uuid, create=True, size=buffer_size)
            self.buffers[buf_uuid] = shm
            self.available_buffers.put(buf_uuid)

    def cleanup(self):
        for shm in self.buffers:
            shm.close()
            shm.unlink()


class PoseEstimatorWorker(multiprocessing.Process):
    def __init__(self, config, input_queue, output_queue, tracker_ready_event):
        super().__init__(daemon=True)
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.tracker_ready_event = tracker_ready_event

        self.ended = True
        self.prev_keypoints = {}
        self.prev_bboxes = {}

        if self.config.tracking_mode == 'deepsort' and self.config.multi_person:
            self.deepsort_tracker = DeepSort(**self.config.get_deepsort_params())

    def run(self):
        model = self.config.pose_model.pose_model_enum.model_class(mode=self.config.pose_model.mode,
            to_openpose=self.config.pose_model.output_format,
            backend=self.config.pose_model.backend,
            device=self.config.pose_model.device)

        try:
            self.det_model = model.det_model
        except AttributeError:  # rtmo
            self.det_model = None
        self.pose_model = model.pose_model

        self.tracker_ready_event.set()

        while not self.ended:
            if all(source.ended for source in self.config.sources) and self.input_queue.isEmpty():
                self.ended = True
                break

            frame_data = self.input_queue.get()
            frame = np.ndarray(frame_data.shape, dtype=frame_data.dtype, buffer=frame_data.buffer.buf)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if not frame_data.placeholder:
                if self.det_model is not None:
                    if frame_data.idx % self.det_frequency == 0:
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
                    frame_data.keypoints, frame_data.scores = self.pose_model(frame, bboxes=bboxes)
                else:  # rtmo
                    frame_data.keypoints, frame_data.scores = self.pose_model(frame)

                if self.multi_person:
                    if self.tracking_mode == 'deepsort':
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


class BaseSynchronizer:
    def __init__(self, config, buffer_manager):
        self.config = config
        self.buffer_manager = buffer_manager

        self.sync_data = {}
        self.ended = False

        self.video_writers = {}

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

    def stop(self):
        self.ended = True


class InputQueueProcessor(multiprocessing.Process, BaseSynchronizer):
    def __init__(self, config, input_queue, buffer_manager):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, config, buffer_manager)
        self.input_queue = input_queue

    def run(self):
        while not self.ended:
            if all(not sources.ended for sources in self.config.sources):
                self.stop()
                break

            if self.config.combined_frames:
                for source in self.config.sources:
                    frame_data = source.frame_queue.get()
                    if frame_data.idx not in self.sync_data:
                        self.sync_data[frame_data.idx] = {}
                    self.sync_data[frame_data.idx][frame_data.source.name] = frame_data

                if frames_list := self.get_frames_list():
                    mosaic = np.zeros(
                        (self.config.pose_model.det_input_size[0], self.config.pose_model.det_input_size[1], 3),
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
                            self.buffer_manager.available_buffers.put(frame_data.buffer.name)

                            source = frame_data.source
                            mosaic[source.y_offset:source.y_offset + source.height,
                                source.x_offset:source.x_offset + source.width] = frame

                    avg_timestamp = np.mean(timestamps)
                    mosaic_data = FrameData(None, avg_timestamp, idx)
                    buffer = self.buffer_manager.available_buffers.get()
                    shm = self.buffer_manager.buffers[buffer]
                    mosaic_data.shape = mosaic.shape
                    mosaic_data.dtype = mosaic.dtype.str
                    mosaic_data.buffer = shm
                    np_mosaic = np.ndarray(mosaic_data.shape, dtype=mosaic_data.dtype, buffer=shm.buf)
                    np_mosaic[:] = mosaic

                    self.input_queue.put(mosaic_data)
            else:
                for source in self.config.sources:
                    self.input_queue.put(source.frame_queue.get())


class OutputQueueProcessor(multiprocessing.Process, BaseSynchronizer):
    def __init__(self, config, output_queue, buffer_manager):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, config, buffer_manager)
        self.output_queue = output_queue

    def run(self):
        if self.save_video and self.config.save_files[0]:
            for source in self.config.sources:
                source.intit_video_writer()

        while not self.ended:
            try:
                frame_data = self.output_queue.get()
                self.process_result_item(frame_data)
            except queue.Empty:
                time.sleep(0.005)

        cv2.destroyAllWindows()

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
        self.buffer_manager.available_buffers.put(mosaic_data.buffer.name)

        if self.save_images or self.save_video or self.display_detection:
            mosaic = draw_skeleton(mosaic, mosaic_data.keypoints, mosaic_data.scores)
            if self.display_detection:
                self.show_mosaic(mosaic)

        for source in self.config.sources:
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
            )

            frame = mosaic[y0:y1, x0:x1]
            self.handle_output(frame, frame_data)

    def handle_individual_frames(self, frame_data):
        if frame_data.idx not in self.sync_data:
            self.sync_data[frame_data.idx] = {}
        self.sync_data[frame_data.idx][frame_data.source.name] = frame_data
        if frames_list := self.get_frames_list():
            if self.display_detection:
                mosaic = np.zeros(
                    (self.config.pose_model.det_input_size[0], self.config.pose_model.det_input_size[1], 3),
                    dtype=np.uint8
                )
            for frame_data in frames_list:
                if not frame_data.placeholder:
                    frame = np.ndarray(
                        frame_data.shape,
                        dtype=np.dtype(frame_data.dtype),
                        buffer=frame_data.buffer.buf,
                    )
                    self.buffer_manager.available_buffers.put(frame_data.buffer.name)
                    if self.config.save_files[0] or self.config.save_files[1] or self.config.display_detection:
                        frame = draw_skeleton(frame, frame_data.keypoints, frame_data.scores)
                    self.handle_output(frame, frame_data)
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
            frame = draw_skel(frame, valid_X, valid_Y, self.pose_model)
            return frame

    def show_mosaic(self, mosaic):
        cv2.imshow("Display", mosaic)
        key = cv2.waitKey(1)
        if key in [ord('q'), 27]:
            logging.info("User closed display.")
            self.stop()

    def handle_output(self, frame, frame_data):
        file_name = f"{frame_data.source.name}_{frame_data.timestamp}_{frame_data.idx:06d}"

        if 'openpose' in self.output_format:
            json_path = os.path.join(frame_data.source.output_dir, f"{file_name}.json")
            save_keypoints_to_openpose(json_path, frame_data.keypoints, frame_data.scores)
        if self.save_images:
            cv2.imwrite(os.path.join(frame_data.source.img_output_dir, f"{file_name}.jpg"), frame)
        if self.save_video:
            frame_data.source.video_writer.write(frame)


class CaptureCoordinator(multiprocessing.Process):
    def __init__(self, config, buffer_manager, tracker_ready_event):
        super().__init__()
        self.config = config
        self.buffer_manager = buffer_manager
        self.tracker_ready_event = tracker_ready_event

        self.ended = False

    def run(self):
        self.tracker_ready_event.wait()
        last_capture_time = time.time()
        for source in self.config.sources:
            source.start_in_process()
            source.capture_ready_event.wait()
        self.config.set_fps(min(source.fps for source in self.config.sources))
        self.interval = 1.0 / self.config.fps
        while not self.ended:
            if all(source.ended for source in self.config.sources):
                self.stop()
                break

            elapsed = time.time() - last_capture_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)

            if self.buffer_manager.available_buffers.qsize() == self.buffer_manager.buffers_count:
                empty = True

            if self.mode == 'continuous' or (self.mode == 'alternating' and empty):
                if self.available_frame_buffers.qsize() >= len(self.config.sources):
                    for source in self.config.sources:
                        buffer = self.buffer_manager.available_buffers.get()
                        shared_memory = self.buffer_manager.buffers[buffer]
                        self.source.command_queues.put(("CAPTURE_FRAME", shared_memory))
                    last_capture_time = time.time()
                else:
                    empty = False

    def stop(self):
        self.ended = True
        for source in self.config.sources:
            self.source.command_queues.put(None)

class PoseEstimationSession:
    def __init__(self, config):
        self.config = config
        self.buffer_manager = BufferManager(config)
        self.capture_coord = None
        self.frame_proc = None
        self.result_proc = None
        self.workers = []

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        self.config.check_pose_estimation()
        self.config.get_mosaic_params()

        self._launch_processes()

        logging.info("PoseEstimationSession started.")

    def _launch_processes(self):

        tracker_ready = multiprocessing.Event()
        self.capture_coord = CaptureCoordinator(
            config=self.config,
            buffer_manager=self.buffer_manager,
            tracker_ready_event=tracker_ready,
        )
        self.capture_coord.start()

        self.input_queue  = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        self.frame_proc = InputQueueProcessor(
            config=self.config,
            input_queue=self.input_queue,
        )
        self.result_proc = OutputQueueProcessor(
            config=self.config,
            output_queue=self.output_queue,
        )
        self.frame_proc.start()
        self.result_proc.start()

        self._spawn_workers(tracker_ready)

        self._init_progress_bars()

    def _spawn_workers(self, tracker_ready):
        cpu = multiprocessing.cpu_count()
        n_workers = max(1, cpu - len(self.config.sources) - 4)
        logging.info(f"Starting {n_workers} workers.")

        def new_worker():
            w = PoseEstimatorWorker(
                cfg_path=self.config,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                tracker_ready_event=tracker_ready,
            )
            w.start()
            return w

        self.workers = [new_worker() for _ in range(n_workers)]

    def _init_progress_bars(self):
        self.bars = {}
        bar_pos = 0
        for src in self.config.sources:
            src.progress_bar.position = bar_pos
            bar_pos += 1

        self.bars["buffers"] = tqdm(
            total=self.buffer_manager.buffers_count,
            desc="Buffers Free", position=bar_pos, leave=True
        )
        bar_pos += 1

        self.bars["workers"] = tqdm(
            total=len(self.workers),
            desc="Active Workers", position=bar_pos, leave=True
        )

    def run(self):
        try:
            prev_ended = 0
            while not self._stop_event.is_set():
                # Maj barres
                self.bars["buffers"].n = self.buffer_manager.available_buffers.qsize()
                self.bars["buffers"].refresh()

                alive = sum(not w.ended for w in self.workers)
                self.bars["workers"].n = alive
                self.bars["workers"].refresh()

                if alive == 0:
                    break

                if cv2.waitKey(1) & 0xFF == ord(" "):
                    logging.info("Space bar pressed: stopping sources.")
                    for s in self.config.sources:
                        s.stop()

                time.sleep(0.05)

        except (KeyboardInterrupt, SystemExit):
            logging.info("User interrupted pose estimation.")
        finally:
            self.stop()

    def stop(self):
        if self._stop_event.is_set():
            return
        self._stop_event.set()

        if self.capture_coord:
            self.capture_coord.stop()
            self.capture_coord.join(timeout=2)

        for s in getattr(self.config, "sources", []):
            s.stop()
            if hasattr(s, "progress_bar"):
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

        if self.buffer_manager:
            self.buffer_manager.cleanup()
        for b in self.bars.values():
            b.close()
        if self.config.display_detection:
            cv2.destroyAllWindows()

        logging.info("PoseEstimationSession stopped.")


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
