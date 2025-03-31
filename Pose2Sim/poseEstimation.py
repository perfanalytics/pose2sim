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

        self.progress_bar = tqdm(
            total=self.buffers_count,
            desc="Buffers Free",
            position=0,
            leave=True,
            colour='blue'
        )

    def update_progress_bar(self):
        self.progress_bar.n = self.available_buffers.qsize()
        self.progress_bar.refresh()

    def cleanup(self):
        self.progress_bar.close()
        for shm in self.buffers:
            shm.close()
            shm.unlink()


class PoseEstimatorWorker(multiprocessing.Process):
    def __init__(self, config, input_queue, output_queue, tracker_ready_event):
        super().__init__()
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

    def build_mosaic(self, frames_list):
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

        return mosaic, mosaic_data

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
                    self.sync_data[frame_data.idx][source.name] = (frame_data)

                if frames_list := self.get_frames_list():
                    mosaic, mosaic_data = self.build_mosaic(frames_list)
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
    def __init__(self, config, output_queue, buffer_manager, fps):
        multiprocessing.Process.__init__(self)
        BaseSynchronizer.__init__(self, config, buffer_manager)
        self.output_queue = output_queue
        self.fps = fps

    def run(self):
        self.init_video_writers()

        while not self.ended:
            try:
                result_item = self.output_queue.get_nowait()
                self.process_result_item(result_item)
            except queue.Empty:
                time.sleep(0.005)

        for vw in self.video_writers.values():
            vw.release()
        cv2.destroyAllWindows()

    def init_video_writers(self):
        if self.save_video and self.source_outputs:
            for source in self.config.sources:


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
    def __init__(self, config, buffer_manager, tracker_ready_event, fps):
        super().__init__()
        self.config = config
        self.buffer_manager = buffer_manager
        self.tracker_ready_event = tracker_ready_event

        self.interval = 1.0 / fps
        self.ended = False

    def run(self):
        self.tracker_ready_event.wait()
        last_capture_time = time.time()
        for source in self.config.sources:
            source.start_in_process()
            source.capture_ready_event.wait()
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
    config.check_pose_estimation()

    config.get_mosaic_params()

    logging.info(f"Model input size: {config.pose_model.det_input_size[0]}x{config.pose_model.det_input_size[1]}")

    buffer_manager = BufferManager(config)

    input_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()

    # Decide how many workers to start
    cpu_count = multiprocessing.cpu_count()
    initial_workers = 1

    if config.multi_workers:
        initial_workers = max(1, cpu_count - len(config.sources) - 4)

    logging.info(f"Starting {initial_workers} workers.")

    tracker_ready_event = multiprocessing.Event()

    fps = min(source.fps for source in config.sources)

    def start_new_worker():
        worker = PoseEstimatorWorker(
            config=config,
            input_queue=input_queue,
            output_queue=output_queue,
            tracker_ready_event=tracker_ready_event,
        )
        worker.start()
        return worker

    workers = [start_new_worker() for _ in range(initial_workers)]

    result_processor = OutputQueueProcessor(
        config=config,
        output_queue=output_queue,
        buffer_manager=buffer_manager,
        fps=fps,
    )
    result_processor.start()

    frame_processor = InputQueueProcessor(
        config=config,
        input_queue=input_queue,
        buffer_manager=buffer_manager,
    )
    frame_processor.start()

    capture_coordinator = CaptureCoordinator(
        config=config,
        buffer_manager=buffer_manager,
        tracker_ready_event=tracker_ready_event,
        fps=fps,
    )
    capture_coordinator.start()

    bar_pos = 0

    for source in config.sources:
        source.progress_bar.position = bar_pos
        bar_pos += 1

    buffer_manager.progress_bar.position = bar_pos
    bar_pos += 1

    if config.multi_workers:
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
            alive_workers = sum(not worker.ended for worker in workers)

            if config.multi_workers:
                worker_bar.n = alive_workers
                worker_bar.refresh()

                current_ended_count = sum(1 for source in config.sources if source.ended)
                ended_delta = current_ended_count - previous_ended_count
                if ended_delta > 0:
                    for _ in range(ended_delta):
                        logging.info("Spawning a new PoseEstimatorWorker.")
                        new_w = start_new_worker()
                        workers.append(new_w)
                        worker_bar.total = len(workers)
                    previous_ended_count = current_ended_count

            if alive_workers == 0:
                break

            if cv2.waitKey(1) & 0xFF == ord(" "):
                logging.info("Space bar pressed: stopping all sources.")
                for source in config.sources:
                    source.stop()

            time.sleep(0.05)

    except KeyboardInterrupt:
        logging.info("User interrupted pose estimation.")

    finally:
        # Stop capture coordinator
        capture_coordinator.stop()
        capture_coordinator.join()
        if capture_coordinator.is_alive():
            capture_coordinator.terminate()

        # Stop media sources
        for source in config.sources:
            source.stop()
            source.progress_bar.close()

        # Stop workers
        for w in workers:
            w.join(timeout=2)
            if w.is_alive():
                logging.warning(f"Forcibly terminating worker {w.pid}")
                w.terminate()

        # Free shared memory
        buffer_manager.cleanup()

        if config.combined_frames:
            frame_processor.stop()
            frame_processor.join(timeout=2)
            if frame_processor.is_alive():
                frame_processor.terminate()

        result_processor.stop()
        result_processor.join(timeout=2)
        if result_processor.is_alive():
            result_processor.terminate()

        if config.multi_workers:
            worker_bar.close()

        if config.display_detection:
            cv2.destroyAllWindows()


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
