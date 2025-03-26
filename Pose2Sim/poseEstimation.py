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

from Pose2Sim.source import WebcamSource, ImageSource, VideoSource
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

        while not self.stopped:
            active_sources = 0
            for source in self.config.sources:
                if not source.ended:
                    active_sources += 1
                    break

            # If no new frames and no active sources remain, stop
            if self.queue.empty() and active_sources == 0:
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

        if self.config.combined_frames:
            self.mosaic_dimensions = self.config.check_pose_estimation()

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

    available_memory = psutil.virtual_memory().available
    frame_bytes = config.pose_model.det_input_size[0] * config.pose_model.det_input_size[1] * 3
    n_buffers_total = int((available_memory / 2) / (frame_bytes * (len(config.sources) / (len(config.sources) + 1))))

    if not config.combined_frames:
        frame_buffer_count = n_buffers_total

        logging.info(f"Allocating {frame_buffer_count} buffers.")
    else:
        frame_buffer_count = len(config.sources) * 3

        logging.info(f"Allocating {frame_buffer_count} frame buffers.")

        pose_buffer_count = min(n_buffers_total - frame_buffer_count, 10000)

        logging.info(f"Allocating {pose_buffer_count} pose buffers.")

    frame_queue = multiprocessing.Queue(maxsize=frame_buffer_count)
    pose_queue = multiprocessing.Queue()
    if config.combined_frames:
        pose_queue = multiprocessing.Queue(maxsize=pose_buffer_count)
    result_queue = multiprocessing.Queue()

    shm_list = []

    frame_buffers = {}
    available_frame_buffers = multiprocessing.Queue(maxsize=frame_buffer_count)

    for i in range(frame_buffer_count):
        buf_uuid = str(uuid.uuid4())
        shm = shared_memory.SharedMemory(create=True, size=frame_bytes)
        frame_buffers[buf_uuid] = shm
        available_frame_buffers.put(buf_uuid)
        shm_list.append(shm)

    pose_buffers = {}
    available_pose_buffers = {}

    if config.combined_frames:
        available_pose_buffers = multiprocessing.Queue(maxsize=pose_buffer_count)
        for i in range(pose_buffer_count):
            buf_uuid = str(uuid.uuid4())
            shm = shared_memory.SharedMemory(create=True, size=frame_bytes * len(config.sources))
            pose_buffers[buf_uuid] = shm
            available_pose_buffers.put(buf_uuid)
            shm_list.append(shm)

    # Decide how many workers to start
    cpu_count = multiprocessing.cpu_count()

    if config.combined_frames:
        initial_workers = max(1, cpu_count - len(config.sources) - 3)
    else:
        initial_workers = max(1, cpu_count - len(config.sources) - 2)

    if not config.multi_workers:
        initial_workers = 1

    logging.info(f"Starting {initial_workers} workers.")

    tracker_ready_event = multiprocessing.Event()

    def spawn_new_worker():
        worker = PoseEstimatorWorker(
            config=config,
            queue=pose_queue if config.combined_frames else frame_queue,
            result_queue=result_queue,
            buffers=pose_buffers if config.combined_frames else frame_buffers,
            available_buffers=available_pose_buffers if config.combined_frames else available_frame_buffers,
            tracker_ready_event=tracker_ready_event,
        )
        worker.start()
        return worker

    workers = [spawn_new_worker() for _ in range(initial_workers)]

    command_queues = {}

    result_processor = ResultQueueProcessor(
        config=config,
        result_queue=result_queue,
        frame_buffers=frame_buffers,
        pose_buffers=pose_buffers,
        available_frame_buffers=available_frame_buffers,
        available_pose_buffers=available_pose_buffers,
    )
    result_processor.start()

    if config.combined_frames:
        frame_processor = FrameQueueProcessor(
            config=config,
            frame_queue=frame_queue,
            pose_queue=pose_queue,
            frame_buffers=frame_buffers,
            pose_buffers=pose_buffers,
            available_frame_buffers=available_frame_buffers,
            available_pose_buffers=available_pose_buffers,
        )
        frame_processor.start()

    # Start capture coordinator
    capture_coordinator = CaptureCoordinator(
        config=config,
        command_queues=command_queues,
        available_frame_buffers=available_frame_buffers,
        tracker_ready_event=tracker_ready_event
    )
    capture_coordinator.start()

    bar_pos = 0

    for source in config.sources:
        source.pb.position = bar_pos
        bar_pos += 1

    frame_buffer_bar = tqdm(
        total=frame_buffer_count,
        desc='Frame Buffers Free',
        position=bar_pos,
        leave=True,
        colour='blue'
    )
    bar_pos += 1

    if config.combined_frames:
        pose_buffer_bar = tqdm(
            total=pose_buffer_count,
            desc='Pose Buffers Free',
            position=bar_pos,
            leave=True,
            colour='blue'
        )
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
            # Update progress bars
            frame_buffer_bar.n = available_frame_buffers.qsize()
            frame_buffer_bar.refresh()

            if config.combined_frames:
                pose_buffer_bar.n = available_pose_buffers.qsize()
                pose_buffer_bar.refresh()

            alive_workers = sum(w.is_alive() for w in workers)

            if config.multi_workers:
                worker_bar.n = alive_workers
                worker_bar.refresh()

                current_ended_count = sum(1 for source in config.sources if source.ended)
                ended_delta = current_ended_count - previous_ended_count
                if ended_delta > 0:
                    for _ in range(ended_delta):
                        logging.info("Spawning a new PoseEstimatorWorker.")
                        new_w = spawn_new_worker()
                        workers.append(new_w)
                        worker_bar.total = len(workers)
                    previous_ended_count = current_ended_count

            # If all sources ended, queue empty, and no alive workers => done
            all_ended = all(source.ended for source in config.sources)
            if all_ended and frame_queue.empty() and pose_queue.empty() and alive_workers == 0:
                logging.info("All sources ended, queues empty, and worker finished. Exiting loop.")
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

        # Stop workers
        for w in workers:
            w.join(timeout=2)
            if w.is_alive():
                logging.warning(f"Forcibly terminating worker {w.pid}")
                w.terminate()

        # Free shared memory
        for shm in shm_list:
            shm.close()
            shm.unlink()

        if config.combined_frames:
            frame_processor.stop()
            frame_processor.join(timeout=2)
            if frame_processor.is_alive():
                frame_processor.terminate()

        result_processor.stop()
        result_processor.join(timeout=2)
        if result_processor.is_alive():
            result_processor.terminate()

        for source in config.sources:
            source.pb.close()
        frame_buffer_bar.close()
        if config.combined_frames:
            pose_buffer_bar.close()
        if config.multi_workers:
            worker_bar.close()

        logging.info("Pose estimation done. Exiting now.")
        logging.shutdown()

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
