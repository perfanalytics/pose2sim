#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#########################################
## SOURCES                             ##
#########################################
'''

import abc
import os
import glob
import csv
import cv2
import math
import logging
import numpy as np
import multiprocessing
import queue
import time

from datetime import datetime
from Pose2Sim.common import natural_sort_key


class BaseSource(abc.ABC):
    def __init__(self, config, data: dict, pose_model):
        self.pose_model = pose_model
        self.config = config
        self.data = data
        self.name = data.get("name")
        self.rotation = data.get("rotation", 0)

        self.extrinsics_files = {}
        self.intrinsics_files = {}

        self.calib_intrinsics = data.get("calib_intrinsics")
        self.calib_extrinsics = data.get("calib_extrinsics")
        self.rotation = data.get("rotation")

        self.ret, self.ret_int = None, None
        self.C, self.S, self.D, self.K, self.R, self.T = [], [], [], [], [], []

        self.idx = 0
        self.processed = 0
        self.ended = False

        self.create_output_folders()

    def start_in_process(self,
                         frame_buffers,
                         frame_queue,
                         shared_counts,
                         frame_size,
                         rotation,
                         command_queue,
                         start_timestamp=None):
        self.frame_buffers = frame_buffers
        self.frame_queue = frame_queue
        self.shared_counts = shared_counts
        self.frame_size = frame_size
        self.rotation = rotation
        self.command_queue = command_queue
        self.start_timestamp = start_timestamp if start_timestamp else time.time()

        self.init_capture()

        self.process = multiprocessing.Process(target=self._capture_loop)
        self.process.start()

    def _capture_loop(self):
        while not self.ended:
            try:
                cmd = self.command_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if cmd is None:
                self.stop()
                break

            if isinstance(cmd, tuple):
                if cmd[0] == "CAPTURE_FRAME":
                    buffer_name = cmd[1]

                    frame, idx, is_placeholder = self.get_frame()
                    if frame is None:
                        self.stop()
                        break

                    if self.shared_counts and self.shared_counts[self.data['id']].get('processed'):
                        with self.shared_counts[self.data['id']]['processed'].get_lock():
                            self.shared_counts[self.data['id']]['processed'].value += 1

                    frame, transform_info = transform(frame,
                                                      self.frame_size[0],
                                                      self.frame_size[1],
                                                      True,
                                                      self.rotation)

                    self.send_frame(frame, idx, buffer_name, is_placeholder, transform_info)

                elif cmd[0] == "STOP_CAPTURE":
                    self.stop()
                    break

                else:
                    pass

        logging.info(f"[{self.name}] Capture loop ended.")

    def get_frame(self):
        frame, placeholder = self.read()
        if frame is None:
            return None, None, False
        current_idx = self.idx
        self.idx += 1

        return frame, current_idx, placeholder

    def send_frame(self, frame, idx, buffer_name, is_placeholder, transform_info):
        shm = self.frame_buffers[buffer_name]
        np_frame = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
        np_frame[:] = frame

        if self.data.get('type') in ('video', 'images'):
            timestamp_str = get_frame_utc_timestamp(self.start_timestamp, idx, self.frame_rate)
        else:
            timestamp_str = get_formatted_utc_timestamp()

        item = (
            buffer_name,
            timestamp_str,
            idx,
            frame.shape,
            frame.dtype.str,
            self.data,
            is_placeholder,
            transform_info
        )
        self.frame_queue.put(item)

        if not is_placeholder and self.shared_counts is not None:
            with self.shared_counts[self.data['id']]['queued'].get_lock():
                self.shared_counts[self.data['id']]['queued'].value += 1

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")

    def get_calib_files(self, folder, extension, calibration_name):
        calib_folder = os.path.join(self.config.calib_dir, folder)
        if not os.path.isdir(calib_folder):
            logging.warning(
                f"[{self.name} - {calibration_name}] Calibration skipped: The specified folder does not exist -> '{folder}'"
            )
            return {}

        files = glob.glob(os.path.join(calib_folder, f"*{extension}"))
        if not files:
            logging.warning(
                f"[{self.name} - {calibration_name}] Calibration skipped: No files with the extension '{extension}' found in folder '{folder}'."
            )
            return {}

        return files

    def extract_frames(self, calib_type):
        files = self.intrinsics_files if calib_type == 'intrinsic' else self.extrinsics_files
        folder = self.calib_intrinsics if calib_type == 'intrinsic' else self.calib_extrinsics

        try:
            cap = cv2.VideoCapture(files[0])
            if not cap.isOpened():
                raise Exception("File could not be opened.")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count == 1:
                return

        except Exception as e:
            logging.error(f"Files found in {folder} are not images or videos.")
            raise ValueError(f"Files found in {folder} are not images or videos.")

        new_files = glob.glob(os.path.join(folder, self.name + '*' + '.png'))
        if new_files and not self.config.overwrite_extraction:
            logging.info("Frames have already been extracted and overwrite_extraction is False.")
            if calib_type == 'intrinsic':
                self.intrinsics_files = new_files
            else:
                self.extrinsics_files = new_files
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            logging.error("FPS is 0, cannot extract frames.")
            raise ValueError("FPS is 0, cannot extract frames.")
        fps = round(fps)

        frame_nb = 0
        logging.info(f"[{self.name}]Extracting frames")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Extract one frame every (fps * extract_every_N_sec) frames.
            if frame_nb % (fps * self.config.extract_every_N_sec) == 0:
                img_path = os.path.join(folder, self.name + '_' + str(frame_nb).zfill(5) + '.png')
                cv2.imwrite(img_path, frame)
            frame_nb += 1
        cap.release()

        new_files = glob.glob(os.path.join(folder, self.name + '*' + '.png'))
        new_files.sort()
        if calib_type == 'intrinsic':
            self.intrinsics_files = new_files
        else:
            self.extrinsics_files = new_files

    def calculate_calibration_residuals(self):
        if len(self.S) != 0 and len(self.D) != 0 and len(self.K) != 0 and len(self.R) != 0 and len(self.T) != 0:
            f_px = self.K[0, 0]
            Dm = np.linalg.norm(self.T)

            if self.ret_int is not None:
                self.ret_int_px = np.around(np.array(self.ret_int), decimals=3)
                self.ret_int_mm = np.around(self.ret_int_px * Dm * 1000 / f_px, decimals=3)
                logging.info(f"[{self.name} - intrinsic] Intrinsic error: {self.ret_int_px} px, which corresponds to {self.ret_int_mm} mm.")
            if self.ret is not None:
                self.ret_px = np.around(np.array(self.ret), decimals=3)
                self.ret_mm = np.around(self.ret_px * Dm * 1000 / f_px, decimals=3)
                logging.info(f"[{self.name} - extrinsic] Residual (RMS) calibration error: {self.ret_px} px, which corresponds to {self.ret_mm} mm.")

    def create_output_folders(self):

        os.makedirs(self.config.pose_dir, exist_ok=True)

        if isinstance(self, WebcamSource):
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_name = f"{self.name}_{now}"

            if self.config.webcam_recording:
                self.output_video_path = os.path.join(self.config.pose_dir, f"{output_dir_name}_record.avi")
                self.csv_file = os.path.join(self.config.pose_dir, f"{output_dir_name}_timestamps.csv")

        else:
            output_dir_name = self.name
            self.json_output_dir = os.path.join(self.config.pose_dir, f"{output_dir_name}_json")
            os.makedirs(self.json_output_dir, exist_ok=True)

        self.img_output_dir = None
        if self.config.save_files[0]:
            self.img_output_dir = os.path.join(self.config.pose_dir, f"{output_dir_name}_img")
            os.makedirs(self.img_output_dir, exist_ok=True)

        if self.config.save_files[1]:
            self.output_video_path = os.path.join(self.config.pose_dir, f"{output_dir_name}_pose.avi")
 
    @property
    @abc.abstractmethod
    def frame_rate(self):
        pass

    @property
    @abc.abstractmethod
    def dimensions(self):
        pass

    @abc.abstractmethod
    def init_capture(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass


class WebcamSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.camera_index = data.get("path")
        self.backend = data.get("backend", None)
        self.capture_codec = data.get("capture_codec")
        self.record_codec = data.get("record_codec", None)
        self.ready = False
        self.readed = 0

    def init_capture(self):
        backend_map = {
            "MSMF": cv2.CAP_MSMF,
            "DSHOW": cv2.CAP_DSHOW,
            "V4L2": cv2.CAP_V4L2,
            "FFMPEG": cv2.CAP_FFMPEG,
            "GSTREAMER": cv2.CAP_GSTREAMER,
            "VFW": cv2.CAP_VFW,
            "WINRT": cv2.CAP_WINRT,
            "AVFOUNDATION": cv2.CAP_AVFOUNDATION,
            "DC1394": cv2.CAP_IEEE1394,
        }

        backend_flag = cv2.CAP_ANY
        if self.backend is not None and self.backend.upper() in backend_map:
            backend_flag = backend_map[self.backend.upper()]
            logging.info(f"[{self.name}] Using backend: {self.backend} (flag: {backend_flag})")

        cap = cv2.VideoCapture(self.camera_index, backend_flag)
        if not cap.isOpened():
            logging.error(f"[{self.name}] Unable to open the camera index: {self.camera_index}.")
            raise ValueError(f"[{self.name}] Unable to open the camera index: {self.camera_index}.")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        capture_fourcc = cv2.VideoWriter_fourcc(*self.capture_codec)
        if not cap.set(cv2.CAP_PROP_FOURCC, capture_fourcc):
            logging.warning(f"[{self.name}] Unable to set capture codec to {self.capture_codec}.")

        self.dimensions(cap)
        self.frame_rate(cap)

        if self.config.webcam_recording:
            record_fourcc = cv2.VideoWriter_fourcc(*self.record_codec)
            self.raw_writer = cv2.VideoWriter(self.output_video_path, record_fourcc, self.fps, (self.width, self.height))

        csvwriter = csv.writer(self.csv_file)
        return (cap, csvwriter)

    def read(self, init, capture):
        (cap, csvwriter) = init
        ret, frame = cap.read()

        hardware_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if hardware_ts and hardware_ts > 0:
            timestamp = hardware_ts / 1000.0
        else:
            timestamp = time.time()

        if self.config.webcam_recording:
            self.readed += 1
            self.raw_writer.write(frame)
            csvwriter.writerow([self.readed, timestamp])
        if capture:
            self.idx += 1
            if not ret or frame is None:
                logging.error(f"[{self.name}] Unable to read the frame.")
                return None, timestamp
            return frame, timestamp

    def frame_rate(self, cap):
        if self.config.frame_rate == "auto":
            cap.set(cv2.CAP_PROP_FPS, 1000)
            self.fps = self.measure_actual_fps(cap, 60)
            logging.info(f"[{self.name}] Measured FPS: {self.fps:.2f}")
        else:
            self.fps = self.config.frame_rate

    def dimensions(self, cap):
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        det_width, det_height = self.pose_model.det_input_size
        if self.width >= self.height:
            scale = det_width / self.width
            target_width = det_width
            target_height = int(self.height * scale)
        else:
            scale = det_height / self.height
            target_height = det_height
            target_width = int(self.width * scale)

        if self.width != target_width or self.height != target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.width, self.height = self.height, self.width

        logging.info(f"[{self.name}] Dimensions after rotation ({self.rotation}°): {self.width}x{self.height}")

    def measure_actual_fps(self, cap, num_frames):
        for _ in range(10):
            cap.grab()
        start_time = time.time()
        frames_captured = 0
        for _ in range(num_frames):
            ret, _ = cap.read()
            if ret:
                frames_captured += 1
        elapsed = time.time() - start_time
        return math.floor(frames_captured / elapsed)


class VideoSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.video_path = data.get("path")
        self.timestamps = data.get("timestamps")

    def init_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"[{self.name}] Cannot open video file: {self.video_path}.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.frame_range[0])
        self.idx = self.config.frame_range[0]

        self.dimensions(cap)
        self.frame_rate(cap)

        return cap

    def read(self, cap, capture):
        if capture:
            ret, frame = cap.read()
            if self.timestamps:
                timestamp = self.timestamps[self.idx]
            else:
                timestamp = self.start_time + (self.idx / self.fps)
            self.idx += 1
            if not ret or frame is None:
                logging.error(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
                return None
            return frame, timestamp

    def frame_rate(self, cap):
        if self.config.frame_rate == "auto":
            self.fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
            logging.info(f"[{self.name}] Measured FPS: {self.fps:.2f}")
        else:
            self.fps = self.config.frame_rate

    def dimensions(self, cap):
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        det_width, det_height = self.pose_model.det_input_size
        if self.width >= self.height:
            scale = det_width / self.width
            target_width = det_width
            target_height = int(self.height * scale)
        else:
            scale = det_height / self.height
            target_height = det_height
            target_width = int(self.width * scale)

        if self.width != target_width or self.height != target_height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.width, self.height = self.height, self.width

        logging.info(f"[{self.name}] Dimensions after rotation ({self.rotation}°): {self.width}x{self.height}")


class ImageSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension")
        self.image_files = None
        self.timestamps = data.get("timestamps", None)

    def init_capture(self):
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, self.image_extension)), key=natural_sort_key)
        if not self.image_files:
            logging.error(f"[{self.name}] No images found in {self.image_dir} with extension {self.image_extension}.")
            raise ValueError(f"[{self.name}] No images found in {self.image_dir} with extension {self.image_extension}.")
        self.idx = self.config.frame_range[0]

        self.dimensions()
        self.frame_rate()

        return None

    def read(self, _, capture):
        if capture:
            if self.idx not in self.config.frame_ranges:
                self.stop()
            path = self.image_files[self.idx]
            if self.timestamps:
                timestamp = self.timestamps[self.idx]
            else:
                timestamp = self.start_time + (self.idx / self.fps)
            frame = cv2.imread(path)
            self.idx += 1
            if frame is None:
                logging.error(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
                return None
            return frame, timestamp

    def frame_rate(self):
        if self.config.frame_rate == "auto":
            logging.error(f"Frame rate cannot be set to 'auto' because '{self.name}' is an image source.")
            raise ValueError(f"Frame rate cannot be set to 'auto' because '{self.name}' is an image source.")
        else:
            self.fps = self.config.frame_rate

    def dimensions(self):
        if not self.image_files:
            logging.error(f"[{self.name}] Unable to get images from {self.image_dir}.")
            raise ValueError(f"[{self.name}] Unable to get images from {self.image_dir}.")
        test_img = cv2.imread(self.image_files[0])
        if test_img is None:
            logging.error(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
            raise ValueError(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
        self.height, self.width = test_img.shape[:2]

        rotation = self.rotation if self.rotation is not None else 0
        if rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (rotation % 180) == 90:
            self.width, self.height = self.height, self.width
        logging.info(f"[{self.name}] Dimensions after rotation ({rotation}°): {self.width}x{self.height}")
