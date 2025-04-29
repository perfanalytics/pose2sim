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

from tqdm import tqdm
from datetime import datetime, timezone, timedelta
from Pose2Sim.common import natural_sort_key
from dataclasses import dataclass
from multiprocessing import shared_memory


@dataclass
class FrameData:
    source: dict
    timestamp: float
    idx: int
    buffer: shared_memory.SharedMemory = None
    shape: tuple = None
    dtype: str = None
    placeholder: bool = False
    keypoints: dict = None
    scores: dict = None
    file_name: str = None


class BaseSource(abc.ABC):
    def __init__(self, config, data: dict, pose_model):
        self.pose_model = pose_model
        self.config = config.config_dict
        self.data = data
        self.name = data.get("name")
        self.rotation = data.get("rotation", 0)

        self.calib_intrinsics = data.get("calib_intrinsics")
        self.calib_extrinsics = data.get("calib_extrinsics")
        self.rotation = data.get("rotation")

        self.ret, self.ret_int = None, None
        self.C, self.S, self.D, self.K, self.R, self.T = [], [], [], [], [], []

        self.idx = 0
        self.processed = 0
        self.ended = False

        self.x_offset = 0
        self.y_offset = 0
        self.desired_width = 0
        self.desired_height = 0

        self.command_queue = multiprocessing.Queue()
        self.frame_queue = multiprocessing.Queue()

        self.scale = 1

        self.capture_ready_event = multiprocessing.Event()

        self.video_writer = None

    def start_in_process(self):
        self.init_capture()

        self.process = multiprocessing.Process(target=self._capture_loop, daemon=True)
        self.process.start()

    def _capture_loop(self):
        self.capture_ready_event.set()
        while not self.ended:

            frame, frame_data = self.read()

            try:
                cmd = self.command_queue.get(block=False)
            except queue.Empty:
                continue

            if cmd is None:
                self.stop()
                break

            if isinstance(cmd, tuple):
                if cmd[0] == "CAPTURE_FRAME":
                    shm = cmd[1]

                    if frame is None:
                        frame_data.placeholder = True
                    else:
                        frame_data.shape = frame.shape
                        frame_data.dtype = frame.dtype.str
                        frame_data.buffer = shm
                        np_frame = np.ndarray(frame_data.shape, dtype=frame_data.dtype, buffer=shm.buf)
                        np_frame[:] = frame

                    self.frame_queue.put(frame_data)

                elif cmd[0] == "STOP_CAPTURE":
                    self.stop()
                    break

                else:
                    pass

    def get_calib_files(self, calib_type, calib_folder, extension, extract_every_N_sec=1):
        folder = self.calib_intrinsics if calib_type == 'intrinsic' else self.calib_extrinsics
        if folder == "live":
            #TODO: Extract frames from webcam
            logging.error(f"[{self.name}] Live calibration not already implemented.")
            raise ValueError(f"[{self.name}] Live calibration not already implemented.")
        else :
            if not os.path.isdir(calib_folder):
                logging.warning(f"[{self.name} - {calib_type}] Calibration skipped: The specified folder has not been found '{folder}'")
            else:
                files = glob.glob(calib_folder)
                if not files:
                    logging.error(f"[{self.name} - {calib_type}] No files found in folder '{folder}'.")
                    raise ValueError(f"[{self.name} - {calib_type}] No files found in folder '{folder}'.")
                else:
                    files = glob.glob(os.path.join(calib_folder, f"*{extension}"))
                    if not files:
                        try:
                            cap = cv2.VideoCapture(files[0])
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                if fps == 0:
                                    cap.release()
                                    logging.error(f"[{self.name} - {calib_type}] FPS is 0, cannot extract frames.")
                                    raise ValueError(f"[{self.name} - {calib_type}] FPS is 0, cannot extract frames.")
                                fps = round(fps)

                            logging.info(f"[{self.name} - {calib_type}] Extracting frames")
                            frame_nb = 0
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_nb += 1
                                if frame_nb % (fps * extract_every_N_sec) == 0:
                                    img_path = os.path.join(folder, self.name + '_' + str(frame_nb).zfill(5) + '.png')
                                    cv2.imwrite(img_path, frame)
                            cap.release()

                            return glob.glob(os.path.join(folder, self.name + '*' + '.png')).sort

                        except Exception as e:
                            logging.error(f"[{self.name} - {calib_type}] Files found in {folder} could not be read.")
                            raise ValueError(f"[{self.name} - {calib_type}] Files found in {folder} could not be read.")
                    else:
                        return files

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

    def intit_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            self.config.fps,
            (self.desired_width, self.desired_height)
        )

    @abc.abstractmethod
    def frame_rate(self):
        pass

    @abc.abstractmethod
    def dimensions(self):
        pass

    @abc.abstractmethod
    def init_capture(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass


class WebcamSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.camera_index = data.get("path")
        self.backend = data.get("backend", None)
        self.capture_codec = data.get("capture_codec")
        self.record_codec = data.get("record_codec", None)
        self.readed = 0

        self.progress_bar = tqdm(
            total=0,
            desc=f"\033[32{self.name} (not connected)\033[0m",
            position=0,
            leave=True,
            bar_format="{desc}"
        )

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
            self.raw_writer = cv2.VideoWriter(self.output_video_path, record_fourcc, self.fps, (self.native_width, self.native_height))

        csvwriter = csv.writer(self.csv_file)
        return (cap, csvwriter)

    def read(self, init, capture):
        (cap, csvwriter) = init
        ret, frame = cap.read()

        hardware_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if hardware_ts and hardware_ts > 0:
            dt = datetime.fromtimestamp(hardware_ts / 1000.0, timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        timestamp = dt.strftime("%Y%m%dT%H%M%S") + f"{dt.microsecond:06d}"

        frame_data = FrameData(self, timestamp, self.idx)

        if not ret or frame is None:
            if capture:
                logging.error(f"[{self.name}] Unable to read the frame.")
                return None, frame_data
            pass

        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.scale != 1:
            frame = cv2.resize(frame, (self.desired_width, self.desired_height), interpolation=cv2.INTER_AREA)

        if self.config.webcam_recording:
            self.readed += 1
            self.raw_writer.write(frame)
            csvwriter.writerow([self.readed, timestamp])

        if capture:
            self.progress_bar.set_description_str(
                f"\033[32m{self.name}\033[0m : {self.processed}/{self.idx} processed/read"
            )
            self.progress_bar.refresh()
            self.idx += 1
            return frame, frame_data

    def frame_rate(self, cap):
        if self.config.frame_rate == "auto":
            cap.set(cv2.CAP_PROP_FPS, 1000)
            self.fps = self.measure_actual_fps(cap, 60)
            logging.info(f"[{self.name}] Measured FPS: {self.fps:.2f}")
        else:
            self.fps = self.config.frame_rate

    def dimensions(self, cap):
        self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.scale = min(self.desired_width / self.native_width, self.desired_height / self.native_height)

        self.target_width = int(self.native_width * self.scale)
        self.target_height = int(self.native_height * self.scale)

        if self.scale != 1 and self.config.webcam_recording:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.scale = min(self.desired_width / self.native_width, self.desired_height / self.native_height)

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.target_width, self.target_height = self.target_height, self.target_width

        self.x_offset = self.x_offset + ((self.desired_width - self.target_width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.target_height) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.target_width}x{self.target_height}")

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

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")
        self.progress_bar.set_description_str(
            f"\033[31m{self.name} (Ended)\033[0m : {self.processed}/{self.idx}"
        )
        self.progress_bar.refresh()
        if self.video_writer:
            self.video_writer.release()

class VideoSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.video_path = data.get("path")
        self.timestamps = data.get("timestamps")

        self.progress_bar = tqdm(
            total=0,
            desc=f"\033[32m{self.name}\033[0m",
            position=0,
            leave=True
        )

    def init_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"[{self.name}] Cannot open video file: {self.video_path}.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.frame_range[0])

        self.progress_bar.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - self.config.frame_range[0]
        self.idx = self.config.frame_range[0]

        self.dimensions(cap)
        self.frame_rate(cap)

        ts = os.path.getmtime(self.video_path)
        self.start_timestamp = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=timezone.utc)

        return cap

    def read(self, cap, capture):
        if capture:           
            if self.idx not in self.config.frame_ranges:
                self.stop()

            self.progress_bar.n = self.idx
            self.progress_bar.refresh()

            ret, frame = cap.read()

            if self.timestamps:
                timestamp = self.timestamps[self.idx]
            else:
                frame_time = self.start_timestamp + timedelta(seconds=(self.idx  / self.fps))
                timestamp = frame_time.strftime("%Y%m%dT%H%M%S") + f"{frame_time.microsecond:06d}"

            frame_data = FrameData(self, timestamp, self.idx)
            self.idx += 1
            if not ret or frame is None:
                logging.error(f"[{self.name}] Unable to read the image from {self.video_path} at frame {self.idx}.")
                return None, frame_data

            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if self.scale != 1:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            return frame, frame_data

    def frame_rate(self, cap):
        if self.config.frame_rate == "auto":
            self.fps = math.floor(cap.get(cv2.CAP_PROP_FPS))
            logging.info(f"[{self.name}] Measured FPS: {self.fps:.2f}")
        else:
            self.fps = self.config.frame_rate

    def dimensions(self, cap):
        self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.scale = min(self.desired_width / self.native_width, self.desired_height / self.native_height)

        self.target_width = int(self.native_width * self.scale)
        self.target_height = int(self.native_height * self.scale)

        if self.scale != 1:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self.native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.scale = min(self.desired_width / self.native_width, self.desired_height / self.native_height)

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.target_width, self.target_height = self.target_height, self.target_width

        self.x_offset = self.x_offset + ((self.desired_width - self.target_width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.target_height) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.target_width}x{self.target_height}")

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")
        self.progress_bar.set_description_str(f"\033[31m{self.name} (Ended)\033[0m")
        self.progress_bar.refresh()
        if self.video_writer:
            self.video_writer.release()


class ImageSource(BaseSource):
    def __init__(self, subconfig, data, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension")
        self.image_files = None
        self.timestamps = data.get("timestamps", None)

        self.progress_bar = tqdm(
            total=0,
            desc=f"\033[32m{self.name}\033[0m",
            position=0,
            leave=True
        )

    def init_capture(self):
        self.image_files = sorted(glob.glob(os.path.join(self.image_dir, self.image_extension)), key=natural_sort_key)
        if not self.image_files:
            logging.error(f"[{self.name}] No images found in {self.image_dir} with extension {self.image_extension}.")
            raise ValueError(f"[{self.name}] No images found in {self.image_dir} with extension {self.image_extension}.")
        self.number_of_frames = len(self.image_files) - self.config.frame_range[0]
        self.idx = self.config.frame_range[0]

        self.dimensions()
        self.frame_rate()

        ts = os.path.getmtime(self.video_path)
        self.start_timestamp = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=timezone.utc)

        return None

    def read(self, _, capture):
        if capture:
            if self.idx not in self.config.frame_ranges:
                self.stop()

            self.progress_bar.n = self.idx
            self.progress_bar.refresh()

            path = self.image_files[self.idx]
            if self.timestamps:
                timestamp = self.timestamps[self.idx]
            else:
                frame_time = self.start_timestamp + timedelta(seconds=(self.idx  / self.fps))
                timestamp = frame_time.strftime("%Y%m%dT%H%M%S") + f"{frame_time.microsecond:06d}"

            frame = cv2.imread(path)

            frame_data = FrameData(self, timestamp, self.idx)
            self.idx += 1
            if frame is None:
                logging.error(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
                return None, frame_data

            if self.rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif self.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif self.rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if self.scale != 1:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            return frame, frame_data

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
        img = cv2.imread(self.image_files[0])
        if img is None:
            logging.error(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
            raise ValueError(f"[{self.name}] Unable to read the image from {self.image_files[0]}.")
        self.native_height, self.native_width = img.shape[:2]

        self.scale = min(self.desired_width / self.native_width, self.desired_height / self.native_height)

        self.target_width = int(self.native_width * self.scale)
        self.target_height = int(self.native_height * self.scale)

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.target_width, self.target_height = self.target_height, self.target_width

        self.x_offset = self.x_offset + ((self.desired_width - self.target_width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.target_height) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.target_width}x{self.target_height}")

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")
        self.progress_bar.set_description_str(f"\033[31m{self.name} (Ended)\033[0m")
        self.progress_bar.refresh()
        if self.video_writer:
            self.video_writer.release()
