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

        self.x_offset = 0
        self.y_offset = 0
        self.desired_width = 0
        self.desired_height = 0

        self.command_queue = multiprocessing.Queue()
        self.frame_queue = multiprocessing.Queue()

        self.scale = 1

        self.create_output_folders()

        self.capture_ready_event = multiprocessing.Event()

    def start_in_process(self):
        self.init_capture()

        self.process = multiprocessing.Process(target=self._capture_loop)
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
            now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
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

    def init_output_video_writer(self, fps):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            fps,
            (self.width, self.height)
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
    def extract_frames(self):
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
            self.raw_writer = cv2.VideoWriter(self.output_video_path, record_fourcc, self.fps, (self.width, self.height))

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

        if not ret or frame is None:
            if capture:
                logging.error(f"[{self.name}] Unable to read the frame.")
                return None, frame_data
            pass

        frame_data = FrameData(self, timestamp, self.idx)

        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.scale != 1:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

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
        #TODO: Si on enregistre la vidéo il ne faut pas au préalable la redimensionner
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

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.target_width, self.target_height = self.target_height, self.target_width

        self.x_offset = self.x_offset + ((self.desired_width - self.target_width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.target_height) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.target_width}x{self.target_height}")

    def extract_frames(self, calib_type):
        folder = self.calib_intrinsics if calib_type == 'intrinsic' else self.calib_extrinsics
        if folder is "live":
            #TODO: Extract frames from webcam
            logging.error(f"[{self.name}] Live calibration not already implemented.")
            raise ValueError(f"[{self.name}] Live calibration not already implemented.")
        else :
            files = self.intrinsics_files if calib_type == 'intrinsic' else self.extrinsics_files

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
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.scale = min(self.desired_width / self.width, self.desired_height / self.height)

        if self.scale != 1:
            target_width = int(self.width * self.scale)
            target_height = int(self.height * self.scale)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.width, self.height = self.height, self.width

        self.x_offset = self.x_offset + ((self.desired_width - self.width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.width) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.width}x{self.height}")


    def extract_frames(self, calib_type):
        folder = self.calib_intrinsics if calib_type == 'intrinsic' else self.calib_extrinsics
        if folder is "live":
            logging.error(f"[{self.name}] Live calibration not available from a video source.")
            raise ValueError(f"[{self.name}] Live calibration not available from a video source.")
        files = self.intrinsics_files if calib_type == 'intrinsic' else self.extrinsics_files

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

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")
        self.progress_bar.set_description_str(f"\033[31m{self.name} (Ended)\033[0m")
        self.progress_bar.refresh()


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
        self.height, self.width = img.shape[:2]

        if self.rotation % 90 != 0:
            logging.error(f"[{self.name}] Rotation must be multiple of 90.")
            raise ValueError(f"[{self.name}] Rotation must be multiple of 90.")
        if (self.rotation % 180) == 90:
            self.width, self.height = self.height, self.width

        self.scale = min(self.desired_width / self.width, self.desired_height / self.height)
        self.width = int(self.width * self.scale)
        self.height = int(self.height * self.scale)

        self.x_offset = self.x_offset + ((self.desired_width - self.width) // 2)
        self.y_offset = self.y_offset + ((self.desired_height - self.width) // 2)

        logging.info(f"[{self.name}] Dimensions of source after rotation ({self.rotation}°) and model scaling (including mosaic scaling if included): {self.width}x{self.height}")


    def extract_frames(self, calib_type):
        files = self.intrinsics_files if calib_type == 'intrinsic' else self.extrinsics_files
        if folder is "live":
            logging.error(f"[{self.name}] Live calibration not available from an image source.")
            raise ValueError(f"[{self.name}] Live calibration not available from an image source.")
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

    def stop(self):
        self.ended = True
        logging.info(f"[{self.name}] Stopped.")
        self.progress_bar.set_description_str(f"\033[31m{self.name} (Ended)\033[0m")
        self.progress_bar.refresh()
