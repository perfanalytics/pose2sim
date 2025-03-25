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
import cv2
import logging
import numpy as np
from datetime import datetime

class BaseSource(abc.ABC):
    def __init__(self, config, data: dict, pose_model):
        self.pose_model = pose_model
        self.config = config
        self.data = data
        self.name = data.get("name")
        self.rotation = data.get("rotation")

        self.extrinsics_files = {}
        self.intrinsics_files = {}

        self.calib_intrinsics = data.get("calib_intrinsics")
        self.calib_extrinsics = data.get("calib_extrinsics")
        self.rotation = data.get("rotation")

        self.ret, self.ret_int, self.C, self.S, self.D, self.K, self.R, self.T = None, None, [], [], [], [], [], []

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

        self.output_record_path = None
        if self.config.webcam_recording:
            self.output_video_path = os.path.join(self.config.pose_dir, f"{output_dir_name}_record.avi")

    @property
    @abc.abstractmethod
    def frame_rate(self):
        """Retourne le frame rate de la source."""
        pass

    @property
    @abc.abstractmethod
    def dimensions(self):
        """Retourne un tuple (largeur, hauteur) correspondant aux dimensions de l'image."""
        pass

class WebcamSource(BaseSource):
    def __init__(self, subconfig, data: dict, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.camera_index = data.get("path", 0)

    def _init_capture(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            logging.error(f"[{self.name}] Unable to open.")
            raise ValueError(f"[{self.name}] Unable to open.")

        det_width, det_height = self.pose_model.det_input_size

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"[{self.name}] Native resolution: {self.width}x{self.height}")

        if self.width >= self.height:
            scale = det_width / self.width
            target_width = det_width
            target_height = int(self.height * scale)
        else:
            scale = det_height / self.height
            target_height = det_height
            target_width = int(self.width * scale)

        if self.width != target_width and self.height != target_height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f"[{self.name}] Requested resolution: {target_width}x{target_height}, actual resolution: {self.width}x{self.height}")

            if self.width != target_width or self.height != target_height:
                logging.warning(f"[{self.name}] Did not accept the requested resolution of {self.width}x{self.height}")

        if self.config.gray_capture:
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        self.cap.set(cv2.CAP_PROP_FPS, 1000)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_rate(self):
        if self.config.frame_rate == "auto":
            self._init_capture()
            logging.info(f"[{self.name}] Frame rate: {self.fps}")
            return self.fps
        else:
            return self.config.frame_rate

    @property
    def dimensions(self):
        rotation = self.rotation if self.rotation is not None else 0
        if rotation % 90 != 0:
            logging.error(f"Rotation must be a multiple of 90. Got: {rotation}")
            raise ValueError(f"Rotation must be a multiple of 90. Got: {rotation}")
        
        if (rotation % 180) == 90:
            self.width, self.height = self.height, self.width
        
        logging.info(f"[{self.name}] Dimensions after rotation ({rotation}°): {self.width}x{self.height}")
        return (self.width, self.height)

class VideoSource(BaseSource):
    def __init__(self, subconfig, data: dict, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.video_path = data.get("path")

    @property
    def frame_rate(self):
        if self.config.frame_rate == "auto":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logging.error(f"Unable to get the frame rate from {self.name}.")
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f"[{self.name}] Frame rate: {fps}")
            cap.release()
            return round(fps) if fps > 0 else None
        else:
            return self.config.frame_rate

    @property
    def dimensions(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"Unable to get the dimensions from {self.name}.")
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        rotation = self.rotation if self.rotation is not None else 0
        if rotation % 90 != 0:
            logging.error(f"Rotation must be a multiple of 90. Got: {rotation}")
            raise ValueError(f"Rotation must be a multiple of 90. Got: {rotation}")
        if (rotation % 180) == 90:
            width, height = height, width

        logging.info(f"[{self.name}] Dimensions after rotation ({rotation}°): {width}x{height}")
        return (width, height)

class ImageSource(BaseSource):
    def __init__(self, subconfig, data: dict, pose_model):
        super().__init__(subconfig, data, pose_model)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension", "*.png")

    @property
    def frame_rate(self):
        if self.config.frame_rate == "auto":
            logging.error(f"Frame rate is set to 'auto' but not possible on images.")
            raise ValueError(f"Frame rate is set to 'auto' but not possible on images.")
        else:
            return self.config.frame_rate

    @property
    def dimensions(self):
        image_files = glob.glob(os.path.join(self.image_dir, self.image_extension))
        if not image_files:
            logging.error(f"Unable to get images from {self.image_dir}.")
            return None

        if image_files[0] is None:
            logging.error(f"Unable to read the image from {image_files[0]}.")
            return None
        height, width = image_files[0].shape[:2]

        rotation = self.rotation if self.rotation is not None else 0
        if rotation % 90 != 0:
            logging.error(f"Rotation must be a multiple of 90. Got: {rotation}")
            raise ValueError(f"Rotation must be a multiple of 90. Got: {rotation}")
        if (rotation % 180) == 90:
            width, height = height, width

        logging.info(f"[{self.name}] Dimensions after rotation ({rotation}°): {width}x{height}")
        return (width, height)
