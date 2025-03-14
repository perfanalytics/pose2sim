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

class BaseSource(abc.ABC):
    def __init__(self, config, data: dict):
        self.config = config
        self.data = data
        self.name = data.get("name")
        self.frame_rate = data.get("frame_rate")

        self.extrinsics_files = {}
        self.intrinsics_files = {}

        self.calib_intrinsics = data.get("calib_intrinsics")
        self.calib_extrinsics = data.get("calib_extrinsics")

        self.ret, self.ret_int, self.C, self.S, self.D, self.K, self.R, self.T = [], [], [], [], [], [], [], []

    def get_calib_files(self, folder, extension, calibration_name):
        if not os.path.isdir(folder):
            logging.warning(
                f"[{self.name} - {calibration_name}] Calibration skipped: The specified folder does not exist -> '{folder}'"
            )
            return None

        files = glob.glob(os.path.join(folder, f"*{extension}"))
        if not files:
            logging.warning(
                f"[{self.name} - {calibration_name}] Calibration skipped: No files with the extension '{extension}' found in folder '{folder}'."
            )
            return None

        return files

    def extract_frames(self, calib_type='intrinsic'):
        files = self.intrinsics_files if calib_type == 'intrinsic' else self.extrinsics_files

        video_path = files[0]

        directory = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = os.path.join(directory, base_name + '_*.png')

        new_files = glob.glob(pattern)
        new_files.sort()
        if new_files and not self.config.overwrite_extraction:
            logging.info("Frames have already been extracted and overwrite_extraction is False.")
            if calib_type == 'intrinsic':
                self.intrinsics_files = new_files
            else:
                self.extrinsics_files = new_files
            return

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Video capture could not be opened.")
        except Exception as e:
            logging.error(f"Failed to open video capture for {video_path}. Error: {e}")
            raise ValueError(f"The file {video_path} does not appear to be a valid video.")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            logging.error("FPS is 0, cannot extract frames.")
            raise ValueError("FPS is 0, cannot extract frames.")
        fps = round(fps)

        frame_nb = 0
        logging.info("Extracting frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Extract one frame every (fps * extract_every_N_sec) frames.
            if frame_nb % (fps * self.config.extract_every_N_sec) == 0:
                img_path = os.path.join(directory, base_name + '_' + str(frame_nb).zfill(5) + '.png')
                cv2.imwrite(img_path, frame)
            frame_nb += 1
        cap.release()

        new_files = glob.glob(pattern)
        new_files.sort()
        if calib_type == 'intrinsic':
            self.intrinsics_files = new_files
        else:
            self.extrinsics_files = new_files

    def calculate_calibration_residuals(self):
        if len(self.S) != 0 and len(self.D) != 0 and len(self.K) != 0 and len(self.R) != 0 and len(self.T) != 0:
            f_px = self.K[0, 0]
            Dm = np.linalg.norm(self.T)

            self.ret_int_px = np.around(np.array(self.ret_int), decimals=3)
            self.ret_px = np.around(np.array(self.ret), decimals=3)
            self.ret_mm = np.around(self.ret_px * Dm * 1000 / f_px, decimals=3)

            logging.info(f"[{self.name} - intrinsic] Intrinsic error: {self.ret_int_px} px.")
            logging.info(f"[{self.name} - extrinsic] Residual (RMS) calibration error: {self.ret_px} px, which corresponds to {self.ret_mm} mm.")

    @abc.abstractmethod
    def determine_frame_rate(self):
        pass

class WebcamSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.camera_index = data.get("path", 0)

    def determine_frame_rate(self):
        if self.frame_rate_config == "auto":
            return None
        else:
            return self.frame_rate_config

class VideoSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.video_path = data.get("path")

    def determine_frame_rate(self):
        if self.frame_rate != "auto":
            return None
        else:
            return self.frame_rate

class ImageSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension", "*.png")

    def determine_frame_rate(self):
        if self.frame_rate == "auto":
            return 60
        else:
            return self.frame_rate

