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
        self.frame_rate_config = data.get("frame_rate")

        self.extrinsics_files = {}
        self.intrinsics_files = {}

        self.calib_intrinsics = data.get("calib_intrinsics")
        self.calib_extrinsics = data.get("calib_extrinsics")

        self.ret, self.ret_int, self.C, self.S, self.D, self.K, self.R, self.T = None, None, [], [], [], [], [], []

    def get_calib_files(self, folder, extension, calibration_name):
        if not os.path.isdir(os.path.join(self.config.calib_dir, folder)):
            logging.warning(
                f"[{self.name} - {calibration_name}] Calibration skipped: The specified folder does not exist -> '{folder}'"
            )
            return {}

        files = glob.glob(os.path.join(self.config.calib_dir, folder, f"*{extension}"))
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
        logging.info("Extracting frames...")
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

    @abc.abstractmethod
    def determine_frame_rate(self):
        pass

class WebcamSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.camera_index = data.get("path", 0)

    @property
    def frame_rate(self):
        if self.frame_rate_config == "auto":
            return None
        else:
            return self.frame_rate_config

class VideoSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.video_path = data.get("path")

    @property
    def frame_rate(self):
        if self.frame_rate_config != "auto":
            return None
        else:
            return self.frame_rate_config

class ImageSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension", "*.png")

    @property
    def frame_rate(self):
        if self.frame_rate_config == "auto":
            return 60
        else:
            return self.frame_rate_config

