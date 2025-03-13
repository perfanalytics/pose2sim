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

from Pose2Sim.config import SubConfig

class BaseSource(abc.ABC):
    def __init__(self, config: SubConfig, data: dict):
        self.config = config
        self.name = data.get("name")
        self.frame_rate = data.get("frame_rate", "auto")

        self.extrinsic_files = self.get_calib_files(config.calib_dir, "calib_extrinsic", config.extrinsic_extension)
        self.intrinsic_files = self.get_calib_files(config.calib_dir, "calib_intrinsic", config.intrinsic_extension)

        self.ret, self.C, self.S, self.D, self.K, self.R, self.T = [], [], [], [], [], [], []

    def get_calib_files(self, calib_dir, key, extension):
        # Construction du chemin du dossier
        folder = os.path.join(calib_dir, self.data.get(key))
        
        if not os.path.isdir(folder):
            logging.error(f"The folder '{folder}' does not exist.")
            raise ValueError(f"The folder '{folder}' does not exist.")
        
        files = glob.glob(os.path.join(folder, f"*{extension}"))
        if not files:
            logging.exception(f"The folder {folder} does not contain any {extension} files.")
            raise ValueError(f"The folder {folder} does not contain any {extension} files.")
        
        return files


    def extract_frames(self, calib_type='intrinsic'):
        files = self.intrinsic_files if calib_type == 'intrinsic' else self.extrinsic_files

        video_path = files[0]

        directory = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        pattern = os.path.join(directory, base_name + '_*.png')

        new_files = glob.glob(pattern)
        new_files.sort()
        if new_files and not self.config.overwrite_extraction:
            logging.info("Frames have already been extracted and overwrite_extraction is False.")
            if calib_type == 'intrinsic':
                self.intrinsic_files = new_files
            else:
                self.extrinsic_files = new_files
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
            logging.error("FPS is 0, cannot extract frames.")
            cap.release()
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
            self.intrinsic_files = new_files
        else:
            self.extrinsic_files = new_files

    @abc.abstractmethod
    def determine_frame_rate(self):
        pass

class WebcamSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.camera_index = data.get("path", 0)
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config
        return 30

class VideoSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.video_path = data.get("path")
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config

        if os.path.isfile(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                return round(fps) if fps > 0 else 30
        else:
            video_files = glob.glob(os.path.join(self.video_path, "*.mp4"))
            if video_files:
                cap = cv2.VideoCapture(video_files[0])
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    return round(fps) if fps > 0 else 30
        return 30

class ImageSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.image_dir = data.get("path")
        self.image_extension = data.get("extension", "*.png")
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config
        return 10

