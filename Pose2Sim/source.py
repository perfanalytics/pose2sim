#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SOURCES                             ##
#########################################

'''


import abc
import os
import cv2
import glob

from Pose2Sim.config import SubConfig

class BaseSource(abc.ABC):
    def __init__(self, config: SubConfig, data: dict):
        self.config = config

        self.frame_rate = None

    @abc.abstractmethod
    def determine_frame_rate(self):
        pass


class WebcamSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.camera_index = data.get("camera_index", 0)
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config
        return 30


class VideoSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config

        video_files = glob.glob(os.path.join(self.video_dir, "*.mp4"))
        if video_files:
            cap = cv2.VideoCapture(video_files[0])
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                return round(fps) if fps > 0 else 30
        return 30


class ImageSource(BaseSource):
    def __init__(self, subconfig, data: dict):
        super().__init__(subconfig, data)
        self.image_extension = data.get("image_extension", "*.png")
        self.frame_rate = self.determine_frame_rate()

    def determine_frame_rate(self):
        if self.frame_rate_config != "auto":
            return self.frame_rate_config

        return 10
