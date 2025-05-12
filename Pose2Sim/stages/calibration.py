#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## CAMERAS CALIBRATION                                                   ##
###########################################################################

Use this module to calibrate your cameras and save results to a .toml file.

It either converts a Qualisys calibration .qca.txt file,
Or calibrates cameras from checkerboard images.

Checkerboard calibration is based on 
https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

INPUTS: 
- a calibration file in the 'calibration' folder (.qca.txt extension)
- OR folders 'calibration/intrinsics' (populated with video or about 30 images) and 'calibration/extrinsics' (populated with video or one image)
- a Config.toml file in the 'User' folder

OUTPUTS: 
- a calibration file in the 'calibration' folder (.toml extension)
'''

# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS


## INIT
import os
import toml
import logging
import pickle
import glob
import numpy as np
import pandas as pd
import cv2
from lxml import etree
import warnings
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from PIL import Image
from mpl_interactions import zoom_factory, panhandler
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Optional

from Pose2Sim.common import world_to_camera_persp, rotate_cam, quat2mat, euclidean_distance, zup2yup
from Pose2Sim.stages.base import BaseStage
from Pose2Sim.source import BaseSource

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


@dataclass
class CalibrationSettings:
    calib_dir: Path
    calib_output_path: Path
    calib_type: str
    overwrite_intrinsics: bool
    overwrite_extrinsics: bool
    calculate_extrinsics: bool

    convert_path: Optional[Path] = None
    binning_factor_qualisys: Optional[int] = None

    overwrite_extraction: Optional[bool] = None
    extract_every_N_sec: Optional[float] = None

    show_detection_intrinsics: Optional[bool] = None
    intrinsics_corners_nb: Optional[int] = None
    intrinsics_square_size: Optional[float] = None
    intrinsics_extension: Optional[str] = None

    extrinsics_method: Optional[str] = None
    extrinsics_extension: Optional[str] = None
    show_reprojection_error: Optional[bool] = None
    extrinsics_corners_nb: Optional[int] = None
    extrinsics_square_size: Optional[float] = None
    object_coords_3d: Optional[list[list[float]]] = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], session_dir: Path) -> "CalibrationSettings":
        cal_cfg   = cfg.get("calibration")
        c_type    = cal_cfg.get("calibration_type")
        calib_dir  = Path(session_dir) / "calibration"
        out_path  = Path(calib_dir) / "Calib.toml"

        kwargs: dict[str, Any] = dict(
            calib_dir             = calib_dir,
            calib_output_path     = out_path,
            calib_type            = c_type,
            overwrite_intrinsics  = cal_cfg.get("overwrite_intrinsics"),
            overwrite_extrinsics  = cal_cfg.get("overwrite_extrinsics"),
            calculate_extrinsics  = cal_cfg.get("calculate_extrinsics"),
        )

        if c_type == "convert":
            conv_cfg  = cal_cfg.get("convert")
            conv_path = Path(conv_cfg.get("convert_from")).expanduser()
            if not conv_path.is_absolute():
                conv_path = Path(calib_dir) / conv_path
            if not conv_path.exists():
                raise FileNotFoundError(f"Conversion file '{conv_path}' not found.")

            kwargs.update(
                convert_path            = conv_path,
                binning_factor_qualisys = conv_cfg.get("qualisys").get("binning_factor"),
            )

        elif c_type == "calculate":
            calc_cfg = cal_cfg.get("calculate", {})
            kwargs.update(
                overwrite_extraction     = calc_cfg.get("overwrite_extraction"),
                extract_every_N_sec      = calc_cfg.get("extract_every_N_sec"),
            )

            intr = calc_cfg.get("intrinsics", {})
            kwargs.update(
                show_detection_intrinsics = intr.get("show_detection_intrinsics"),
                intrinsics_corners_nb     = intr.get("intrinsics_corners_nb"),
                intrinsics_square_size    = intr.get("intrinsics_square_size") / 1000.0,
                intrinsics_extension      = intr.get("intrinsics_extension"),
            )

            extr = calc_cfg.get("extrinsics", {})
            method = extr.get("extrinsics_method", "board")
            kwargs.update(
                extrinsics_method        = method,
                extrinsics_extension     = extr.get("extrinsics_extension"),
                show_reprojection_error  = extr.get("show_reprojection_error"),
            )

            if method == "board":
                board = extr.get("board", {})
                kwargs.update(
                    extrinsics_corners_nb  = board.get("extrinsics_corners_nb"),
                    extrinsics_square_size = board.get("extrinsics_square_size") / 1000.0,
                )
            elif method == "scene":
                scene = extr.get("scene", {})
                kwargs.update(object_coords_3d = scene.get("object_coords_3d"))

        return cls(**kwargs)


class Calibration(ABC):
    """
    Abstract base class for calibration conversion.
    
    This class defines the interface for calibrating both intrinsic and extrinsic parameters.
    """
    @abstractmethod
    def calibrate_intrinsics(self, source):
        """
        Abstract method to perform calibration.
        
        file_paths can be a string (for single-file calibrations) or a list/tuple for dual-file calibrations.
        """
        pass

    @abstractmethod
    def calibrate_extrinsics(self, source):
        """
        Abstract method to perform calibration.
        
        file_paths can be a string (for single-file calibrations) or a list/tuple for dual-file calibrations.
        """
        pass


class QcaCalibration(Calibration):
    """
    Converts a Qualisys .qca.txt calibration file.
    
    This class extracts intrinsic parameters (image size, intrinsic matrix, distortion, and intrinsic residual error)
    from the calibration file and, if extrinsics are provided, converts the extrinsic rotation and translation 
    using a world-to-camera conversion. The rotation is then converted to an OpenCV rotation vector.
    """
    def __init__(self, path, binning_factor):
        self.path = path
        root = etree.parse(self.path).getroot()
        self.binning_factor = binning_factor
        self.camera_tags = root.findall('cameras/camera')
        self.fov_tags = root.findall('cameras/camera/fov_video')
        self.intrinsic_tags = root.findall('cameras/camera/intrinsic')
        self.transform_tags = root.findall('cameras/camera/transform')

    def calibrate_intrinsics(self, source):
        for i, tag in enumerate(self.camera_tags):
            name = tag.attrib.get('serial')
            if name == source.name:
                source.ret_int = float(tag.attrib.get('avg-residual'))
                video_res = tag.attrib.get('video_resolution')
                res_value = int(video_res[:-1]) if video_res not in (None, "N/A") else 1080
                fov_tag = self.fov_tags[i]
                w = (float(fov_tag.attrib.get('right')) - float(fov_tag.attrib.get('left')) + 1) / self.binning_factor / (1080 / res_value)
                h = (float(fov_tag.attrib.get('bottom')) - float(fov_tag.attrib.get('top')) + 1) / self.binning_factor / (1080 / res_value)
                source.S = [w, h]
                intrinsic_tag = self.intrinsic_tags[i]
                fu = float(intrinsic_tag.get('focalLengthU')) / 64 / self.binning_factor / (1080 / res_value)
                fv = float(intrinsic_tag.get('focalLengthV')) / 64 / self.binning_factor / (1080 / res_value)
                left_val = float(fov_tag.attrib.get('left'))
                top_val = float(fov_tag.attrib.get('top'))
                cu = (float(intrinsic_tag.get('centerPointU')) / 64 / self.binning_factor - left_val) / (1080 / res_value)
                cv_val = (float(intrinsic_tag.get('centerPointV')) / 64 / self.binning_factor - top_val) / (1080 / res_value)
                source.K = np.array([fu, 0., cu, 0., fv, cv_val, 0., 0., 1.]).reshape(3, 3)
                k1 = float(intrinsic_tag.get('radialDistortion1')) / 64 / self.binning_factor
                k2 = float(intrinsic_tag.get('radialDistortion2')) / 64 / self.binning_factor
                p1 = float(intrinsic_tag.get('tangentalDistortion1')) / 64 / self.binning_factor
                p2 = float(intrinsic_tag.get('tangentalDistortion2')) / 64 / self.binning_factor
                source.D = [k1, k2, p1, p2]
                source.R = []
                source.T = []
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.info(f"[{source.name}] No intrinsic calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        for i, tag in enumerate(self.camera_tags):
            name = tag.attrib.get('serial')
            if name == source.name:
                # Warn if intrinsic calibration has not been performed.
                if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
                    logging.warning(f"You should not calibrate the extrinsics of {source.name} without first calibrating the intrinsics.")

                source.ret = None
                transform_tag = self.transform_tags[i]
                r11 = float(transform_tag.get('r11'))
                r12 = float(transform_tag.get('r12'))
                r13 = float(transform_tag.get('r13'))
                r21 = float(transform_tag.get('r21'))
                r22 = float(transform_tag.get('r22'))
                r23 = float(transform_tag.get('r23'))
                r31 = float(transform_tag.get('r31'))
                r32 = float(transform_tag.get('r32'))
                r33 = float(transform_tag.get('r33'))
                # Build and transpose the rotation matrix
                R_value = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3, 3).T
                tx = float(transform_tag.get('x')) / 1000
                ty = float(transform_tag.get('y')) / 1000
                tz = float(transform_tag.get('z')) / 1000
                T_value = [tx, ty, tz]
                # Convert from world to camera coordinates
                r_transf, t_transf = world_to_camera_persp(R_value, T_value)
                # Apply additional camera rotation as needed
                r_transf, t_transf = rotate_cam(r_transf, t_transf, ang_x=np.pi, ang_y=0, ang_z=0)
                # Convert rotation matrix to an OpenCV rotation vector
                source.R = cv2.Rodrigues(r_transf)[0].flatten()
                source.T = t_transf
        if len(source.R) == 0 and len(source.T) == 0:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class OptitrackCalibration(Calibration):
    """
    Placeholder for Optitrack calibration conversion.
    
    Since the Optitrack calibration values are provided externally (see Readme.md),
    this class raises an error to instruct the user to retrieve those values manually.
    """
    def calibrate_intrinsics(self, source):
        logging.warning('Refer to Readme.md for retrieving Optitrack calibration values.')
        raise NameError("Refer to Readme.md for retrieving Optitrack calibration values.")

    def calibrate_extrinsics(self, source):
        logging.warning('Refer to Readme.md for retrieving Optitrack calibration values.')
        raise NameError("Refer to Readme.md for retrieving Optitrack calibration values.")


class ViconCalibration(Calibration):
    """
    Converts a Vicon .xcp calibration file.
    
    This class extracts intrinsic parameters (including the residual error)
    and computes extrinsic parameters by converting the quaternion orientation to a rotation matrix,
    switching from world to camera coordinates, and converting the rotation to OpenCV format.
    """

    def __init__(self, path):
        self.path = path
        self.cameras = etree.parse(self.path).getroot().findall('Camera')

    def calibrate_intrinsics(self, source):
        for cam_elem in self.cameras:
            name = cam_elem.attrib.get('DEVICEID')
            if name == source.name:
                keyframe = cam_elem.findall('KeyFrames/KeyFrame')[0]
                source.ret_int = float(keyframe.attrib.get('WORLD_ERROR'))
                source.S = [float(t) for t in cam_elem.attrib.get('SENSOR_SIZE').split()]
                fu = float(keyframe.attrib.get('FOCAL_LENGTH'))
                pixel_aspect = float(cam_elem.attrib.get('PIXEL_ASPECT_RATIO'))
                fv = fu / pixel_aspect
                cam_center = keyframe.attrib.get('PRINCIPAL_POINT').split()
                cu, cv_val = [float(c) for c in cam_center]
                source.K = np.array([fu, 0., cu, 0., fv, cv_val, 0., 0., 1.]).reshape(3, 3)
                try:
                    dist = keyframe.attrib.get('VICON_RADIAL2').split()[3:5]
                    dist = [float(d) for d in dist]
                except Exception:
                    dist = [float(x) for x in keyframe.attrib.get('VICON_RADIAL').split()]
                source.D = dist + [0.0, 0.0]
                source.R = []
                source.T = []
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.info(f"[{source.name}] No intrinsic calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        for cam_elem in self.cameras:
            name = cam_elem.attrib.get('DEVICEID')
            if name == source.name:
                keyframe = cam_elem.findall('KeyFrames/KeyFrame')[0]
                if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
                    logging.warning(f"You should not calibrate the extrinsics of {source.name} without first calibrating the intrinsics.")

                source.ret = None
                rot = keyframe.attrib.get('ORIENTATION').split()
                R_quat = [float(r) for r in rot]
                r_orig = quat2mat(R_quat, scalar_idx=3)
                t_orig = [float(t) for t in keyframe.attrib.get('POSITION').split()]
                r_trans, t_trans = world_to_camera_persp(r_orig, t_orig)
                source.R = np.array(cv2.Rodrigues(r_trans)[0]).flatten()
                source.T = np.array(t_trans)
        if len(source.R) == 0 and len(source.T) == 0:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class EasyMocapCalibration(Calibration):
    """
    Reads EasyMocap .yml calibration files.
    
    This class processes separate intrinsic and extrinsic YAML files. For intrinsics,
    it assigns the intrinsic matrix, image size, and distortion parameters. For extrinsics,
    it reads the extrinsic rotation and translation directly from the YAML file.
    """

    def __init__(self, path):
        self.path = path

    def calibrate_intrinsics(self, source):
        _, intrinsic_path = self.path
        intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
        cam_number = int(intrinsic_yml.getNode('names').size())
        for i in range(cam_number):
            name = intrinsic_yml.getNode('names').at(i).string()
            if name == source.name:
                K_mat = intrinsic_yml.getNode(f'K_{name}').mat()
                source.ret_int = None
                source.S = [K_mat[0, 2] * 2, K_mat[1, 2] * 2]
                source.K = K_mat
                source.D = intrinsic_yml.getNode(f'dist_{name}').mat().flatten()[:-1]
                source.R = []
                source.T = []
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.info(f"[{source.name}] No intrinsic calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        extrinsic_path, _ = self.path
        extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)
        cam_number = int(extrinsic_yml.getNode('names').size())
        for i in range(cam_number):
            name = extrinsic_yml.getNode('names').at(i).string()
            if name == source.name:
                if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
                    logging.warning(f"You should not calibrate the extrinsics of {source.name} without first calibrating the intrinsics.")

                source.ret = None
                source.R = extrinsic_yml.getNode(f'R_{name}').mat().flatten()
                source.T = extrinsic_yml.getNode(f'T_{name}').mat().flatten()
        if len(source.R) == 0 and len(source.T) == 0:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class BiocvCalibration(Calibration):
    """
    Converts bioCV calibration files.
    
    This class reads calibration data from text files. For intrinsics, it parses the image size, intrinsic matrix,
    and distortion parameters. For extrinsics, it converts the rotation matrix (to an OpenCV rotation vector)
    and adjusts the translation (dividing by 1000 for unit conversion).
    """

    def __init__(self, path):
        self.path = path

    def calibrate_intrinsics(self, source):
        for i, f_path in enumerate(self.path):
            with open(f_path) as f:
                calib_data = f.read().split('\n')
            name = f'cam_{str(i).zfill(2)}'
            if name == source.name:
                source.ret_int = None
                source.S = [float(calib_data[0]), float(calib_data[1])]
                source.K = np.array([
                    list(map(float, calib_data[2].split())),
                    list(map(float, calib_data[3].split())),
                    list(map(float, calib_data[4].split()))
                ])
                source.D = [float(d) for d in calib_data[-2].split()[:4]]
                source.R = []
                source.T = []
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.info(f"[{source.name}] No intrinsic calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        for i, f_path in enumerate(self.path):
            with open(f_path) as f:
                calib_data = f.read().split('\n')
            name = f'cam_{str(i).zfill(2)}'
            if name == source.name:
                if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
                    logging.warning(f"You should not calibrate the extrinsics of {source.name} without first calibrating the intrinsics.")

                source.ret = None
                RT = np.array([list(map(float, line.split())) for line in calib_data[6:9]])
                source.R = cv2.Rodrigues(RT[:, :3])[0].squeeze()
                source.T = RT[:, 3] / 1000
        if len(source.R) == 0 and len(source.T) == 0:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class LoadCalibration(Calibration):
    def __init__(self, path):
        self.path = path
        calib_file = glob.glob(self.path)
        if len(calib_file) == 0:
            logging.info(f"No calibration file found in {self.path}.")
        else:
            self.data = toml.load(calib_file)

    def calibrate_intrinsics(self, source):
        if source.name in self.data:
            source_data = self.data[source.name]
            source.ret_int = 0.0
            source.S = source_data['size']
            source.K = np.array(source_data['matrix'])
            source.D = source_data['distortions']
        else :
            logging.info(f"[{source.name}] No calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        if source.name in self.data:
            source_data = self.data[source.name]
            source.ret = 0.0
            source.R = [0.0, 0.0, 0.0]
            source.T = [0.0, 0.0, 0.0]
        else:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class OpencapCalibration(Calibration):
    """
    Converts OpenCap calibration files.
    
    For intrinsics, this class reads the intrinsic matrix, image size, and distortion parameters from a pickle file.
    For extrinsics, it converts the rotation using a world-to-camera conversion and additional rotations
    so that the final rotation (and translation) are in the OpenCV camera frame.
    """

    def __init__(self, path):
        self.path = path

    def calibrate_intrinsics(self, source):
        for i, f_path in enumerate(self.path):
            with open(f_path, 'rb') as f_pickle:
                calib_data = pickle.load(f_pickle)
            name = f'cam_{str(i).zfill(2)}'
            if name == source.name:
                source.ret_int = None
                source.S = list(map(float, calib_data['imageSize'].squeeze()[:-1]))
                source.D = list(map(float, calib_data['distortion'][0][:-1]))
                source.K = calib_data['intrinsicMat']
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.info(f"[{source.name}] No intrinsic calibration found in {self.path}.")

    def calibrate_extrinsics(self, source):
        for i, f_path in enumerate(self.path):
            with open(f_path, 'rb') as f_pickle:
                calib_data = pickle.load(f_pickle)
            name = f'cam_{str(i).zfill(2)}'
            if name == source.name:
                if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
                    logging.warning(f"You should not calibrate the extrinsics of {source.name} without first calibrating the intrinsics.")

                source.ret = None
                R_cam = calib_data['rotation']
                T_cam = calib_data['translation'].squeeze()
                # Convert from world frame to camera frame
                R_w, T_w = world_to_camera_persp(R_cam, T_cam)
                # Apply rotation: -pi/2 around x and pi around z to adjust frame orientation
                R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi)
                # Convert back to camera frame
                R_c_90, T_c_90 = world_to_camera_persp(R_w_90, T_w_90)
                source.R = cv2.Rodrigues(R_c_90)[0].squeeze()
                source.T = T_cam / 1000
        if len(source.R) == 0 and len(source.T) == 0:
            logging.info(f"[{source.name}] No extrinsic calibration found in {self.path}.")


class PointCalibration(Calibration):
    """
    Calibrates intrinsic and extrinsic parameters using images or videos.
    
    For intrinsic calibration, computes the intrinsic matrix,
    distortion parameters, and the intrinsic residual error. For extrinsic calibration,
    compute the camera pose and then calculates the reprojection error.
    """

    def __init__(self, settings):
        self.settings = settings

    def calibrate_intrinsics(self, source):
        files = source.get_calib_files("intrinsics", self.settings.calib_dir, self.settings.intrinsics_extension)
        logging.info(f'Intrinsic calibration for {source.name}:')
        objp = np.zeros((self.settings.intrinsics_corners_nb[0] * self.settings.intrinsics_corners_nb[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.settings.intrinsics_corners_nb[0], 0:self.settings.intrinsics_corners_nb[1]].T.reshape(-1, 2)
        objp[:, :2] *= self.settings.intrinsics_square_size
        objpoints = []  # 3D points in world space
        imgpoints = []  # 2D points in image plane
        for file in files:
            imgp_confirmed, objp_confirmed = findCorners(file, self.settings.intrinsics_corners_nb, objp=objp, show=self.settings.show_detection_intrinsics)
            if isinstance(imgp_confirmed, np.ndarray):
                imgpoints.append(imgp_confirmed)
                objpoints.append(objp_confirmed if self.settings.show_detection_intrinsics else objp)
        if len(imgpoints) < 10:
            logging.info(f"Only {len(imgpoints)} images with detected corners for {source.name}. Intrinsic calibration may be inaccurate.")
        img = cv2.imread(str(file))
        objpoints = np.array(objpoints)
        ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[1::-1], None, None,
            flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_USE_LU)
        )
        h, w = img.shape[:2]
        source.ret_int = ret_cam
        source.S = [w, h]
        source.K = dist[0]
        source.D = mtx

    def calibrate_extrinsics(self, source):
        files = source.get_calib_files('extrinsic', self.settings.calib_dir, self.settings.extrinsics_extension)
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0:
            logging.warning(f"Cannot calibrate extrinsics for {source.name} without first calibrating intrinsics.")
            raise ValueError(f"Cannot calibrate extrinsics for {source.name} without first calibrating intrinsics.")
        logging.info(f'Extrinsic calibration for {source.name}:')
        img = cv2.imread(source.extrinsic_files[0])
        object_coords_3d = self.settings.object_coords_3d
        if self.settings.extrinsics_method == 'board':
            object_coords_3d = np.zeros((self.extrinsics_corners_nb[0] * self.extrinsics_corners_nb[1], 3), np.float32)
            object_coords_3d[:, :2] = np.mgrid[0:self.extrinsics_corners_nb[0], 0:self.extrinsics_corners_nb[1]].T.reshape(-1, 2)
            object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * self.settings.extrinsics_square_size
            imgp, objp = findCorners(files[0], self.settings.extrinsics_corners_nb, objp=object_coords_3d, show=self.settings.show_reprojection_error)
            if len(imgp) == 0:
                logging.exception('No corners detected. Use "scene" method or verify detection settings.')
                raise ValueError('No corners detected.')
        elif self.settings.extrinsics_method == 'scene':
            imgp, objp = imgp_objp_visualizer_clicker(img, imgp=[], objp=object_coords_3d, img_path=files[0])
            if len(imgp) == 0:
                logging.exception('No points clicked (or fewer than required).')
                raise ValueError('No points clicked (or fewer than required).')
            if len(objp) < 10:
                logging.info(f"Only {len(objp)} reference points for {source.name}. Extrinsic calibration may be imprecise.")
        trc_write(object_coords_3d, os.path.join(self.settings.calib_dir, f'Object_points.trc'))
        mtx, dist = np.array(source.K), np.array(source.D)
        _, r, t = cv2.solvePnP(objp * 1000, imgp, mtx, dist)
        source.R = r.flatten()  # Extrinsic rotation vector in OpenCV format
        source.T = t.flatten() / 1000  # Extrinsic translation vector in meters (converted)
        # Projection of object points to image plane
        # # Former way, distortions used to be ignored
        # Kh_cam = np.block([mtx, np.zeros(3).reshape(3,1)])
        # r_mat, _ = cv2.Rodrigues(r)
        # H_cam = np.block([[r_mat,t.reshape(3,1)], [np.zeros(3), 1 ]])
        # P_cam = Kh_cam @ H_cam
        # proj_obj = [ ( P_cam[0] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)),  P_cam[1] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)) ) for o in objp]
        proj_obj = np.squeeze(cv2.projectPoints(objp, r, t, mtx, dist)[0])
        proj_obj_all = np.squeeze(cv2.projectPoints(object_coords_3d,r,t,mtx,dist)[0])

        if self.settings.show_reprojection_error:
            # Reopen image, otherwise 2 sets of text are overlaid
            img = cv2.imread(source.extrinsic_files[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for o in proj_obj_all:
                cv2.circle(img, (int(o[0]), int(o[1])), 8, (0,0,255), -1) 
            for i in imgp:
                cv2.drawMarker(img, (int(i[0][0]), int(i[0][1])), (0,255,0), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
            cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA) 
            cv2.drawMarker(img, (20,40), (0,255,0), cv2.MARKER_CROSS, 15, 2)
            cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
            cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
            cv2.circle(img, (20,60), 8, (0,0,255), -1)    
            cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
            cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
            im_pil = Image.fromarray(img)
            im_pil.show(title = os.path.basename(source.extrinsic_files[0]))

            # Calculate reprojection error

        imgp_to_objreproj_dist = [euclidean_distance(proj_obj[n], imgp[n]) for n in range(len(proj_obj))]
        rms_px = np.sqrt(np.sum([d**2 for d in imgp_to_objreproj_dist]))
        source.ret = rms_px


def findCorners(img_path, corner_nb, objp=[], show=True):
    '''
    Find corners in the photo of a checkerboard.
    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.
    
    Make sure that: 
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners
    
    INPUTS:
    - img_path: path to image (or video)
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [4,7]
    - optional: show: choose whether to show corner detections
    - optional: objp: array [3d corner coordinates]

    OUTPUTS:
    - imgp_confirmed: array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)
    # If corners are found, refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')

        if show:
            # Draw corners
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
            # Add corner index 
            for i, corner in enumerate(imgp):
                if i in [0, corner_nb[0]-1, corner_nb[0]*(corner_nb[1]-1), corner_nb[0]*corner_nb[1] -1]:
                    x, y = corner.ravel()
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 7) 
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2) 
            
            return imgp_objp_visualizer_clicker(img, imgp=imgp, objp=objp, img_path=img_path)
        else:
            return imgp
            

    # If corners are not found, dismiss or click points by hand
    else:
        if show:
            # Visualizer and key press event handler
            logging.info(f'{os.path.basename(img_path)}: Corners not found: please label them by hand.')
            return imgp_objp_visualizer_clicker(img, imgp=[], objp=objp, img_path=img_path)
        else:
            logging.info(f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in your settings.')
            return []


## TODO: NEED A REWORK !
def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=''):
    '''
    Shows image img. 
    If imgp is given, displays them in green
    If objp is given, can be displayed in a 3D plot if 'C' is pressed.
    If img_path is given, just uses it to name the window

    If 'Y' is pressed, closes all and returns confirmed imgp and (if given) objp
    If 'N' is pressed, closes all and returns nothing
    If 'C' is pressed, allows clicking imgp by hand. If objp is given:
        Displays them in 3D as a helper. 
        Left click to add a point, right click to remove the last point.
        Press 'H' to indicate that one of the objp is not visible on image
        Closes all and returns imgp and objp if all points have been clicked
    Allows for zooming and panning with middle click
    
    INPUTS:
    - img: image opened with openCV
    - optional: imgp: detected image points, to be accepted or not. Array of [[2d corner coordinates]]
    - optional: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''
    global old_image_path
    old_image_path = img_path
                                 
    def on_key(event):
        '''
        Handles key press events:
        'Y' to return imgp, 'N' to dismiss image, 'C' to click points by hand.
        Left click to add a point, 'H' to indicate it is not visible, right click to remove the last point.
        '''

        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count
        
        if event.key == 'y':
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if 'scat' not in globals() or 'imgp_confirmed' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array([imgp.astype('float32') for imgp in imgp_confirmed])
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == 'c':
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            img_for_pointing = cv2.imread(old_image_path)
            if img_for_pointing is None:
                cap = cv2.VideoCapture(old_image_path)
                ret, img_for_pointing = cap.read()
            img_for_pointing = cv2.cvtColor(img_for_pointing, cv2.COLOR_BGR2RGB)
            ax.imshow(img_for_pointing)
            # To update the image
            plt.draw()

            if 'objp_confirmed' in globals():
                del objp_confirmed
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([],[],s=100,marker='+',color='g')
            plt.connect('button_press_event', on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title('Object points to be clicked')
                ax_3d = fig_3d.add_subplot(projection='3d')
                plt.rc('xtick', labelsize=5)
                plt.rc('ytick', labelsize=5)
                for i, (xs,ys,zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs,ys,zs, marker='.', color='k')
                    ax_3d.text(xs,ys,zs,  f'{str(i+1)}', size=10, zorder=1, color='k') 
                set_axes_equal(ax_3d)
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=90, azim=-90)
                else:
                    ax_3d.view_init(vertical_axis='z')
                fig_3d.show()

        if event.key == 'h':
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0  and 'ax_3d' in globals():
                count = [0 if 'count' not in globals() else count+1][0]
                objp_confirmed_notok = objp[count]
                if 'events' not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    # Plot 3D
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # Plot 3D
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()

                    # Ask for confirmation
                    root = tk.Tk()
                    root.withdraw()
                    response = messagebox.askyesno("Confirmation", "Confirm and go to next image")
                    # Confirmed
                    if response == True:
                        plt.close('all')
                        # if all objp have been clicked or indicated as not visible, close all
                        objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])[:-1]
                        imgp_confirmed = np.array(np.expand_dims(scat.get_offsets(), axis=1), np.float32) 
                        for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                            if var_to_delete in globals():
                                del globals()[var_to_delete]
                    # Not confirmed
                    else:
                        root.destroy()
                        # remove from plot 
                        ax_3d.collections[-1].remove()
                        fig_3d.canvas.draw()
                        count -= 1
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
            else:
                pass


    def on_click(event):
        '''
        Detect click position on image
        If right click, last point is removed
        '''
        
        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, events, count, xydata
        
        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1: 
            # To remember the event to cancel after right click
            if 'events' in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata,[[event.xdata,event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)    
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if 'count' not in globals() else count+1][0]
                if count==0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:

                    # plot 3D
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                    # Ask for confirmation
                    root = tk.Tk()
                    root.withdraw()
                    response = messagebox.askyesno("Confirmation", "Confirm and go to next image")
                    # Confirmed
                    if response == True:
                        plt.close('all')
                        # retrieve objp_confirmed
                        objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])
                        imgp_confirmed = np.array(imgp_confirmed, np.float32)
                        # delete all
                        for var_to_delete in ['events', 'count', 'scat', 'scat_3d', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                            if var_to_delete in globals():
                                del globals()[var_to_delete]
                    # Not confirmed
                    else:
                        root.destroy()
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        # remove from plot 
                        ax_3d.collections[-1].remove()
                        fig_3d.canvas.draw()
                        count -= 1
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                
        # Right click: 
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3: # right click
            if 'events' in globals():
                # If last event was left click: 
                if 'button' in dir(events[-1]):
                    if events[-1].button == 1: 
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        if len(objp) != 0:
                            if count >= 0: 
                                count -= 1
                            # Remove last point from objp_confirmed
                            objp_confirmed = objp_confirmed[:-1]
                            # remove from plot 
                            if len(ax_3d.collections) > len(objp):
                                ax_3d.collections[-1].remove()
                                fig_3d.canvas.draw()
                            
                # If last event was 'h' key
                elif events[-1].key == 'h':
                    if len(objp) != 0:
                        if count >= 1: count -= 1
                        # Remove last point from objp_confirmed_notok
                        objp_confirmed_notok = objp_confirmed_notok[:-1]
                        # remove from plot  
                        if len(ax_3d.collections) > len(objp):
                            ax_3d.collections[-1].remove()
                            fig_3d.canvas.draw()                
    

    def set_axes_equal(ax):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    
    # Put image in a matplotlib figure for more controls
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x),int(y)), (128,128,128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    
    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['toolbar'] = 'toolmanager'

    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if 'imgp_confirmed' in globals() and 'objp_confirmed' in globals():
        return imgp_confirmed, objp_confirmed
    elif 'imgp_confirmed' in globals() and not 'objp_confirmed' in globals():
        return imgp_confirmed, None
    else:
        return None, None


class CalibrationStage(BaseStage):
    name = "calibration"

    def __init__(self, settings: CalibrationSettings, sources: list[BaseSource]):
        self.settings    = settings
        self.sources     = sources

    def run(self, data_in):
        if self.settings.calib_type == "convert":            
            filename = os.path.basename(self.settings.convert_path).lower()

            if filename.endswith(".qca.txt"):
                calibration = QcaCalibration(self, self.settings.convert_path, self.settings.binning_factor_qualisys)
            elif filename.endswith(".xcp"):
                calibration = ViconCalibration(self, self.settings.convert_path)
            elif filename.endswith(".pickle"):
                calibration = OpencapCalibration(self, self.settings.convert_path)
            elif filename.endswith(".yml"):
                calibration = EasyMocapCalibration(self, self.settings.convert_path)
            elif filename.endswith(".calib"):
                calibration = BiocvCalibration(self, self.settings.convert_path)
            elif filename.endswith(".csv"):
                calibration = OptitrackCalibration(self, self.settings.convert_path)
            elif any(filename.endswith(ext) for ext in [".anipose", ".freemocap", ".caliscope"]):
                #TODO
                logging.info("\n--> No conversion required for Caliscope, AniPose, or FreeMocap. Calibration will be ignored.\n")
                return None
            else:
                raise NameError(f"File {filename} not supported for conversion.")

        elif self.settings.calib_type == "calculate":
            calibration = PointCalibration(self, self.settings)

        elif self.settings.calib_type == "load":
            calibration = LoadCalibration(self, self.settings.calib_output_path)

        else:
            logging.info("Invalid calibration_type in your settings.")
            return ValueError("Invalid calibration_type in your settings.")

        for source in self.sources:
            if not self.settings.overwrite_intrinsics and len(source.S) != 0 and len(source.D) != 0 and len(source.K) != 0:
                logging.info(
                    f"[{source.name} - intrinsic] Existing intrinsic calibration."
                )
                logging.info(
                    'To recalculate, set "overwrite_intrinsics" to true in your settings.'
                )
            else:
                calibration.calibrate_intrinsics(source)

            if not self.settings.overwrite_extrinsics and len(source.R) != 0 and len(source.T) != 0:
                logging.info(
                    f"[{source.name} - entrinsic] Existing extrinsic calibration."
                )
                logging.info(
                    'To recalculate, set "overwrite_extrinsics" to true in your settings.'
                )
            else:
                calibration.calibrate_intrinsics(source)

        return data_in

    def from_config(self, config):
        self.settings = CalibrationSettings.from_config(config)

    def save_data(self, data_out):
        toml_write(self.sources, self.settings.calib_output_path)


def toml_write(sources, calib_output_path):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''
    data = {}

    for source in sources:
        if len(source.S) == 0 and len(source.D) == 0 and len(source.K) == 0 and len(source.R) == 0 and len(source.T) == 0:
            logging.warning(f"[{source.name}] has not been calibrated.")
    
        else:
            source.calculate_calibration_residuals()

            data[source.name] = {
                "name": source.name,
                "size": [float(source.S[0]), float(source.S[1])],
                "matrix": [
                    [float(source.K[0, 0]), 0.0,                float(source.K[0, 2])],
                    [0.0,                   float(source.K[1, 1]), float(source.K[1, 2])],
                    [0.0,                   0.0,                1.0]
                ],
                "distortions": [
                    float(source.D[0]), float(source.D[1]),
                    float(source.D[2]), float(source.D[3])
                ],
                "rotation": [
                    float(source.R[0]), float(source.R[1]), float(source.R[2])
                ],
                "translation": [
                    float(source.T[0]), float(source.T[1]), float(source.T[2])
                ],
                "fisheye": False
            }
    
    data["metadata"] = {
        "adjusted": False,
        "error": 0.0
    }
    
    with open(calib_output_path, 'w') as f:
        toml.dump(data, f)

    logging.info(f'Calibration file is stored at {calib_output_path}.')


def trc_write(object_coords_3d, trc_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - object_coords_3d: list of 3D point lists
    - trc_path: output path of the trc file

    OUTPUT:
    - trc file with 2 frames of the same 3D points
    '''

    #Header
    DataRate = CameraRate = OrigDataRate = 1
    NumFrames = 2
    NumMarkers = len(object_coords_3d)
    keypoints_names = np.arange(NumMarkers)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(str(k) for k in keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(NumMarkers)])]
    
    # Zup to Yup coordinate system
    object_coords_3d = pd.DataFrame([np.array(object_coords_3d).flatten(), np.array(object_coords_3d).flatten()])
    object_coords_3d = zup2yup(object_coords_3d)
    
    #Add Frame# and Time columns
    object_coords_3d.index = np.array(range(0, NumFrames)) + 1
    object_coords_3d.insert(0, 't', object_coords_3d.index / DataRate)

    #Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        object_coords_3d.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path
