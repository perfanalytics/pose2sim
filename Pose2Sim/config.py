#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
##########################################################################
## CONFIG MODULE                                                        ##
##########################################################################

This module defines:
- The main Config class that loads and merges Config.toml files.
- The SubConfig subclass that encapsulates an individual configuration
  and directly exposes useful properties and methods to get the settings.
'''

import os
import glob
import logging
import toml
import cv2
import math
import sys
from pathlib import Path
import re
import json
import ast

from Pose2Sim.common import natural_sort_key
from Pose2Sim.MarkerAugmenter import utilsDataman
from Pose2Sim.model import PoseModel
from Pose2Sim.source import WebcamSource


class Config:
    def __init__(self, config_input=None):
        self.config_input = config_input
        self.config_dict = [self._build_merged_config(config_input)][0]

        self.fps = None

    def _recursive_update(self, base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._recursive_update(base[key], value)
            else:
                base[key] = value
        return base

    def _build_merged_config(self, config_input):
        if isinstance(config_input, dict):
            return config_input

        if config_input is None:
            current_dir = os.getcwd()
        else:
            current_dir = os.path.realpath(config_input)

        config_paths = self._find_parent_configs(current_dir)

        merged_config = {}
        for cfg_file in reversed(config_paths):
            try:
                current_cfg = toml.load(cfg_file)
                merged_config = self._recursive_update(merged_config, current_cfg)
            except Exception as e:
                raise ValueError(f"Unable to read {cfg_file}: {str(e)}")

        if not merged_config:
            raise FileNotFoundError("No Config.toml found.")

        return merged_config

    def _find_parent_configs(self, start_dir):
        configs_found = []
        current_dir = start_dir

        while True:
            config_toml = os.path.join(current_dir, "Config.toml")
            if os.path.isfile(config_toml):
                configs_found.append(config_toml)

            parent = os.path.dirname(current_dir)
            if parent == current_dir:
                break
            current_dir = parent

        return configs_found

    @property
    def session_dir(self):
        return os.path.realpath(os.getcwd())

    # Project

    @property
    def project(self):
        return self.config_dict.get("project")

    @property
    def frame_rate(self):
        return self.project.get("frame_rate")

    @property 
    def frame_range(self):
        frame_range = self.project.get('frame_range')
        if not frame_range:
            return None
        if len(frame_range) == 2 and all(isinstance(x, int) for x in frame_range):
            return set(range(frame_range))
        return set(frame_range)

    # Pose

    @property
    def pose(self):
        return self.config_dict.get("pose")

    @property
    def pose_model(self):
        return self.pose.get('pose_model')

    @property
    def mode(self):
        return self.pose.get('mode')
    
    @property
    def det_frequency(self):
        return self.config_dict.get("det_frequency")

    @property
    def backend(self):
        return self.pose.get('backend')

    @property
    def device(self):
        return self.pose.get('device')

    # PoseEstimation

    @property
    def pose_dir(self):
        return os.path.join(self.session_dir, 'pose')

    def set_fps(self, value):
        self.fps = value
        logging.info(f'[Pose estimation] capture frame rate set to: {self.fps}.')

    @property
    def pose_estimation(self):
        return self.config_dict.get("poseEstimation")
    
    @property
    def output_format(self):
        return self.pose_estimation.get('output_format')

    @property
    def webcam_recording(self):
        return self.pose_estimation.get('webcam_recording')
    
    @property
    def save_files(self):
        save_files = self.pose_estimation.get('save_video')
        save_images = ('to_images' in save_files)
        save_video = ('to_video' in save_files)
        return save_video, save_images
    
    @property
    def tracking_mode(self):
        return self.pose_estimation.get('tracking_mode')
    
    @property
    def multi_person(self):
        return self.pose_estimation.get('multi_person')

    @property 
    def combined_frames(self):
        return self.pose_estimation.get('combined_frames')

    @property 
    def multi_workers(self):
        return self.pose_estimation.get('multi_workers')

    def check_pose_estimation(self):
        overwrite_pose = self.pose_estimation.get('overwrite_pose')

        for source in self.sources:
            if not isinstance(source, WebcamSource):
                if os.path.exists(os.path.join(self.pose_dir, source.name)) and not overwrite_pose:
                    logging.info(f'[{source.name} - pose estimation] Skipping as it has already been done.'
                                'To recalculate, set overwrite_pose to true in Config.toml.')
                    return
                elif os.path.exists(os.path.join(self.pose_dir, source.name)) and overwrite_pose:
                    logging.info(f'[{source.name} - pose estimation] Overwriting estimation results.')
    
    def get_deepsort_params(self):
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except:  # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/', ':/').replace('":"\\', ':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params)  # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        
        return deepsort_params

    # Logging

    @property
    def use_custom_logging(self):
        return self.config_dict.get("logging").get("use_custom_logging")

    # Calibration

    @property
    def calib_output_path(self):
        return os.path.join(self.session_dir, f"Calib.toml")

    @property
    def calibration(self):
        return self.config_dict.get("calibration")

    @property
    def calib_type(self):
        return self.calibration.get("calibration_type")

    @property
    def overwrite_intrinsics(self):
        return self.calibration.get("overwrite_intrinsics")

    @property
    def overwrite_extrinsics(self):
        return self.calibration.get("overwrite_extrinsics")

    @property
    def calculate_extrinsics(self):
        return self.calibration.get("calculate_extrinsics")

    # Calibration - Convert

    @property
    def convert_path(self):
        convert_path = self.calibration.get("convert").get("convert_from")    
        if not convert_path:
            raise NameError("Conversion file path not specified in configuration.")

        if not os.path.isabs(convert_path):
            convert_path = os.path.join(self.session_dir, convert_path)

        if not os.path.exists(convert_path):
            raise NameError(f"File {convert_path} not found.")
        return convert_path
    
    # Calibration - Convert - Qualisys

    @property
    def binning_factor_qualisys(self):
        return self.config.calibration.get("convert").get("qualisys").get("binning_factor", 1)

    # Calibration - Calculate

    @property
    def overwrite_extraction(self):
        return self.calibration.get("calculate").get("overwrite_extraction")

    @property
    def extract_every_N_sec(self):
        return self.calibration.get("calculate").get("extract_every_N_sec")

    # Calibration - Calculate - Intrinsics

    @property
    def show_detection_intrinsics(self):
        return self.calibration.get("calculate").get("intrinsics").get("show_detection_intrinsics")

    @property
    def intrinsics_corners_nb(self):
        return self.calibration.get("calculate").get("intrinsics").get("intrinsics_corners_nb")

    @property
    def intrinsics_square_size(self):
        return self.calibration.get("calculate").get("intrinsics").get("intrinsics_square_size") / 1000.0

    @property
    def intrinsics_extension(self):
        return self.calibration.get("calculate").get("intrinsics").get("intrinsics_extension")

    # Calibration - Calculate - Extrinsics

    @property
    def extrinsics_method(self):
        return self.calibration.get("calculate").get("extrinsics").get("extrinsics_method")

    @property
    def extrinsics_extension(self):
        return self.calibration.get("calculate").get("extrinsics").get("extrinsics_extension")

    @property
    def show_reprojection_error(self):
        return self.calibration.get("calculate").get("extrinsics").get("show_reprojection_error")

    # Calibration - Calculate - Extrinsics - Board

    @property
    def extrinsics_corners_nb(self):
        return self.calibration.get("calculate").get("extrinsics").get("board").get("extrinsics_corners_nb")

    @property
    def extrinsics_square_size(self):
        return self.calibration.get("calculate").get("extrinsics").get("board").get("extrinsics_square_size") / 1000.0

    # Calibration - Calculate - Extrinsics - Scene

    @property
    def object_coords_3d(self):
        return self.calibration.get("calculate").get("extrinsics").get("scene").get("object_coords_3d")

    # Filtering

    @property
    def filtering(self):
        return self.config_dict.get("filtering")

    # Kinematics

    @property
    def kinematics(self):
        return self.config_dict.get("kinematics")
    
    @property
    def fastest_frames_to_remove_percent(self):
        return self.kinematics.get("fastest_frames_to_remove_percent")
    
    @property
    def close_to_zero_speed_m(self):
        return self.kinematics.get("close_to_zero_speed_m")
    
    @property
    def large_hip_knee_angles(self):
        return self.kinematics.get("large_hip_knee_angles")
    
    @property
    def trimmed_extrema_percent(self):
        return self.kinematics.get("trimmed_extrema_percent")

    @property
    def default_height(self):
        return self.kinematics.get("default_height")

    # Marker Augmentation

    @property
    def markerAugmentation(self):
        return self.config_dict.get("markerAugmentation")

    # Triangulation

    @property
    def triangulation(self):
        return self.config_dict.get("triangulation")

    # Synchronization

    @property
    def synchronization(self):
        return self.config_dict.get("synchronization")

    # Person Association

    @property
    def personAssociation(self):
        return self.config_dict.get("personAssociation")

    @property
    def osim_setup_dir(self):
        '''
        Returns the path to the 'OpenSim_Setup' directory within the Pose2Sim package.
        '''
        pose2sim_path = Path(sys.modules["Pose2Sim"].__file__).resolve().parent
        return pose2sim_path / "OpenSim_Setup"


    def get_filtering_params(self):
        '''
        Returns parameters related to data filtering, such as input TRC paths,
        output TRC paths, filter type, frame rate, etc.
        '''
        pose3d_dir = os.path.realpath(os.path.join(self.session_dir, "pose-3d"))
        display_figures = self.filtering.get("display_figures")
        filter_type = self.filtering.get("type")
        make_c3d = self.filtering.get("make_c3d")

        video_dir = os.path.join(self.session_dir, "videos")
        vid_img_extension = self.pose_conf.get("vid_img_extension")
        video_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))

        trc_path_in = [file for file in glob.glob(os.path.join(pose3d_dir, "*.trc")) if "filt" not in file]
        trc_f_out = [f"{os.path.basename(t).split('.')[0]}_filt_{filter_type}.trc" for t in trc_path_in]
        trc_path_out = [os.path.join(pose3d_dir, t) for t in trc_f_out]

        return trc_path_in, trc_path_out, filter_type, display_figures, make_c3d

    def get_kinematics_params(self):
        '''
        Returns parameters needed for kinematics calculations (OpenSim),
        including file paths, subject height/mass, and trimming thresholds.
        '''
        subject_height = self.project.get("participant_height")
        subject_mass = self.project.get("participant_mass")

        use_augmentation = self.kinematics.get("use_augmentation")

        pose3d_dir = Path(self.session_dir) / "pose-3d"
        kinematics_dir = Path(self.session_dir) / "kinematics"
        kinematics_dir.mkdir(parents=True, exist_ok=True)
        opensim_logs_file = kinematics_dir / "opensim_logs.txt"

        # Possibly override the pose model if we use marker augmentation
        if use_augmentation:
            self.pose_model = PoseModel.LSTM

        # Find all TRC files
        if use_augmentation:
            trc_files = [f for f in pose3d_dir.glob("*.trc") if "_LSTM" in f.name]
            if not trc_files:
                use_augmentation = False
                logging.warning("No LSTM TRC file found. Using non-augmented TRC files.")
        else:
            trc_files = []

        if not trc_files:
            trc_files = [
                f
                for f in pose3d_dir.glob("*.trc")
                if "_LSTM" not in f.name and "_filt" in f.name and "_scaling" not in f.name
            ]

        if not trc_files:
            trc_files = [
                f
                for f in pose3d_dir.glob("*.trc")
                if "_LSTM" not in f.name and "_scaling" not in f.name
            ]

        if not trc_files:
            raise ValueError(f"No TRC file found in {pose3d_dir}.")

        # Create list of dicts describing the paths for each TRC file
        trc_files_info = []
        for trc_file in sorted(trc_files, key=natural_sort_key):
            file_info = {
                "trc_file": trc_file.resolve(),
                "scaled_model_path": (kinematics_dir / (trc_file.stem + ".osim")).resolve(),
                "scaling_setup_path": (kinematics_dir / (trc_file.stem + "_scaling_setup.xml")).resolve(),
                "ik_setup_path": (kinematics_dir / (trc_file.stem + "_ik_setup.xml")).resolve(),
                "output_motion_file": (kinematics_dir / (trc_file.stem + ".mot")).resolve(),
            }
            trc_files_info.append(file_info)

        if not subject_height or subject_height == 0:
            subject_height = [1.75] * len(trc_files)
            logging.warning("No subject height found in Config.toml. Using default of 1.75m.")

        if not subject_mass or subject_mass == 0:
            subject_mass = [70] * len(trc_files)
            logging.warning("No subject mass found in Config.toml. Using default of 70kg.")

        return (
            opensim_logs_file,
            trc_files,
            kinematics_dir,
        )

    def get_scaling_params(self):
        '''
        Returns parameters needed for scaling an OpenSim model.
        '''
        geometry_path = str(self.osim_setup_dir / "Geometry")

        use_contacts_muscles = self.kinematics.get("use_contacts_muscles")
        if use_contacts_muscles:
            pose_model_file = "Model_Pose2Sim_contacts_muscles.osim"
        else:
            pose_model_file = "Model_Pose2Sim.osim"

        unscaled_model_path = self.osim_setup_dir / pose_model_file
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")

        markers_path = self.osim_setup_dir / self.pose_model.marker_file
        scaling_path = self.osim_setup_dir / self.pose_model.scaling_file

        right_left_symmetry = self.kinematics.get("right_left_symmetry")
        remove_scaling_setup = self.kinematics.get("remove_individual_scaling_setup")

        return geometry_path, unscaled_model_path, markers_path, scaling_path, right_left_symmetry, remove_scaling_setup

    def get_performI_K_params(self):
        '''
        Returns IK (Inverse Kinematics) setup parameters.
        '''
        ik_path = self.osim_setup_dir / self.pose_model.ik_file
        remove_IK_setup = self.kinematics.get("remove_individual_ik_setup")

        return ik_path, remove_IK_setup

    def get_augment_markers_params(self):
        '''
        Returns parameters needed for marker augmentation (LSTM-based).
        '''
        pathInputTRCFile = os.path.realpath(os.path.join(self.session_dir, "pose-3d"))
        pathOutputTRCFile = os.path.realpath(os.path.join(self.session_dir, "pose-3d"))

        make_c3d = self.markerAugmentation.get("make_c3d")

        subject_height = self.project.get("participant_height")
        subject_mass = self.project.get("participant_mass")

        fastest_frames_to_remove_percent = self.kinematics.get("fastest_frames_to_remove_percent")
        close_to_zero_speed = self.kinematics.get("close_to_zero_speed_m")
        large_hip_knee_angles = self.kinematics.get("large_hip_knee_angles")
        trimmed_extrema_percent = self.kinematics.get("trimmed_extrema_percent")

        augmenterDir = os.path.dirname(utilsDataman.__file__)
        augmenterModelName = "LSTM"
        augmenter_model = "v0.3"
        offset = True

        # Determine which TRC files to augment
        all_trc_files = [
            f for f in glob.glob(os.path.join(pathInputTRCFile, "*.trc")) if "_LSTM" not in f
        ]
        trc_no_filtering = [
            f for f in all_trc_files if "filt" not in os.path.basename(f)
        ]
        trc_filtering = [
            f for f in all_trc_files if "filt" in os.path.basename(f)
        ]

        if len(all_trc_files) == 0:
            raise ValueError("No TRC files found.")

        if len(trc_filtering) > 0:
            trc_files = trc_filtering
        else:
            trc_files = trc_no_filtering
        trc_files = sorted(trc_files, key=natural_sort_key)

        return (
            subject_height,
            subject_mass,
            trc_files,
            fastest_frames_to_remove_percent,
            close_to_zero_speed,
            large_hip_knee_angles,
            trimmed_extrema_percent,
            augmenter_model,
            augmenterDir,
            augmenterModelName,
            pathInputTRCFile,
            pathOutputTRCFile,
            make_c3d,
            offset,
        )

    def get_triangulation_params(self):
        '''
        Returns parameters needed for triangulation.
        '''

        multi_person = self.project.get("multi_person")
        likelihood_threshold = self.triangulation.get("likelihood_threshold_triangulation")
        interpolation_kind = self.triangulation.get("interpolation")
        interp_gap_smaller_than = self.triangulation.get("interp_if_gap_smaller_than")
        fill_large_gaps_with = self.triangulation.get("fill_large_gaps_with")
        show_interp_indices = self.triangulation.get("show_interp_indices")
        undistort_points = self.triangulation.get("undistort_points")
        make_c3d = self.triangulation.get("make_c3d")

        pose_dir = os.path.join(self.session_dir, "pose")
        poseSync_dir = os.path.join(self.session_dir, "pose-sync")
        poseTracked_dir = os.path.join(self.session_dir, "pose-associated")
        error_threshold_triangulation = self.triangulation.get("reproj_error_threshold_triangulation")
    
        return (
            pose_dir,
            poseSync_dir,
            poseTracked_dir,
            multi_person,
            likelihood_threshold,
            interpolation_kind,
            interp_gap_smaller_than,
            fill_large_gaps_with,
            show_interp_indices,
            undistort_points,
            make_c3d,
            error_threshold_triangulation,
        )

    def get_triangulation_from_best_cameras_params(self):
        '''
        Returns a tuple of parameters used for triangulation from the best cameras only.
        '''
        error_threshold_triangulation = self.triangulation.get("reproj_error_threshold_triangulation")
        min_cameras_for_triangulation = self.triangulation.get("min_cameras_for_triangulation")
        handle_LR_swap = self.triangulation.get("handle_LR_swap")
        undistort_points = self.triangulation.get("undistort_points")

        logging.info(f"Limb swapping was {'handled' if handle_LR_swap else 'not handled'}.")
        logging.info(f"Lens distortions were {'accounted for' if undistort_points else 'not accounted for'}.")

        return error_threshold_triangulation, min_cameras_for_triangulation, handle_LR_swap, undistort_points

    def get_synchronization_params(self):
        """
        Returns parameters for synchronization.
        """
        pose_dir = os.path.realpath(os.path.join(self.session_dir, "pose"))
        fps = self.project.get("frame_rate")
        display_sync_plots = self.synchronization.get("display_sync_plots")
        keypoints_to_consider = self.synchronization.get("keypoints_to_consider")
        approx_time_maxspeed = self.synchronization.get("approx_time_maxspeed")
        time_range_around_maxspeed = self.synchronization.get("time_range_around_maxspeed")
        synchronization_gui = self.synchronization.get("synchronization_gui")

        likelihood_threshold = self.synchronization.get("likelihood_threshold")
        filter_cutoff = int(self.synchronization.get("filter_cutoff", 0))
        filter_order = int(self.synchronization.get("filter_order", 0))

        # Determine frame rate from the first video if set to 'auto'
        video_dir = os.path.join(self.session_dir, "videos")
        vid_img_extension = self.pose_conf.get("vid_img_extension", "")
        vid_or_img_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))

        # If not video, check inside subdirectories
        if not vid_or_img_files:
            image_folders = [
                f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))
            ]
            for image_folder in image_folders:
                vid_or_img_files.extend(glob.glob(os.path.join(video_dir, image_folder, "*" + vid_img_extension)))

        if fps == "auto":
            try:
                cap = cv2.VideoCapture(vid_or_img_files[0])
                cap.read()
                if cap.read()[0] is False:
                    raise ValueError("Could not read frame.")
                fps = round(cap.get(cv2.CAP_PROP_FPS))
            except:
                fps = 60

        lag_range = time_range_around_maxspeed * fps

        # Retrieve keypoints from model
        model = self.pose_model.load_model_instance()

        return (
            pose_dir,
            fps,
            display_sync_plots,
            keypoints_to_consider,
            approx_time_maxspeed,
            time_range_around_maxspeed,
            synchronization_gui,
            likelihood_threshold,
            filter_cutoff,
            filter_order,
            lag_range,
            model,
            vid_or_img_files,
        )

    def get_make_trc_params(self, f_range, id_person):
        '''
        Returns the paths and info necessary to create a TRC file.
        '''
        multi_person = self.project.get("multi_person", False)

        if multi_person:
            seq_name = f"{os.path.basename(os.path.realpath(self.session_dir))}_P{id_person+1}"
        else:
            seq_name = f"{os.path.basename(os.path.realpath(self.session_dir))}"

        pose3d_dir = os.path.join(self.session_dir, "pose-3d")

        # Determine frame rate
        video_dir = os.path.join(self.session_dir, "videos")
        vid_img_extension = self.pose_conf.get("vid_img_extension", "")
        video_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))
        frame_rate = self.config_dict.get("project", {}).get("frame_rate")
        if frame_rate == "auto":
            try:
                cap = cv2.VideoCapture(video_files[0])
                cap.read()
                if cap.read()[0] is False:
                    raise ValueError("Could not read frame.")
                frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
            except:
                frame_rate = 60

        if not os.path.exists(pose3d_dir):
            os.mkdir(pose3d_dir)

        trc_f = f"{seq_name}_{f_range[0]}-{f_range[1]}.trc"
        trc_path = os.path.realpath(os.path.join(pose3d_dir, trc_f))

        return trc_path, trc_f, frame_rate

    def get_person_association_params(self):
        '''
        Returns parameters for person association (single or multi-person).
        '''
        session_dir = self.session_dir
        multi_person = self.project.get("multi_person")

        min_cameras_for_triangulation = self.triangulation.get("min_cameras_for_triangulation")
        undistort_points = self.triangulation.get("undistort_points")

        tracked_keypoint = self.personAssociation.get("single_person").get("tracked_keypoint")
        reconstruction_error_threshold = self.personAssociation.get("multi_person").get("reconstruction_error_threshold")
        min_affinity = self.personAssociation.get("multi_person").get("min_affinity")
        error_threshold_tracking = self.personAssociation.get("single_person").get("reproj_error_threshold_association")
        likelihood_threshold_association = self.personAssociation.get("likelihood_threshold_association")

        try:
            calib_dir = [
                os.path.join(self.session_dir, c)
                for c in os.listdir(self.session_dir)
                if os.path.isdir(os.path.join(self.session_dir, c)) and "calib" in c.lower()
            ][0]
        except:
            raise Exception("No calibration directory found.")

        pose_dir = os.path.join(session_dir, "pose")
        poseSync_dir = os.path.join(session_dir, "pose-sync")
        poseTracked_dir = os.path.join(session_dir, "pose-associated")

        return (
            session_dir,
            self.session_dir,
            pose_dir,
            poseSync_dir,
            poseTracked_dir,
            multi_person,
            reconstruction_error_threshold,
            min_affinity,
            tracked_keypoint,
            min_cameras_for_triangulation,
            error_threshold_tracking,
            likelihood_threshold_association,
            undistort_points,
        )