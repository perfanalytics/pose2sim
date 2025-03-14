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
import sys
import pandas as pd
from pathlib import Path
from copy import deepcopy
import numpy as np

from Pose2Sim.model import PoseModel
from Pose2Sim.common import natural_sort_key, zup2yup
from Pose2Sim.MarkerAugmenter import utilsDataman
from Pose2Sim.source import WebcamSource, ImageSource, VideoSource
from Pose2Sim.subject import Subject
from Pose2Sim.calibration import QcaCalibration, OptitrackCalibration, ViconCalibration, EasyMocapCalibration, BiocvCalibration, OpencapCalibration, CheckerboardCalibration


class SubConfig:
    def __init__(self, config_dict, session_dir):
        '''
        Initializes a sub-configuration from a dictionary and the session_dir
        calculated by the Config class.
        '''
        self._config = config_dict
        self.session_dir = session_dir
        self.subjects = self.subjects()
        self.sources = self.sources()

    def subjects(self):
        """
        Construit une liste d'objets Subject à partir de la config TOML.
        """
        subjects_data_list = self._config.get("subjects", [])
        subjects_list = []
        for sub_data in subjects_data_list:
            subject_obj = Subject(self, sub_data)
            subjects_list.append(subject_obj)
        return subjects_list
    
    def sources(self):
        sources_data_list = self._config.get("sources", [])
        sources_list = []
        for src_data in sources_data_list:
            path_val = src_data.get("path", "")
            if isinstance(path_val, int):
                source_obj = WebcamSource(self, src_data)
            else:
                path_str = str(path_val)
                abs_path = os.path.abspath(path_str)

                if os.path.isdir(abs_path):
                    source_obj = ImageSource(self, src_data)

                elif os.path.isfile(abs_path):
                    source_obj = VideoSource(self, src_data)

                elif path_str.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    logging.error(f"Video file '{path_str}' not found.")
                    raise FileNotFoundError(f"Video file '{path_str}' not found.")

                elif path_str.endswith(("/", "\\")):
                    logging.error(f"Folder '{path_str}' not found.")
                    raise FileNotFoundError(f"Folder '{path_str}' not found.")
                
                else:
                    logging.error(f"Unable to create a source from '{path_str}'.")
                    raise FileNotFoundError(f"Unable to create a source from '{path_str}'.")

            sources_list.append(source_obj)
        return sources_list

    @property
    def calibration(self):
        return self._config.get("calibration", {})

    @property
    def calib_dir(self):
        '''
        Returns the calibration folder based on session_dir.
        If no folder containing 'calib' is found, an exception is raised.
        '''
        try:
            calib_dirs = [
                os.path.join(self.session_dir, c)
                for c in os.listdir(self.session_dir)
                if "calib" in c.lower() and os.path.isdir(os.path.join(self.session_dir, c))
            ]
            return calib_dirs[0]
        except IndexError:
            raise FileNotFoundError("No calibration folder found in the project directory.")

    @property
    def extrinsics_corners_nb(self):
        return self.calibration.get("calculate", {}).get("extrinsics", {}).get("board", {}).get("extrinsics_corners_nb")

    @property
    def extrinsics_square_size(self):
        return self.calibration.get("calculate", {}).get("extrinsics", {}).get("board", {}).get("extrinsics_square_size") / 1000.0

    @property
    def calculate_extrinsics(self):
        return self.calibration.get("calculate", {}).get("extrinsics", {}).get("calculate_extrinsics")
    
    @property
    def extract_every_N_sec(self):
        return self.calibration.get("calculate", {}).get("intrinsics", {}).get("extract_every_N_sec")

    @property
    def overwrite_extraction(self):
        return False

    @property
    def show_detection_intrinsics(self):
        return self.calibration.get("calculate", {}).get("intrinsics", {}).get("show_detection_intrinsics")

    @property
    def intrinsics_corners_nb(self):
        return self.calibration.get("calculate", {}).get("intrinsics", {}).get("intrinsics_corners_nb")

    @property
    def intrinsics_square_size(self):
        return self.calibration.get("calculate", {}).get("intrinsics", {}).get("intrinsics_square_size") / 1000.0

    @property
    def logging(self):
        return self._config.get("logging", {})

    @property
    def filtering(self):
        return self._config.get("filtering", {})

    @property
    def kinematics(self):
        return self._config.get("kinematics", {})
    
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

    @property
    def markerAugmentation(self):
        return self._config.get("markerAugmentation", {})

    @property
    def triangulation(self):
        return self._config.get("triangulation", {})

    @property
    def synchronization(self):
        return self._config.get("synchronization", {})

    @property
    def personAssociation(self):
        return self._config.get("personAssociation", {})
    
    @property
    def project(self):
        return self._config.get("project", {})

    @property
    def project_dir(self):
        '''
        Returns the absolute path to the project directory.
        Raises FileNotFoundError if the directory is not found.
        '''
        project_dir = self.project.get("project_dir")
        abs_path = os.path.realpath(project_dir)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Project directory does not exist: {abs_path}")
        return abs_path

    @property
    def frame_range(self):
        '''
        Returns the frame range specified in the configuration.
        '''
        return self.project.get("frame_range")

    @property
    def pose(self):
        return self._config.get("pose", {})

    @property
    def pose_model(self):
        '''
        Returns the PoseModel object initialized from the configuration.
        '''
        return PoseModel.from_config(self.pose.get("pose_model"))

    @property
    def osim_setup_dir(self):
        '''
        Returns the path to the 'OpenSim_Setup' directory within the Pose2Sim package.
        '''
        pose2sim_path = Path(sys.modules["Pose2Sim"].__file__).resolve().parent
        return pose2sim_path / "OpenSim_Setup"

    def calibrate_sources(self):
        calib_data = toml.load(self.calib_file)
        for source in self.sources:
            if source.name in calib_data:
                data = calib_data[source.name]
                source.ret = 0.0
                source.S = data['size']
                source.K = np.array(data['matrix'])
                source.D = data['distortions']
                source.R = [0.0, 0.0, 0.0]
                source.T = [0.0, 0.0, 0.0]
            else :
                logging.info(f"[{source.name}] No existing calibration found.")

    @property
    def object_coords_3d(self):
        object_coords_3d = self.calibration.get("calculate", {}).get("extrinsics", {}).get("object_coords_3d", {})
        if self.extrinsics_method in {'board', 'scene'}:
            # Define 3D object points
            if self.extrinsics_method == 'board':
                object_coords_3d = np.zeros((self.extrinsics_corners_nb[0] * self.extrinsics_corners_nb[1], 3), np.float32)
                object_coords_3d[:, :2] = np.mgrid[0:self.extrinsics_corners_nb[0], 0:self.extrinsics_corners_nb[1]].T.reshape(-1, 2)
                object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * self.extrinsics_square_size
                return object_coords_3d
            elif self.extrinsics_method == 'scene':
                return object_coords_3d

            elif self.extrinsics_method == 'keypoints':
                raise NotImplementedError('This has not been integrated yet.')

        else:
            raise ValueError('Wrong value for extrinsics_method')

    def get_calibration_params(self):
        '''
        Extracts and returns calibration parameters from the configuration.

        Returns a tuple:
          - calib_output_path: path to the calibration file to write
          - calib_full_type: full calibration type (e.g. 'convert_qualisys', 'calculate', etc.)
          - args_calib_fun: arguments to pass to the calibration function

        In case of error (unknown calibration type or missing file), raises an exception.
        '''
        calib_type = self.calibration.get("calibration_type")
        overwrite_intrinsics = self.calibration.get("calculate", {}).get("intrinsics", {}).get("overwrite_intrinsics", False)
        overwrite_extrinsics = True

        convert_path = None

        if calib_type == "convert":
            convert_path = self.calibration.get("convert", {}).get("convert_from")
            if not convert_path:
                raise NameError("Conversion file path not specified in configuration.")

            if not os.path.isabs(convert_path):
                convert_path = os.path.join(self.calib_dir, convert_path)

            if not os.path.exists(convert_path):
                raise NameError(f"File {convert_path} not found in {self.calib_dir}.")
            
            filename = os.path.basename(convert_path).lower()

        if self.calib_file:
            self.calibrate_sources()
            data = toml.load(self.calib_output_path)
        else:
            data = {}

        extrinsinc = False
        for source in self.sources:
            if not overwrite_intrinsics and len(source.S) != 0 and len(source.D) != 0 and len(source.K) != 0:
                logging.info(
                    f"[{source.name} - intrinsic] Intrinsic calibration loaded from '{os.path.relpath(self.calib_file)}'."
                )
                logging.info(
                    'To recalculate, set "overwrite_intrinsics" to true in Config.toml.'
                )
            else:
                if calib_type == "convert":
                    source.intrinsics_files = convert_path
                elif calib_type == "calculate":
                    source.intrinsics_files = source.get_calib_files(source.calib_intrinsics, self.intrinsics_extension, "intrinsics")

            if not overwrite_extrinsics and len(source.R) != 0 and len(source.T) != 0:
                logging.info(
                    f"[{source.name} - entrinsic] Extrinsic calibration loaded from '{os.path.relpath(self.calib_file)}'."
                )
                logging.info(
                    'To recalculate, set "overwrite_extrinsics" to true in Config.toml.'
                )
            else:
                extrinsinc = True
                if calib_type == "convert":
                    source.extrinsics_files = convert_path
                elif calib_type == "calculate":
                    source.extrinsics_files = source.get_calib_files(source.calib_extrinsics, self.extrinsics_extension, "extrinsics")

        if calib_type == "convert":
            if filename.endswith(".qca.txt"):
                calibration = QcaCalibration(self, self.calibration.get("convert", {}).get("qualisys", {}).get("binning_factor", 1))
            elif filename.endswith(".xcp"):
                calibration = ViconCalibration(self, 1)
            elif filename.endswith(".pickle"):
                calibration = OpencapCalibration(self, 1)
            elif filename.endswith(".yml"):
                calibration = EasyMocapCalibration(self, 1)
            elif filename.endswith(".calib"):
                calibration = BiocvCalibration(self, 1)
            elif filename.endswith(".csv"):
                calibration = OptitrackCalibration(self, 1)
            elif any(filename.endswith(ext) for ext in [".anipose", ".freemocap", ".caliscope"]):
                logging.info("\n--> No conversion required for Caliscope, AniPose, or FreeMocap. Calibration will be ignored.\n")
                return None
            else:
                raise NameError(f"File {filename} not supported for conversion.")

        elif calib_type == "calculate":
            if extrinsinc:
                trc_write(self.object_coords_3d, os.path.join(self.calib_dir, f'Object_points.trc'))
            calibration = CheckerboardCalibration(self, None)

        else:
            logging.info("Invalid calibration_type in Config.toml")
            return ValueError("Invalid calibration_type in Config.toml")

        return data, calibration, convert_path

    @property
    def calib_file(self):
        calib_file = glob.glob(os.path.join(self.calib_dir, "Calib.toml"))
        if len(calib_file) == 0:
            logging.info("No existing calibration file found.")
            return None
        return calib_file[0]
    
    @property 
    def calib_output_path(self):
        return os.path.join(self.calib_dir, f"Calib.toml")

    @property 
    def intrinsics_extension(self):
        return self.calibration.get("calculate", {}).get("intrinsics", {}).get("intrinsics_extension")

    @property 
    def extrinsics_method(self):
        return self.calibration.get("calculate", {}).get("extrinsics", {}).get("extrinsics_method")

    @property 
    def extrinsics_extension(self):
        if self.extrinsics_method == "board":
            return self.calibration.get("calculate", {}).get("extrinsics", {}).get("board", {}).get("extrinsics_extension")
        else:
            return self.calibration.get("calculate", {}).get("extrinsics", {}).get("scene", {}).get("extrinsics_extension")

    @property 
    def show_reprojection_error(self):
        if self.extrinsics_method == "board":
            return self.calibration.get("calculate", {}).get("extrinsics", {}).get("board", {}).get("show_reprojection_error")
        else:
            return self.calibration.get("calculate", {}).get("extrinsics", {}).get("scene", {}).get("show_reprojection_error")

    def get_filtering_params(self):
        '''
        Returns parameters related to data filtering, such as input TRC paths,
        output TRC paths, filter type, frame rate, etc.
        '''
        pose3d_dir = os.path.realpath(os.path.join(self.project_dir, "pose-3d"))
        display_figures = self.filtering.get("display_figures")
        filter_type = self.filtering.get("type")
        make_c3d = self.filtering.get("make_c3d")

        video_dir = os.path.join(self.project_dir, "videos")
        vid_img_extension = self.pose_conf.get("vid_img_extension")
        video_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))
        frame_rate = self._config.get("project", {}).get("frame_rate")

        if frame_rate == "auto":
            try:
                cap = cv2.VideoCapture(video_files[0])
                cap.read()
                if cap.read()[0] is False:
                    raise ValueError("Could not read frame.")
                frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
            except:
                frame_rate = 60

        trc_path_in = [file for file in glob.glob(os.path.join(pose3d_dir, "*.trc")) if "filt" not in file]
        trc_f_out = [f"{os.path.basename(t).split('.')[0]}_filt_{filter_type}.trc" for t in trc_path_in]
        trc_path_out = [os.path.join(pose3d_dir, t) for t in trc_f_out]

        return trc_path_in, trc_path_out, filter_type, frame_rate, display_figures, make_c3d

    def get_kinematics_params(self):
        '''
        Returns parameters needed for kinematics calculations (OpenSim),
        including file paths, subject height/mass, and trimming thresholds.
        '''
        subject_height = self.project.get("participant_height")
        subject_mass = self.project.get("participant_mass")

        use_augmentation = self.kinematics.get("use_augmentation")

        pose3d_dir = Path(self.project_dir) / "pose-3d"
        kinematics_dir = Path(self.project_dir) / "kinematics"
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
        pathInputTRCFile = os.path.realpath(os.path.join(self.project_dir, "pose-3d"))
        pathOutputTRCFile = os.path.realpath(os.path.join(self.project_dir, "pose-3d"))

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

        pose_dir = os.path.join(self.project_dir, "pose")
        poseSync_dir = os.path.join(self.project_dir, "pose-sync")
        poseTracked_dir = os.path.join(self.project_dir, "pose-associated")
        error_threshold_triangulation = self.triangulation.get("reproj_error_threshold_triangulation")

        # Retrieve keypoints from model
        model = self.pose_model.load_model_instance()

        return (
            model,
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
        pose_dir = os.path.realpath(os.path.join(self.project_dir, "pose"))
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
        video_dir = os.path.join(self.project_dir, "videos")
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
            seq_name = f"{os.path.basename(os.path.realpath(self.project_dir))}_P{id_person+1}"
        else:
            seq_name = f"{os.path.basename(os.path.realpath(self.project_dir))}"

        pose3d_dir = os.path.join(self.project_dir, "pose-3d")

        # Determine frame rate
        video_dir = os.path.join(self.project_dir, "videos")
        vid_img_extension = self.pose_conf.get("vid_img_extension", "")
        video_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))
        frame_rate = self._config.get("project", {}).get("frame_rate")
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

    def get_pose_estimation_params(self):
        '''
        Returns the parameters required for pose estimation.
        '''
        multi_person = self.project.get("multi_person")
        video_dir = os.path.join(self.project_dir, "videos")
        pose_dir = os.path.join(self.project_dir, "pose")

        mode = self.pose.get("mode")  # e.g. lightweight, balanced, performance
        vid_img_extension = self.pose.get("vid_img_extension")
        output_format = self.pose.get("output_format")
        save_video = "to_video" in self.pose.get("save_video", [])
        save_images = "to_images" in self.pose.get("save_video", [])
        display_detection = self.pose.get("display_detection")
        overwrite_pose = self.pose.get("overwrite_pose")
        det_frequency = self.pose.get("det_frequency")
        tracking_mode = self.pose.get("tracking_mode")
        deepsort_params = self.pose.get("deepsort_params")

        backend = self.pose.get("backend")
        device = self.pose.get("device")

        # Determine frame rate
        video_files = glob.glob(os.path.join(video_dir, "*" + vid_img_extension))
        frame_rate = self.project.get("frame_rate")
        if frame_rate == "auto":
            try:
                cap = cv2.VideoCapture(video_files[0])
                if not cap.isOpened():
                    raise FileNotFoundError(
                        f"Error: Could not open {video_files[0]}. Check that the file exists."
                    )
                frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
                if frame_rate == 0:
                    frame_rate = 30
                    logging.warning(
                        f"Error: Could not retrieve frame rate from {video_files[0]}. Defaulting to 30fps."
                    )
            except:
                frame_rate = 30

        if det_frequency > 1:
            logging.info(
                f"Inference is only run every {det_frequency} frames. In-between frames, pose estimation tracks previously detected points."
            )
        elif det_frequency == 1:
            logging.info("Inference is run on every single frame.")
        else:
            raise ValueError(
                f"Invalid det_frequency: {det_frequency}. Must be an integer >= 1."
            )

        # Example usage check
        try:
            pose_listdirs_names = next(os.walk(pose_dir))[1]
            _ = os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        except StopIteration:
            pass

        if not overwrite_pose:
            logging.info(
                "Skipping pose estimation since it was already done. "
                "Set 'overwrite_pose' to true in Config.toml to run again."
            )
        else:
            logging.info(
                "Overwriting previous pose estimation. "
                "Set 'overwrite_pose' to false in Config.toml to keep the previous results."
            )
            raise  # Intentionally raise to demonstrate logic flow

        return (
            frame_rate,
            mode,
            output_format,
            save_video,
            save_images,
            display_detection,
            multi_person,
            det_frequency,
            video_dir,
            vid_img_extension,
            det_frequency,
            tracking_mode,
            deepsort_params,
            backend,
            device,
        )

    def get_person_association_params(self):
        '''
        Returns parameters for person association (single or multi-person).
        '''
        project_dir = self.project_dir
        multi_person = self.project.get("multi_person")

        min_cameras_for_triangulation = self.triangulation.get("min_cameras_for_triangulation")
        undistort_points = self.triangulation.get("undistort_points")

        tracked_keypoint = self.personAssociation.get("single_person", {}).get("tracked_keypoint")
        reconstruction_error_threshold = self.personAssociation.get("multi_person", {}).get("reconstruction_error_threshold")
        min_affinity = self.personAssociation.get("multi_person", {}).get("min_affinity")
        error_threshold_tracking = self.personAssociation.get("single_person", {}).get("reproj_error_threshold_association")
        likelihood_threshold_association = self.personAssociation.get("likelihood_threshold_association")

        # Retrieve keypoints from model
        model = self.pose_model.load_model_instance()

        try:
            calib_dir = [
                os.path.join(self.session_dir, c)
                for c in os.listdir(self.session_dir)
                if os.path.isdir(os.path.join(self.session_dir, c)) and "calib" in c.lower()
            ][0]
        except:
            raise Exception("No calibration directory found.")

        pose_dir = os.path.join(project_dir, "pose")
        poseSync_dir = os.path.join(project_dir, "pose-sync")
        poseTracked_dir = os.path.join(project_dir, "pose-associated")

        return (
            project_dir,
            self.session_dir,
            model,
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


class Config:
    def __init__(self, config_input=None):
        '''
        Initializes the main configuration.

        Arguments:
          - config_input: Either a configuration dictionary,
                          or the path to the folder containing Config.toml files.
                          If None, the current folder is used.
        '''
        self.config_input = config_input
        self.level, self.config_dicts = self._read_config_files(config_input)
        self.session_dir = self._determine_session_dir()
        self.use_custom_logging = self.config_dicts[0].get("logging", {}).get("use_custom_logging", False)
        self.sub_configs = [SubConfig(cfg, self.session_dir) for cfg in self.config_dicts]

    def _recursive_update(self, base, updates):
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._recursive_update(base[key], value)
            else:
                base[key] = value
        return base

    def _determine_level(self, config_dir):
        len_paths = [
            len(root.split(os.sep))
            for root, dirs, files in os.walk(config_dir)
            if "Config.toml" in files
        ]
        if not len_paths:
            raise FileNotFoundError("You must have a Config.toml file in each trial folder or root folder.")
        level = max(len_paths) - min(len_paths) + 1
        return level

    def _read_config_files(self, config_input):
        if isinstance(config_input, dict):
            # Single config dictionary
            level = 2
            config_dicts = [config_input]
            if config_dicts[0].get("project", {}).get("project_dir") is None:
                raise ValueError("Please specify the project directory in the configuration.")
        else:
            # config_input is a folder path or None
            config_dir = "." if config_input is None else config_input
            level = self._determine_level(config_dir)

            if level == 1:
                try:
                    session_config = toml.load(os.path.join(config_dir, "..", "Config.toml"))
                    trial_config = toml.load(os.path.join(config_dir, "Config.toml"))
                    session_config = self._recursive_update(session_config, trial_config)
                except Exception:
                    session_config = toml.load(os.path.join(config_dir, "Config.toml"))
                session_config.get("project", {}).update({"project_dir": config_dir})
                config_dicts = [session_config]

            elif level == 2:
                session_config = toml.load(os.path.join(config_dir, "Config.toml"))
                config_dicts = []
                for root, dirs, files in os.walk(config_dir):
                    if "Config.toml" in files and root != config_dir:
                        trial_config = toml.load(os.path.join(root, "Config.toml"))
                        temp = deepcopy(session_config)
                        temp = self._recursive_update(temp, trial_config)
                        temp.get("project", {}).update(
                            {"project_dir": os.path.join(config_dir, os.path.relpath(root))}
                        )
                        if os.path.basename(root) not in temp.get("project", {}).get("exclude_from_batch", []):
                            config_dicts.append(temp)
            else:
                raise ValueError("Unsupported configuration level.")

        return level, config_dicts

    def _determine_session_dir(self):
        try:
            if self.level == 2:
                session_dir = os.path.realpath(os.getcwd())
            else:
                session_dir = os.path.realpath(os.path.join(os.getcwd(), ".."))
            calib_dirs = [
                os.path.join(session_dir, d)
                for d in os.listdir(session_dir)
                if os.path.isdir(os.path.join(session_dir, d)) and "calib" in d.lower()
            ]
            if calib_dirs:
                return session_dir
            else:
                return os.path.realpath(os.getcwd())
        except Exception:
            return os.path.realpath(os.getcwd())

    def __repr__(self):
        return (
            f"Config(level={self.level}, nb_configs={len(self.config_dicts)}, "
            f"session_dir='{self.session_dir}', use_custom_logging={self.use_custom_logging})"
        )
    
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