#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## CONFIG MODULE                                                        ##
###########################################################################

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
from pathlib import Path
from copy import deepcopy

from Pose2Sim.model import PoseModel
from Pose2Sim.common import natural_sort_key
from Pose2Sim.MarkerAugmenter import utilsDataman

class SubConfig:
    def __init__(self, config_dict, session_dir):
        '''
        Initializes a sub-configuration from a dictionary and the session_dir
        computed by the Config class.
        '''
        self._config = config_dict
        self._session_dir = session_dir

    @property
    def project_dir(self):
        '''
        Returns the absolute path of the project directory.
        Raises a FileNotFoundError if the project directory is not found.
        '''
        proj_dir = self._config.get("project", {}).get("project_dir")
        abs_path = os.path.realpath(proj_dir)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Project directory does not exist: {abs_path}")
        
        return abs_path

    @property
    def session_dir(self):
        '''
        Returns the session folder passed by the Config class.
        '''
        return self._session_dir

    @property
    def calib_dir(self):
        '''
        Returns the session folder passed by the Config class.
        '''
        calib_dir = [os.path.join(self.project_dir, c) for c in os.listdir(self.project_dir) if ('Calib' in c or 'calib' in c)][0]
        return calib_dir

    @property
    def frame_range(self):
        '''
        Returns the frame range defined in the configuration.
        '''
        return self._config.get("project", {}).get("frame_range")

    @property
    def calib_dir(self):
        """
        Retourne le dossier de calibration basé sur session_dir.
        Si aucun dossier contenant "calib" n'est trouvé, une exception est levée.
        """
        if self.session_dir is None:
            raise ValueError("La propriété 'session_dir' n'est pas définie dans la configuration.")
        try:
            calib_dirs = [os.path.join(self.session_dir, c)
                          for c in os.listdir(self.session_dir)
                          if "calib" in c.lower() and os.path.isdir(os.path.join(self.session_dir, c))]
            return calib_dirs[0]
        except IndexError:
            raise FileNotFoundError("Aucun dossier de calibration trouvé dans le dossier projet.")

    @property
    def calibration(self):
        """Retourne la section [calibration] de la configuration."""
        return self._config.get("calibration", {})
    
    @property
    def pose_model(self):
        return PoseModel.from_config(self._config.get("pose", {}).get("pose_model"))

    @property
    def logging(self):
        """Retourne la configuration du logging."""
        return self._config.get("logging", {})
    
    @property
    def object_coords_3d(self):
        return self.calibration.get('calculate', {}).get('extrinsics', {}).get('object_coords_3d', {})
    
    @property
    def extrinsics_corners_nb(self):
        return self.calibration.get('calculate', {}).get('extrinsics', {}).get('board').get('extrinsics_corners_nb')
    
    @property
    def extrinsics_square_size(self):
        return self.calibration.get('calculate', {}).get('extrinsics', {}).get('board').get('extrinsics_square_size') / 1000 # convert to meters

    @property
    def osim_setup_dir(self):
        pose2sim_path = Path(sys.modules['Pose2Sim'].__file__).resolve().parent
        return pose2sim_path / 'OpenSim_Setup'
           

    def get_calibration_params(self):
        """
        Extrait et retourne les paramètres de calibration à partir de la configuration.
        
        Retourne un dictionnaire contenant :
          - calib_output_path : chemin du fichier de calibration à écrire
          - calib_full_type   : type complet de calibration (convert_qualisys, calculate, etc.)
          - args_calib_fun    : arguments à passer à la fonction de calibration
          
        En cas d'erreur (exemple : type de calibration inconnu ou fichier introuvable), une exception est levée.
        """
        calib_dir = self.calib_dir
        calib_settings = self.calibration
        calib_type = calib_settings.get('calibration_type')
        
        if calib_type == 'convert':
            convert_filetype = calib_settings.get('convert').get('convert_from')
            try:
                if convert_filetype == 'qualisys':
                    convert_ext = '.qca.txt'
                    file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}*'))[0]
                    binning_factor = calib_settings.get('convert').get('qualisys').get('binning_factor')
                elif convert_filetype == 'optitrack':
                    file_to_convert_path = ['']
                    binning_factor = 1
                elif convert_filetype == 'vicon':
                    convert_ext = '.xcp'
                    file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}'))[0]
                    binning_factor = 1
                elif convert_filetype == 'opencap':  # tous les fichiers avec extension .pickle
                    convert_ext = '.pickle'
                    file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                    binning_factor = 1
                elif convert_filetype == 'easymocap':  # fichiers .yml
                    convert_ext = '.yml'
                    file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                    binning_factor = 1
                elif convert_filetype == 'biocv':  # fichiers avec extension .calib
                    convert_ext = '.calib'
                    file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                    binning_factor = 1
                elif convert_filetype in ['anipose', 'freemocap', 'caliscope']:
                    logging.info("\n--> Aucune conversion nécessaire pour Caliscope, AniPose ni FreeMocap. Calibration ignorée.\n")
                    return None
                else:
                    convert_ext = '???'
                    file_to_convert_path = ['']
                    raise NameError(f"Calibration conversion from {convert_filetype} is not supported.")
                assert file_to_convert_path != []
            except Exception as e:
                raise NameError(f"Aucun fichier avec l'extension {convert_ext} trouvé dans {calib_dir}.") from e

            calib_output_path = os.path.join(calib_dir, f'Calib_{convert_filetype}.toml')
            calib_full_type = f"{calib_type}_{convert_filetype}"
            args_calib_fun = [file_to_convert_path, binning_factor]

        elif calib_type == 'calculate':
            extrinsics_method = calib_settings.get('calculate').get('extrinsics').get('extrinsics_method')
            calib_output_path = os.path.join(calib_dir, f'Calib_{extrinsics_method}.toml')
            calib_full_type = calib_type
            args_calib_fun = self

        else:
            logging.info("Wrong calibration_type in Config.toml")
            return None

        return calib_output_path, calib_full_type, args_calib_fun
    
    def get_calib_calc_params(self):
        """
        Extrait les paramètres nécessaires pour la calibration (calculate) depuis la section [calibration][calculate].

        Renvoie un dictionnaire contenant :
          - calib_dir                : dossier de calibration
          - intrinsics_config_dict   : dictionnaire des paramètres pour les intrinsics
          - extrinsics_config_dict   : dictionnaire des paramètres pour les extrinsics
          - overwrite_intrinsics     : booléen, True si on doit recalculer les intrinsics
          - calculate_extrinsics     : booléen, True si on doit calculer les extrinsics
          - use_existing_intrinsics  : booléen, True si on récupère les intrinsics depuis un fichier existant
          - calib_file               : chemin vers le fichier de calibration existant (None sinon)
        """
        calib_dir = self.calib_dir
        calculate_section = self.calibration.get('calculate', {})
        intrinsics_config = calculate_section.get('intrinsics', {})
        extrinsics_config = calculate_section.get('extrinsics', {})
        overwrite_intrinsics = intrinsics_config.get('overwrite_intrinsics', False)
        calculate_extrinsics = extrinsics_config.get('calculate_extrinsics', False)
        extrinsics_dir = os.path.join(calib_dir, 'extrinsics')
        
        # Recherche d'un fichier de calibration existant
        calib_file_list = glob.glob(os.path.join(calib_dir, 'Calib*.toml'))
        use_existing_intrinsics = not overwrite_intrinsics and bool(calib_file_list)
        calib_file = calib_file_list[0] if use_existing_intrinsics else None
        
        if use_existing_intrinsics:
            logging.info(f'\nPreexisting calibration file found: \'{calib_file}\'.')
            logging.info(f'\nRetrieving intrinsic parameters from file. Set "overwrite_intrinsics" to true in Config.toml to recalculate them.')
        
        return calculate_extrinsics, use_existing_intrinsics, calib_file, extrinsics_dir
    
    def get_intrinsics_params(self):
        try:
            intrinsics_cam_listdirs_names = next(os.walk(os.path.join(self.calib_dir, 'intrinsics')))[1]
        except StopIteration:
            logging.exception(f'Error: No {os.path.join(self.calib_dir, "intrinsics")} folder found.')
            raise Exception(f'Error: No {os.path.join(self.calib_dir, "intrinsics")} folder found.')
        intrinsics_config = self.calibration.get('calculate').get('intrinsics')
        intrinsics_extension = intrinsics_config.get('intrinsics_extension')
        extract_every_N_sec = intrinsics_config.get('extract_every_N_sec')
        overwrite_extraction = False
        show_detection_intrinsics = intrinsics_config.get('show_detection_intrinsics')
        intrinsics_corners_nb = intrinsics_config.get('intrinsics_corners_nb')
        intrinsics_square_size = intrinsics_config.get('intrinsics_square_size') / 1000 # convert to meters

        img_vid_files_list = []
        for i,cam in enumerate(intrinsics_cam_listdirs_names):
            img_vid_files = glob.glob(os.path.join(self.calib_dir, 'intrinsics', cam, f'*.{intrinsics_extension}'))
            img_vid_files_list.append(cam, img_vid_files)
            if len(img_vid_files) == 0:
                logging.exception(f'The folder {os.path.join(self.calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
                raise ValueError(f'The folder {os.path.join(self.calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')

        return show_detection_intrinsics, intrinsics_corners_nb, extract_every_N_sec, overwrite_extraction, intrinsics_square_size, img_vid_files_list
    
    def get_extrinsics_params(self):
        try:
            extrinsics_cam_listdirs_names = next(os.walk(os.path.join(self.calib_dir, 'extrinsics')))[1]
        except StopIteration:
            logging.exception(f'Error: No {os.path.join(self.calib_dir, "extrinsics")} folder found.')
            raise Exception(f'Error: No {os.path.join(self.calib_dir, "extrinsics")} folder found.')
        
        img_vid_files_list = []
        for i, cam in enumerate(extrinsics_cam_listdirs_names):
            extrinsics_extension = [self.calibration.get('calculate', {}).get('extrinsics', {}).get('board').get('extrinsics_extension') if extrinsics_method == 'board'
                                    else self.calibration.get('calculate', {}).get('extrinsics', {}).get('scene').get('extrinsics_extension')][0]
            show_reprojection_error = [self.calibration.get('calculate', {}).get('extrinsics', {}).get('board').get('show_reprojection_error') if extrinsics_method == 'board'
                                    else self.calibration.get('calculate', {}).get('extrinsics', {}).get('scene').get('show_reprojection_error')][0]
            img_vid_files = glob.glob(os.path.join(self.calib_dir, 'extrinsics', cam, f'*.{extrinsics_extension}'))
            img_vid_files_list.append(cam, img_vid_files)
            if len(img_vid_files) == 0:
                logging.exception(f'The folder {os.path.join(self.calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
                raise ValueError(f'The folder {os.path.join(self.calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
        
        extrinsics_method = self.calibration.get('calculate', {}).get('extrinsics', {}).get('extrinsics_method')
        calib_output_path = os.path.join(self.calib_dir, f'Object_points.trc')

        return show_reprojection_error, extrinsics_method, img_vid_files_list, calib_output_path
    
    def get_filtering_params(self):
        project_dir = self.get('project').get('project_dir')
        pose3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
        display_figures = self.get('filtering').get('display_figures')
        filter_type = self.get('filtering').get('type')
        make_c3d = self.get('filtering').get('make_c3d')

        video_dir = os.path.join(project_dir, 'videos')
        vid_img_extension = self['pose']['vid_img_extension']
        video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
        frame_rate = self.get('project').get('frame_rate')
        if frame_rate == 'auto': 
            try:
                cap = cv2.VideoCapture(video_files[0])
                cap.read()
                if cap.read()[0] == False:
                    raise
                frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
            except:
                frame_rate = 60

        
        # Trc paths
        trc_path_in = [file for file in glob.glob(os.path.join(pose3d_dir, '*.trc')) if 'filt' not in file]
        trc_f_out = [f'{os.path.basename(t).split(".")[0]}_filt_{filter_type}.trc' for t in trc_path_in]
        trc_path_out = [os.path.join(pose3d_dir, t) for t in trc_f_out]

        return trc_path_in, trc_path_out, filter_type, frame_rate, display_figures, make_c3d
    
    def get_kinematics_params(self):
        # Read config_dict
        project_dir = self.get('project').get('project_dir')
        session_dir = self.session_dir
        session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
        use_augmentation = self.get('kinematics').get('use_augmentation')

        subject_height = self.get('project').get('participant_height')
        subject_mass = self.get('project').get('participant_mass')

        fastest_frames_to_remove_percent = self.get('kinematics').get('fastest_frames_to_remove_percent')
        large_hip_knee_angles = self.get('kinematics').get('large_hip_knee_angles')
        trimmed_extrema_percent = self.get('kinematics').get('trimmed_extrema_percent')
        close_to_zero_speed = self.get('kinematics').get('close_to_zero_speed_m')
        default_height = self.get('kinematics').get('default_height')

        pose3d_dir = Path(project_dir) / 'pose-3d'
        kinematics_dir = Path(project_dir) / 'kinematics'
        kinematics_dir.mkdir(parents=True, exist_ok=True)
        opensim_logs_file = kinematics_dir / 'opensim_logs.txt'

        if use_augmentation:
            self.pose_model = PoseModel.LSTM

        # Find all trc files
        if use_augmentation:
            trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' in f.name]
            if not trc_files:
                use_augmentation = False
                logging.warning("Aucun fichier TRC LSTM trouvé. Utilisation des fichiers TRC non augmentés.")
        else:
            trc_files = []

        if not trc_files:
            trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_filt' in f.name and '_scaling' not in f.name]

        if not trc_files:
            trc_files = [f for f in pose3d_dir.glob('*.trc') if '_LSTM' not in f.name and '_scaling' not in f.name]

        if not trc_files:
            raise ValueError(f'Aucun fichier TRC trouvé dans {pose3d_dir}.')

        # Création d'une liste de dictionnaires avec les chemins associés pour chaque fichier TRC
        trc_files_info = []
        for trc_file in sorted(trc_files, key=natural_sort_key):
            file_info = {
                'trc_file': trc_file.resolve(),
                'scaled_model_path': (kinematics_dir / (trc_file.stem + '.osim')).resolve(),
                'scaling_setup_path': (kinematics_dir / (trc_file.stem + '_scaling_setup.xml')).resolve(),
                'ik_setup_path': (kinematics_dir / (trc_file.stem + '_ik_setup.xml')).resolve(),
                'output_motion_file': (kinematics_dir / (trc_file.stem + '.mot')).resolve()
            }
            trc_files_info.append(file_info)

        if subject_height is None or subject_height == 0:
            subject_height = [1.75] * len(trc_files)
            logging.warning("No subject height found in Config.toml. Using default height of 1.75m.")

        if subject_mass is None or subject_mass == 0:
            subject_mass = [70] * len(trc_files)
            logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")

        return opensim_logs_file, trc_files, fastest_frames_to_remove_percent, close_to_zero_speed, large_hip_knee_angles, trimmed_extrema_percent, default_height, kinematics_dir

    def get_scaling_params(self):
        geometry_path = str(self.osim_setup_dir / 'Geometry')
        
        use_contacts_muscles = self.get('kinematics').get('use_contacts_muscles')
        if use_contacts_muscles:
            pose_model_file = 'Model_Pose2Sim_contacts_muscles.osim'
        else:
            pose_model_file = 'Model_Pose2Sim.osim'
        
        unscaled_model_path = self.osim_setup_dir / pose_model_file
        if not unscaled_model_path:
            raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
        markers_path = self.osim_setup_dir / self.pose_model.marker_file
        scaling_path = self.osim_setup_dir / self.pose_model.scaling_file
        
        right_left_symmetry = self.get('kinematics').get('right_left_symmetry')
        remove_scaling_setup = self.get('kinematics').get('remove_individual_scaling_setup')

        return geometry_path, unscaled_model_path, markers_path, scaling_path, right_left_symmetry, remove_scaling_setup
    
    def get_performI_K_params(self):
        ik_path = self.osim_setup_dir / self.pose_model.ik_file
        remove_IK_setup = self.get('kinematics').get('remove_individual_ik_setup')

        return ik_path, remove_IK_setup
    
    def get_augment_markers_params(self):
        project_dir = self.get('project').get('project_dir')
        pathInputTRCFile = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
        pathOutputTRCFile = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
        make_c3d = self.get('markerAugmentation').get('make_c3d')
        subject_height = self.get('project').get('participant_height')
        subject_mass = self.get('project').get('participant_mass')
        
        fastest_frames_to_remove_percent = self.get('kinematics').get('fastest_frames_to_remove_percent')
        close_to_zero_speed = self.get('kinematics').get('close_to_zero_speed_m')
        large_hip_knee_angles = self.get('kinematics').get('large_hip_knee_angles')
        trimmed_extrema_percent = self.get('kinematics').get('trimmed_extrema_percent')
        default_height = self.get('kinematics').get('default_height')

        augmenterDir = os.path.dirname(utilsDataman.__file__)
        augmenterModelName = 'LSTM'
        augmenter_model = 'v0.3'
        offset = True

        # Apply all trc files
        all_trc_files = [f for f in glob.glob(os.path.join(pathInputTRCFile, '*.trc')) if '_LSTM' not in f]
        trc_no_filtering = [f for f in glob.glob(os.path.join(pathInputTRCFile, '*.trc')) if
                            '_LSTM' not in f and 'filt' not in f]
        trc_filtering = [f for f in glob.glob(os.path.join(pathInputTRCFile, '*.trc')) if '_LSTM' not in f and 'filt' in f]

        if len(all_trc_files) == 0:
            raise ValueError('No trc files found.')
        if len(trc_filtering) > 0:
            trc_files = trc_filtering
        else:
            trc_files = trc_no_filtering
        sorted(trc_files, key=natural_sort_key)

        if subject_height is None or subject_height == 0:
            subject_height = [default_height] * len(trc_files)
            logging.warning(f"No subject height found in Config.toml. Using default height of {default_height}m.")

        if subject_mass is None or subject_mass == 0:
            subject_mass = [70] * len(trc_files)
            logging.warning("No subject mass found in Config.toml. Using default mass of 70kg.")

        return trc_files, fastest_frames_to_remove_percent, close_to_zero_speed, large_hip_knee_angles, trimmed_extrema_percent, default_height, augmenter_model, augmenterDir, augmenterModelName, pathInputTRCFile, pathOutputTRCFile, make_c3d, offset


class Config:
    def __init__(self, config_input=None):
        """
        Initialise la configuration principale.
        
        Paramètres :
          - config_input : soit un dictionnaire de configuration,
                           soit le chemin vers le dossier contenant les fichiers Config.toml.
                           Si None, le dossier courant est utilisé.
        """
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
            raise FileNotFoundError("Vous devez avoir un fichier Config.toml dans chaque dossier trial ou racine.")
        level = max(len_paths) - min(len_paths) + 1
        return level

    def _read_config_files(self, config_input):
        if isinstance(config_input, dict):
            level = 2
            config_dicts = [config_input]
            if config_dicts[0].get("project", {}).get("project_dir") is None:
                raise ValueError('Veuillez spécifier le répertoire du projet dans la configuration.')
        else:
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
                        temp.get("project", {}).update({
                            "project_dir": os.path.join(config_dir, os.path.relpath(root))
                        })
                        if not os.path.basename(root) in temp.get("project", {}).get("exclude_from_batch", []):
                            config_dicts.append(temp)
            else:
                raise ValueError("Niveau de configuration non supporté.")
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
        return (f"Config(level={self.level}, nb_configs={len(self.config_dicts)}, "
                f"session_dir='{self.session_dir}', use_custom_logging={self.use_custom_logging})")
