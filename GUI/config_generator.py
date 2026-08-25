from pathlib import Path
import toml


class ConfigGenerator:
    def __init__(self):
        # Load templates from files to avoid comment issues
        self.config_3d_template_path = Path('templates') / '3d_config_template.toml'
        self.config_2d_template_path = Path('templates') /'2d_config_template.toml'
        
        # Create templates directory if it doesn't exist
        Path('templates').mkdir(parents=True, exist_ok=True)
        
        # Write the template files if they don't exist
        self.create_template_files()
    
    def create_template_files(self):
        """Create template files if they don't exist"""
        # Create 3D template file
        if not self.config_3d_template_path.exists():
            with open(self.config_3d_template_path, 'w', encoding='utf-8') as f:
                toml.dump(self.get_3d_template(), f)
        
        # Create 2D template file
        if not self.config_2d_template_path.exists():
            with open(self.config_2d_template_path, 'w', encoding='utf-8') as f:
                toml.dump(self.get_2d_template(), f)
    
    def get_3d_template(self):
        """Return the 3D configuration template"""
        try:
            from Pose2Sim import Pose2Sim
            config_template_3d = toml.load(Path(Pose2Sim.__file__).parent / 'Demo_SinglePerson' / 'Config.toml')
        except:
            # Fallback to default structure if Pose2Sim not available
            config_template_3d = self.get_default_3d_structure()
        return config_template_3d
    
    def get_2d_template(self):
        """Return the 2D configuration template"""
        try:
            from Sports2D import Sports2D
            config_template_2d = toml.load(Path(Sports2D.__file__).parent / 'Demo/Config_demo.toml')
        except:
            # Fallback to default structure if Sports2D not available
            config_template_2d = self.get_default_2d_structure()
        return config_template_2d
    
    def get_default_3d_structure(self):
        """Default 3D config structure if Pose2Sim not installed"""
        return {
                'project': 
                {
                    'multi_person': False,
                    'participant_height': 'auto',
                    'participant_mass': 70.0,
                    'frame_rate': 'auto',
                    'frame_range': 'auto',
                    'exclude_from_batch': []
                },
                'pose': 
                {
                    'vid_img_extension': 'mp4',
                    'pose_model': 'Body_with_feet',
                    'mode': 'balanced',
                    'det_frequency': 4,
                    'device': 'auto',
                    'backend': 'auto',
                    'tracking_mode': 'sports2d',
                    'max_distance_px': 100,
                    'deepsort_params': "{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8}",
                    'handle_LR_swap': False,
                    'undistort_points': False,
                    'display_detection': True,
                    'overwrite_pose': False,
                    'save_video': 'to_video',
                    'output_format': 'openpose',
                    'CUSTOM': 
                    {
                        'name': 'Hip',
                        'id': 19,
                        'children': [{'name': 'RHip',
                            'id': 12,
                            'children': [{'name': 'RKnee',
                            'id': 14,
                            'children': [{'name': 'RAnkle',
                                'id': 16,
                                'children': [{'name': 'RBigToe',
                                'id': 21,
                                'children': [{'name': 'RSmallToe', 'id': 23}]},
                                {'name': 'RHeel', 'id': 25}]}]}]},
                            {'name': 'LHip',
                            'id': 11,
                            'children': [{'name': 'LKnee',
                            'id': 13,
                            'children': [{'name': 'LAnkle',
                                'id': 15,
                                'children': [{'name': 'LBigToe',
                                'id': 20,
                                'children': [{'name': 'LSmallToe', 'id': 22}]},
                                {'name': 'LHeel', 'id': 24}]}]}]},
                            {'name': 'Neck',
                            'id': 18,
                            'children': [{'name': 'Head',
                            'id': 17,
                            'children': [{'name': 'Nose', 'id': 0}]},
                            {'name': 'RShoulder',
                            'id': 6,
                            'children': [{'name': 'RElbow',
                                'id': 8,
                                'children': [{'name': 'RWrist', 'id': 10}]}]},
                            {'name': 'LShoulder',
                            'id': 5,
                            'children': [{'name': 'LElbow',
                                'id': 7,
                                'children': [{'name': 'LWrist', 'id': 9}]}]}]}]
                        }
                    },
                'synchronization': 
                {
                    'synchronization_gui': True,
                    'display_sync_plots': True,
                    'save_sync_plots': True,
                    'keypoints_to_consider': 'all',
                    'approx_time_maxspeed': 'auto',
                    'time_range_around_maxspeed': 2.0,
                    'likelihood_threshold': 0.4,
                    'filter_cutoff': 6,
                    'filter_order': 4
                },
                'calibration': 
                {
                    'calibration_type': 'convert',
                    'convert': 
                    {
                        'convert_from': 'qualisys',
                    'caliscope': {},
                    'qualisys': {'binning_factor': 1},
                    'optitrack': {},
                    'vicon': {},
                    'opencap': {},
                    'easymocap': {},
                    'biocv': {},
                    'anipose': {},
                    'freemocap': {}
                    },
                    'calculate': 
                    {
                        'intrinsics': 
                        {
                            'overwrite_intrinsics': False,
                            'show_detection_intrinsics': True,
                            'intrinsics_extension': 'jpg',
                            'extract_every_N_sec': 1,
                            'intrinsics_corners_nb': [4, 7],
                            'intrinsics_square_size': 60
                        },
                        'extrinsics': 
                        {
                            'calculate_extrinsics': True,
                            'extrinsics_method': 'scene',
                            'moving_cameras': False,
                            'board': 
                            {
                                'show_reprojection_error': True,
                                'extrinsics_extension': 'png',
                                'board_position': 'vertical',
                                'extrinsics_corners_nb': [4, 7],
                                'extrinsics_square_size': 60
                            },
                            'scene': 
                            {   
                                'show_reprojection_error': True,
                                'extrinsics_extension': 'png',
                                'object_coords_3d': [[-2.0, 0.3, 0.0],
                                [-2.0, 0.0, 0.0],
                                [-2.0, 0.0, 0.05],
                                [-2.0, -0.3, 0.0],
                                [0.0, 0.3, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.05],
                                [0.0, -0.3, 0.0]]
                            },
                            'keypoints': {}
                        }
                    }
                },
                'personAssociation': 
                {
                    'likelihood_threshold_association': 0.3,
                    'single_person': 
                    {
                        'likelihood_threshold_association': 0.3,
                    'reproj_error_threshold_association': 20,
                    'tracked_keypoint': 'Neck'
                    },
                    'multi_person': 
                    {
                        'reconstruction_error_threshold': 0.1,
                        'min_affinity': 0.2
                    }
                },
                'triangulation': 
                {
                    'reproj_error_threshold_triangulation': 15,
                    'likelihood_threshold_triangulation': 0.3,
                    'min_cameras_for_triangulation': 2,
                    'interp_if_gap_smaller_than': 20,
                    'max_distance_m': 1.0,
                    'interpolation': 'linear',
                    'remove_incomplete_frames': False,
                    'sections_to_keep': 'all',
                    'min_chunk_size': 10,
                    'fill_large_gaps_with': 'last_value',
                    'show_interp_indices': True,
                    'make_c3d': True
                },
                'filtering': 
                {
                    'reject_outliers': True,
                    'filter': True,
                    'type': 'butterworth',
                    'display_figures': True,
                    'save_filt_plots': True,
                    'make_c3d': True,
                    'butterworth': {'cut_off_frequency': 6, 'order': 4},
                    'kalman': {'trust_ratio': 500, 'smooth': True},
                    'gcv_spline': {'cut_off_frequency': 'auto', 'smoothing_factor': 1.0},
                    'loess': {'nb_values_used': 5},
                    'gaussian': {'sigma_kernel': 1},
                    'median': {'kernel_size': 3},
                    'butterworth_on_speed': {'order': 4, 'cut_off_frequency': 10}
                },
                'markerAugmentation': 
                {
                    'feet_on_floor': False, 
                    'make_c3d': True
                },
                'kinematics': 
                {
                    'use_augmentation': True,
                    'use_simple_model': False,
                    'right_left_symmetry': True,
                    'default_height': 1.7,
                    'remove_individual_scaling_setup': True,
                    'remove_individual_ik_setup': True,
                    'fastest_frames_to_remove_percent': 0.1,
                    'close_to_zero_speed_m': 0.2,
                    'large_hip_knee_angles': 45,
                    'trimmed_extrema_percent': 0.5
                },
                'logging': {'use_custom_logging': False}
            }
    
    def get_default_2d_structure(self):
        """Default 2D config structure if Sports2D not installed"""
        return  {
                'base': # CRITICAL: Use 'base' not 'project' for 2D
                {
                    'video_input': 'demo.mp4',
                    'nb_persons_to_detect': 'all',
                    'person_ordering_method': 'on_click',
                    'first_person_height': 1.65,
                    'visible_side': ['auto', 'front', 'none'],
                    'load_trc_px': '',
                    'compare': False,
                    'time_range': [],
                    'video_dir': '',
                    'webcam_id': 0,
                    'input_size': [1280, 720],
                    'show_realtime_results': True,
                    'save_vid': True,
                    'save_img': True,
                    'save_pose': True,
                    'calculate_angles': True,
                    'save_angles': True,
                    'result_dir': ''
                },
                'pose': 
                {
                    'slowmo_factor': 1,
                    'pose_model': 'Body_with_feet',
                    'mode': 'balanced',
                    'det_frequency': 4,
                    'device': 'auto',
                    'backend': 'auto',
                    'tracking_mode': 'sports2d',
                    'keypoint_likelihood_threshold': 0.3,
                    'average_likelihood_threshold': 0.5,
                    'keypoint_number_threshold': 0.3,
                    'max_distance': 250,
                    'CUSTOM': 
                    {
                        'name': 'Hip',
                        'id': 19,
                        'children': [{'name': 'RHip',
                            'id': 12,
                            'children': [{'name': 'RKnee',
                            'id': 14,
                            'children': [{'name': 'RAnkle',
                                'id': 16,
                                'children': [{'name': 'RBigToe',
                                'id': 21,
                                'children': [{'name': 'RSmallToe', 'id': 23}]},
                                {'name': 'RHeel', 'id': 25}]}]}]},
                            {'name': 'LHip',
                            'id': 11,
                            'children': [{'name': 'LKnee',
                            'id': 13,
                            'children': [{'name': 'LAnkle',
                                'id': 15,
                                'children': [{'name': 'LBigToe',
                                'id': 20,
                                'children': [{'name': 'LSmallToe', 'id': 22}]},
                                {'name': 'LHeel', 'id': 24}]}]}]},
                            {'name': 'Neck',
                            'id': 18,
                            'children': [{'name': 'Head',
                            'id': 17,
                            'children': [{'name': 'Nose', 'id': 0}]},
                            {'name': 'RShoulder',
                            'id': 6,
                            'children': [{'name': 'RElbow',
                                'id': 8,
                                'children': [{'name': 'RWrist', 'id': 10}]}]},
                            {'name': 'LShoulder',
                            'id': 5,
                            'children': [{'name': 'LElbow',
                                'id': 7,
                                'children': [{'name': 'LWrist', 'id': 9}]}]}]}]}
                    },
                    'px_to_meters_conversion': 
                    {
                        'to_meters': True,
                        'make_c3d': True,
                        'save_calib': True,
                        'floor_angle': 'auto',
                        'xy_origin': ['auto'],
                        'calib_file': ''
                    },
                    'angles': 
                    {
                        'display_angle_values_on': ['body', 'list'],
                        'fontSize': 0.3,
                        'joint_angles': ['Right ankle', 'Left ankle','Right knee', 'Left knee', 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder', 'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist'],
                        'segment_angles': ['Right foot', 'Left foot', 'Right shank', 'Left shank', 'Right thigh', 'Left thigh', 'Pelvis', 'Trunk', 'Shoulders', 'Head', 'Right arm', 'Left arm', 'Right forearm', 'Left forearm'],
                        'flip_left_right': True,
                        'correct_segment_angles_with_floor_angle': True
                    },
                    'post-processing': 
                    {
                        'interpolate': True,
                        'interp_gap_smaller_than': 10,
                        'fill_large_gaps_with': 'last_value',
                        'sections_to_keep': 'all',
                        'min_chunk_size': 10,
                        'reject_outliers': True,
                        'filter': True,
                        'show_graphs': True,
                        'save_graphs': True,
                        'filter_type': 'butterworth',
                        'butterworth': {'cut_off_frequency': 6, 'order': 4},
                        'kalman': {'trust_ratio': 500, 'smooth': True},
                        'gcv_spline': {'gcv_cut_off_frequency': 'auto', 'gcv_smoothing_factor': 0.1},
                        'loess': {'nb_values_used': 5},
                        'gaussian': {'sigma_kernel': 1},
                        'median': {'kernel_size': 3},
                        'butterworth_on_speed': {'order': 4, 'cut_off_frequency': 10}
                    },
                    'kinematics': 
                    {
                        'do_ik': False,
                        'use_augmentation': False,
                        'feet_on_floor': False,
                        'use_simple_model': False,
                        'participant_mass': [55.0, 67.0],
                        'right_left_symmetry': True,
                        'default_height': 1.7,
                        'fastest_frames_to_remove_percent': 0.1,
                        'close_to_zero_speed_px': 50,
                        'close_to_zero_speed_m': 0.2,
                        'large_hip_knee_angles': 45,
                        'trimmed_extrema_percent': 0.5,
                        'remove_individual_scaling_setup': True,
                        'remove_individual_ik_setup': True
                    },
                    'logging': 
                    {
                        'use_custom_logging': False
                    }
                    }
    
    def generate_2d_config(self, config_path, settings):
        """Generate configuration file for 2D analysis"""
        try:
            # Load the template
            config = toml.load(self.config_2d_template_path)
            
            # Debug print to check settings
            print("=" * 60)
            print("2D Settings being applied:")
            print(settings)
            print("=" * 60)
            
            # CRITICAL FIX: Update sections recursively - this will overwrite template values
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                # Force update - don't preserve template defaults
                self.update_nested_section(config[section_name], section_data, force_overwrite=True)
            
            # Debug print final config
            print("=" * 60)
            print("Final 2D Config:")
            print(config)
            print("=" * 60)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            
            print(f"2D Config file saved successfully to {config_path}")
            return True
        except Exception as e:
            print(f"Error generating 2D config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_3d_config(self, config_path, settings):
        """Generate configuration file for 3D analysis"""
        try:
            # Parse the template
            config = toml.load(self.config_3d_template_path)
            
            # Debug print to check settings
            print("=" * 60)
            print("3D Settings being applied:")
            print(settings)
            print("=" * 60)
            
            # CRITICAL force overwrite template values
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                # Force update - don't preserve template defaults
                self.update_nested_section(config[section_name], section_data, force_overwrite=True)
            
            # Debug print final config
            print("=" * 60)
            print("Final 3D Config:")
            print(config)
            print("=" * 60)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            
            print(f"3D Config file saved successfully to {config_path}")
            return True
        except Exception as e:
            print(f"Error generating 3D config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_nested_section(self, config_section, settings_section, force_overwrite=False):
        """
        Recursively update nested sections of the configuration file.
        
        Args:
            config_section: The config dictionary section to update
            settings_section: The settings dictionary section with new values
            force_overwrite: If True, always overwrite config values with settings values
        """
        if not isinstance(settings_section, dict):
            return
        
        for key, value in settings_section.items():
            if isinstance(value, dict):
                # If the key doesn't exist in the config section, create it
                if key not in config_section:
                    config_section[key] = {}
                
                # Recursively update the subsection
                self.update_nested_section(config_section[key], value, force_overwrite)
            else:
                # CRITICAL FIX: Always update the value (overwrite template defaults)
                config_section[key] = value