import os
import tomlkit

class ConfigGenerator:
    def __init__(self):
        # Load templates from files to avoid comment issues
        self.config_3d_template_path = os.path.join('templates', '3d_config_template.toml')
        self.config_2d_template_path = os.path.join('templates', '2d_config_template.toml')
        
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        # Write the template files if they don't exist
        self.create_template_files()
    
    def create_template_files(self):
        """Create template files if they don't exist"""
        # Create 3D template file
        if not os.path.exists(self.config_3d_template_path):
            with open(self.config_3d_template_path, 'w', encoding='utf-8') as f:
                f.write(self.get_3d_template())
        
        # Create 2D template file
        if not os.path.exists(self.config_2d_template_path):
            with open(self.config_2d_template_path, 'w', encoding='utf-8') as f:
                f.write(self.get_2d_template())
    
    def get_3d_template(self):
        """Return the 3D configuration template"""
        return '''###############################################################################
## PROJECT PARAMETERS                                                        ##
###############################################################################

[project]
multi_person = true
participant_height = 'auto' 
participant_mass = 70.0

frame_rate = 'auto'
frame_range = []
exclude_from_batch = []

[pose]
vid_img_extension = 'avi'
pose_model = 'Body_with_feet'
mode = 'balanced'
det_frequency = 4
device = 'auto'
backend = 'auto'
tracking_mode = 'sports2d'
deepsort_params = """{'max_age':30, 'n_init':3, 'nms_max_overlap':0.8, 'max_cosine_distance':0.3, 'nn_budget':200, 'max_iou_distance':0.8}"""
display_detection = true
overwrite_pose = true
save_video = 'none'
output_format = 'openpose'

[synchronization]
display_sync_plots = true
keypoints_to_consider = 'all'
approx_time_maxspeed = 'auto'
time_range_around_maxspeed = 2.0
likelihood_threshold = 0.4
filter_cutoff = 6
filter_order = 4

[calibration]
calibration_type = 'convert'

   [calibration.convert]
   convert_from = 'qualisys'
      [calibration.convert.caliscope]
      [calibration.convert.qualisys]
      binning_factor = 1
      [calibration.convert.optitrack]
      [calibration.convert.vicon]
      [calibration.convert.opencap]
      [calibration.convert.easymocap]
      [calibration.convert.biocv]
      [calibration.convert.anipose]
      [calibration.convert.freemocap]

   [calibration.calculate] 
      [calibration.calculate.intrinsics]
      overwrite_intrinsics = false
      show_detection_intrinsics = true
      intrinsics_extension = 'png'
      extract_every_N_sec = 1
      intrinsics_corners_nb = [3,5] 
      intrinsics_square_size = 34

      [calibration.calculate.extrinsics]
      calculate_extrinsics = true
      extrinsics_method = 'scene'
      moving_cameras = false

         [calibration.calculate.extrinsics.board]
         show_reprojection_error = true
         extrinsics_extension = 'mp4'
         extrinsics_corners_nb = [4,7]
         extrinsics_square_size = 60

         [calibration.calculate.extrinsics.scene]
         show_reprojection_error = true
         extrinsics_extension = 'mp4'
         object_coords_3d = [[0.0, 0.0, 0.0], [-0.50, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.5, 0.0, 0.0], [0.00, 0.50, 0.0], [-0.50, 0.50, 0.0], [-1.0, 0.50, 0.0], [-1.50, 0.50, 0.0]]
        
         [calibration.calculate.extrinsics.keypoints]

[personAssociation]
   likelihood_threshold_association = 0.3

   [personAssociation.single_person]
   reproj_error_threshold_association = 20
   tracked_keypoint = 'Neck'
   
   [personAssociation.multi_person]
   reconstruction_error_threshold = 0.1
   min_affinity = 0.2

[triangulation]
reproj_error_threshold_triangulation = 15
likelihood_threshold_triangulation= 0.3
min_cameras_for_triangulation = 2
interpolation = 'linear'
interp_if_gap_smaller_than = 10
fill_large_gaps_with = 'last_value'
show_interp_indices = true
handle_LR_swap = false
undistort_points = false
make_c3d = true

[filtering]
type = 'butterworth'
display_figures = true
make_c3d = true

   [filtering.butterworth]
   order = 4 
   cut_off_frequency = 6
   [filtering.kalman]
   trust_ratio = 100
   smooth = true
   [filtering.butterworth_on_speed]
   order = 4 
   cut_off_frequency = 10
   [filtering.gaussian]
   sigma_kernel = 2
   [filtering.LOESS]
   nb_values_used = 30
   [filtering.median]
   kernel_size = 9

[markerAugmentation]
make_c3d = true

[kinematics]
use_augmentation = true
use_contacts_muscles = true
right_left_symmetry = true
default_height = 1.7
remove_individual_scaling_setup = true
remove_individual_IK_setup = true
fastest_frames_to_remove_percent = 0.1
close_to_zero_speed_m = 0.2
large_hip_knee_angles = 45
trimmed_extrema_percent = 0.5

[logging]
use_custom_logging = false

[pose.CUSTOM]
name = "Hip"
id = 19
  [[pose.CUSTOM.children]]
  name = "RHip"
  id = 12
     [[pose.CUSTOM.children.children]]
     name = "RKnee"
     id = 14
        [[pose.CUSTOM.children.children.children]]
        name = "RAnkle"
        id = 16
           [[pose.CUSTOM.children.children.children.children]]
           name = "RBigToe"
           id = 21
              [[pose.CUSTOM.children.children.children.children.children]]
              name = "RSmallToe"
              id = 23
           [[pose.CUSTOM.children.children.children.children]]
           name = "RHeel"
           id = 25
  [[pose.CUSTOM.children]]
  name = "LHip"
  id = 11
     [[pose.CUSTOM.children.children]]
     name = "LKnee"
     id = 13
        [[pose.CUSTOM.children.children.children]]
        name = "LAnkle"
        id = 15
           [[pose.CUSTOM.children.children.children.children]]
           name = "LBigToe"
           id = 20
              [[pose.CUSTOM.children.children.children.children.children]]
              name = "LSmallToe"
              id = 22
           [[pose.CUSTOM.children.children.children.children]]
           name = "LHeel"
           id = 24
  [[pose.CUSTOM.children]]
  name = "Neck"
  id = 18
     [[pose.CUSTOM.children.children]]
     name = "Head"
     id = 17
        [[pose.CUSTOM.children.children.children]]
        name = "Nose"
        id = 0
     [[pose.CUSTOM.children.children]]
     name = "RShoulder"
     id = 6
        [[pose.CUSTOM.children.children.children]]
        name = "RElbow"
        id = 8
           [[pose.CUSTOM.children.children.children.children]]
           name = "RWrist"
           id = 10
     [[pose.CUSTOM.children.children]]
     name = "LShoulder"
     id = 5
        [[pose.CUSTOM.children.children.children]]
        name = "LElbow"
        id = 7
           [[pose.CUSTOM.children.children.children.children]]
           name = "LWrist"
           id = 9'''
    
    def get_2d_template(self):
        """Return the 2D configuration template"""
        return '''###############################################################################
## SPORTS2D PROJECT PARAMETERS                                               ##
###############################################################################

[project]
video_input = 'cam2.mp4'
px_to_m_from_person_id = 0
px_to_m_person_height = 1.75
visible_side = ['auto']
load_trc_px = ''
compare = false
time_range = []
video_dir = ''
webcam_id = 0
input_size = [1280, 720]

[process]
multiperson = true
show_realtime_results = true
save_vid = true
save_img = false
save_pose = true
calculate_angles = true
save_angles = true
result_dir = ''

[pose]
slowmo_factor = 1
pose_model = 'Body_with_feet'
mode = 'balanced'
det_frequency = 4
device = 'auto'
backend = 'auto'
tracking_mode = 'sports2d'
keypoint_likelihood_threshold = 0.3
average_likelihood_threshold = 0.5
keypoint_number_threshold = 0.3

[px_to_meters_conversion]
to_meters = true
make_c3d = true
save_calib = true
floor_angle = 'auto'
xy_origin = ['auto']
calib_file = ''

[angles]
display_angle_values_on = ['body', 'list']
fontSize = 0.3
joint_angles = ['Right ankle', 'Left ankle', 'Right knee', 'Left knee', 'Right hip', 'Left hip', 'Right shoulder', 'Left shoulder', 'Right elbow', 'Left elbow', 'Right wrist', 'Left wrist']
segment_angles = ['Right foot', 'Left foot', 'Right shank', 'Left shank', 'Right thigh', 'Left thigh', 'Pelvis', 'Trunk', 'Shoulders', 'Head', 'Right arm', 'Left arm', 'Right forearm', 'Left forearm']
flip_left_right = true
correct_segment_angles_with_floor_angle = true

[post-processing]
interpolate = true
interp_gap_smaller_than = 10
fill_large_gaps_with = 'last_value'
filter = true
show_graphs = true
filter_type = 'butterworth'
   [post-processing.butterworth]
   order = 4 
   cut_off_frequency = 6
   [post-processing.gaussian]
   sigma_kernel = 1
   [post-processing.loess]
   nb_values_used = 5
   [post-processing.median]
   kernel_size = 3

[kinematics]
do_ik = true
use_augmentation = true
use_contacts_muscles = true
participant_mass = [67.0, 55.0]
right_left_symmetry = true
default_height = 1.7
fastest_frames_to_remove_percent = 0.1
close_to_zero_speed_px = 50
close_to_zero_speed_m = 0.2
large_hip_knee_angles = 45
trimmed_extrema_percent = 0.5
remove_individual_scaling_setup = true
remove_individual_ik_setup = true

[logging]
use_custom_logging = false

[pose.CUSTOM]
name = "Hip"
id = 19
  [[pose.CUSTOM.children]]
  name = "RHip"
  id = 12
     [[pose.CUSTOM.children.children]]
     name = "RKnee"
     id = 14
        [[pose.CUSTOM.children.children.children]]
        name = "RAnkle"
        id = 16
           [[pose.CUSTOM.children.children.children.children]]
           name = "RBigToe"
           id = 21
              [[pose.CUSTOM.children.children.children.children.children]]
              name = "RSmallToe"
              id = 23
           [[pose.CUSTOM.children.children.children.children]]
           name = "RHeel"
           id = 25
  [[pose.CUSTOM.children]]
  name = "LHip"
  id = 11
     [[pose.CUSTOM.children.children]]
     name = "LKnee"
     id = 13
        [[pose.CUSTOM.children.children.children]]
        name = "LAnkle"
        id = 15
           [[pose.CUSTOM.children.children.children.children]]
           name = "LBigToe"
           id = 20
              [[pose.CUSTOM.children.children.children.children.children]]
              name = "LSmallToe"
              id = 22
           [[pose.CUSTOM.children.children.children.children]]
           name = "LHeel"
           id = 24
  [[pose.CUSTOM.children]]
  name = "Neck"
  id = 18
     [[pose.CUSTOM.children.children]]
     name = "Head"
     id = 17
        [[pose.CUSTOM.children.children.children]]
        name = "Nose"
        id = 0
     [[pose.CUSTOM.children.children]]
     name = "RShoulder"
     id = 6
        [[pose.CUSTOM.children.children.children]]
        name = "RElbow"
        id = 8
           [[pose.CUSTOM.children.children.children.children]]
           name = "RWrist"
           id = 10
     [[pose.CUSTOM.children.children]]
     name = "LShoulder"
     id = 5
        [[pose.CUSTOM.children.children.children]]
        name = "LElbow"
        id = 7
           [[pose.CUSTOM.children.children.children.children]]
           name = "LWrist"
           id = 9'''
    
    def generate_2d_config(self, config_path, settings):
        """Generate configuration file for 2D analysis"""
        try:
            # Load the template
            with open(self.config_2d_template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Parse the template
            config = tomlkit.parse(template)
            
            # Debug print to check settings
            print("2D Settings being applied:", settings)
            
            # Update sections recursively
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                self.update_nested_section(config[section_name], section_data)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(tomlkit.dumps(config))
            
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
            # Load the template
            with open(self.config_3d_template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Parse the template
            config = tomlkit.parse(template)
            
            # Debug print to check settings
            print("3D Settings being applied:", settings)
            
            # Update sections recursively
            for section_name, section_data in settings.items():
                if section_name not in config:
                    config[section_name] = {}
                
                self.update_nested_section(config[section_name], section_data)
            
            # Write the updated config with pretty formatting
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(tomlkit.dumps(config))
            
            print(f"3D Config file saved successfully to {config_path}")
            return True
        except Exception as e:
            print(f"Error generating 3D config: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_nested_section(self, config_section, settings_section):
        """Recursively update nested sections of the configuration file"""
        if not isinstance(settings_section, dict):
            return
        
        for key, value in settings_section.items():
            if isinstance(value, dict):
                # If the key doesn't exist in the config section, create it
                if key not in config_section:
                    config_section[key] = {}
                
                # Recursively update the subsection
                self.update_nested_section(config_section[key], value)
            else:
                # Update the value
                config_section[key] = value