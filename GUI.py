# This is a beta version of the GUI, improvements should be made as soon as possible.
import os
import shutil
import threading
import tkinter as tk
import customtkinter as ctk  
from tkinter import messagebox, filedialog, simpledialog, ttk  # Import ttk for Notebook
import tomlkit
from tomlkit import parse, dumps
import numpy as np
import cv2
from PIL import Image
from customtkinter import CTkImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ast

class App:
    def __init__(self, root):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("System")  # Modes: "System" (default), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

        self.root = root
        self.root.title("Pose2Sim Configuration")
        self.root.geometry("1200x900")
        self.change_intrinsics_extension = False
        self.language = None  # Will hold 'en' for English or 'fr'
        self.process_mode = None  # 'single' or 'batch'
        self.num_trials = 0  # Number of trials in batch mode
        self.config_template= r"""###############################################################################
## PROJECT PARAMETERS                                                        ##
###############################################################################


# Configure your project parameters here. 
# 
# IMPORTANT:
# If a parameter is not found here, Pose2Sim will look for its value in the 
# Config.toml file of the level above. This way, you can set global 
# instructions for the Session and alter them for specific Participants or Trials.
#
# If you wish to overwrite a parameter for a specific trial or participant,  
# edit its Config.toml file by uncommenting its key (e.g., [project])
# and editing its value (e.g., frame_range = [10,300]). Or else, uncomment 
# [filtering.butterworth] and set cut_off_frequency = 10, etc.



[project]
multi_person = false # true for trials with multiple participants. If false, only the main person in scene is analyzed (and it run much faster). 
participant_height = 1.72 # m # float if single person, list of float if multi-person (same order as the Static trials) # Only used for marker augmentation
participant_mass = 70.0 # kg # Only used for marker augmentation and scaling

frame_rate = 'auto' # fps # int or 'auto'. If 'auto', finds from video (or defaults to 60 fps if you work with images) 
frame_range = [] # For example [10,300], or [] for all frames. 
## If cameras are not synchronized, designates the frame range of the camera with the shortest recording time
## N.B.: If you want a time range instead, use frame_range = time_range * frame_rate
## For example if you want to analyze from 0.1 to 2 seconds with a 60 fps frame rate, 
## frame_range = [0.1, 2.0]*frame_rate = [6, 120]

exclude_from_batch = [] # List of trials to be excluded from batch analysis, ['<participant_dir/trial_dir>', 'etc'].
# e.g. ['S00_P00_Participant/S00_P00_T00_StaticTrial', 'S00_P00_Participant/S00_P00_T01_BalancingTrial']

[pose]
vid_img_extension = 'mp4' # any video or image extension
pose_model = 'HALPE_26'  #With RTMLib: HALPE_26 (body and feet, default), COCO_133 (body, feet, hands), COCO_17 (body)
                         # /!\ Only RTMPose is natively embeded in Pose2Sim. For all other pose estimation methods, you will have to run them yourself, and then refer to the documentation to convert the files if needed
                         #With MMPose: HALPE_26, COCO_133, COCO_17, CUSTOM. See CUSTOM example at the end of the file
                         #With openpose: BODY_25B, BODY_25, BODY_135, COCO, MPII
                         #With mediapipe: BLAZEPOSE
                         #With alphapose: HALPE_26, HALPE_68, HALPE_136, COCO_133
                         #With deeplabcut: CUSTOM. See example at the end of the file
mode = 'balanced' # 'lightweight', 'balanced', 'performance'
det_frequency = 1 # Run person detection only every N frames, and inbetween track previously detected bounding boxes (keypoint detection is still run on all frames). 
                  # Equal to or greater than 1, can be as high as you want in simple uncrowded cases. Much faster, but might be less accurate. 
display_detection = true
overwrite_pose = false # set to false if you don't want to recalculate pose estimation when it has already been done
save_video = 'to_video' # 'to_video' or 'to_images', 'none', or ['to_video', 'to_images']
output_format = 'openpose' # 'openpose', 'mmpose', 'deeplabcut', 'none' or a list of them # /!\ only 'openpose' is supported for now


[synchronization]
display_sync_plots = true # true or false (lowercase)
keypoints_to_consider = ['RWrist'] # 'all' if all points should be considered, for example if the participant did not perform any particicular sharp movement. In this case, the capture needs to be 5-10 seconds long at least
                           # ['RWrist', 'RElbow'] list of keypoint names if you want to specify keypoints with a sharp vertical motion.
approx_time_maxspeed = 'auto' # 'auto' if you want to consider the whole capture (default, slower if long sequences)
                           # [10.0, 2.0, 8.0, 11.0] list of times (seconds) if you want to specify the approximate time of a clear vertical event for each camera
time_range_around_maxspeed = 2.0 # Search for best correlation in the range [approx_time_maxspeed - time_range_around_maxspeed, approx_time_maxspeed  + time_range_around_maxspeed]
likelihood_threshold = 0.4 # Keypoints whose likelihood is below likelihood_threshold are filtered out
filter_cutoff = 6 # time series are smoothed to get coherent time-lagged correlation
filter_order = 4


# Take heart, calibration is not that complicated once you get the hang of it!
[calibration]
calibration_type = 'convert' # 'convert' or 'calculate'

   [calibration.convert]
   convert_from = 'qualisys' # 'caliscope', 'qualisys', 'optitrack', vicon', 'opencap', 'easymocap', 'biocv', 'anipose', or 'freemocap'
      [calibration.convert.caliscope]  # No parameter needed
      [calibration.convert.qualisys]
      binning_factor = 1 # Usually 1, except when filming in 540p where it usually is 2
      [calibration.convert.optitrack]  # See readme for instructions
      [calibration.convert.vicon]      # No parameter needed
      [calibration.convert.opencap]    # No parameter needed
      [calibration.convert.easymocap]  # No parameter needed
      [calibration.convert.biocv]      # No parameter needed
      [calibration.convert.anipose]    # No parameter needed
      [calibration.convert.freemocap]  # No parameter needed
  

   [calibration.calculate] 
      # Camera properties, theoretically need to be calculated only once in a camera lifetime
      [calibration.calculate.intrinsics]
      overwrite_intrinsics = false # set to false if you don't want to recalculate intrinsic parameters
      show_detection_intrinsics = true # true or false (lowercase)
      intrinsics_extension = 'jpg' # any video or image extension
      extract_every_N_sec = 1 # if video, extract frames every N seconds (can be <1 )
      intrinsics_corners_nb = [4,7] 
      intrinsics_square_size = 60 # mm

      # Camera placements, need to be done before every session
      [calibration.calculate.extrinsics]
      calculate_extrinsics = true # true or false (lowercase) 
      extrinsics_method = 'scene' # 'board', 'scene', 'keypoints'
      # 'board' should be large enough to be detected when laid on the floor. Not recommended.
      # 'scene' involves manually clicking any point of know coordinates on scene. Usually more accurate if points are spread out.
      # 'keypoints' uses automatic pose estimation of a person freely walking and waving arms in the scene. Slighlty less accurate, requires synchronized cameras.
      moving_cameras = false # Not implemented yet

         [calibration.calculate.extrinsics.board]
         show_reprojection_error = true # true or false (lowercase)
         extrinsics_extension = 'png' # any video or image extension
         extrinsics_corners_nb = [4,7] # [H,W] rather than [w,h]
         extrinsics_square_size = 60 # mm # [h,w] if square is actually a rectangle

         [calibration.calculate.extrinsics.scene]
         show_reprojection_error = true # true or false (lowercase)
         extrinsics_extension = 'png' # any video or image extension
         # list of 3D coordinates to be manually labelled on images. Can also be a 2 dimensional plane. 
         # in m -> unlike for intrinsics, NOT in mm!
         object_coords_3d =   [[-2.0,  0.3,  0.0], 
                              [-2.0 , 0.0,  0.0], 
                              [-2.0, 0.0,  0.05], 
                              [-2.0, -0.3 ,  0.0], 
                              [0.0,  0.3,  0.0], 
                              [0.0, 0.0,  0.0], 
                              [0.0, 0.0,  0.05], 
                              [0.0, -0.3,  0.0]]
        
         [calibration.calculate.extrinsics.keypoints]
         # Coming soon!


[personAssociation]
   likelihood_threshold_association = 0.3

   [personAssociation.single_person]
   reproj_error_threshold_association = 20 # px
   tracked_keypoint = 'Neck' # If the neck is not detected by the pose_model, check skeleton.py 
               # and choose a stable point for tracking the person of interest (e.g., 'right_shoulder' or 'RShoulder')
   
   [personAssociation.multi_person]
   reconstruction_error_threshold = 0.1 # 0.1 = 10 cm
   min_affinity = 0.2 # affinity below which a correspondence is ignored


[triangulation]
reproj_error_threshold_triangulation = 15 # px
likelihood_threshold_triangulation= 0.3
min_cameras_for_triangulation = 2
interpolation = 'linear' #linear, slinear, quadratic, cubic, or none
                        # 'none' if you don't want to interpolate missing points
interp_if_gap_smaller_than = 10 # do not interpolate bigger gaps
show_interp_indices = true # true or false (lowercase). For each keypoint, return the frames that need to be interpolated
fill_large_gaps_with = 'last_value' # 'last_value', 'nan', or 'zeros' 
handle_LR_swap = false # Better if few cameras (eg less than 4) with risk of limb swapping (eg camera facing sagittal plane), otherwise slightly less accurate and slower
undistort_points = false # Better if distorted image (parallel lines curvy on the edge or at least one param > 10^-2), but unnecessary (and slightly slower) if distortions are low
make_c3d = true # save triangulated data in c3d format in addition to trc


[filtering]
type = 'butterworth' # butterworth, kalman, gaussian, LOESS, median, butterworth_on_speed
display_figures = true # true or false (lowercase) 
make_c3d = true # also save triangulated data in c3d format

   [filtering.butterworth]
   order = 4 
   cut_off_frequency = 6 # Hz
   [filtering.kalman]
   # How much more do you trust triangulation results (measurements), than previous data (process assuming constant acceleration)?
   trust_ratio = 100 # = measurement_trust/process_trust ~= process_noise/measurement_noise
   smooth = true # should be true, unless you need real-time filtering
   [filtering.butterworth_on_speed]
   order = 4 
   cut_off_frequency = 10 # Hz
   [filtering.gaussian]
   sigma_kernel = 2 #px
   [filtering.LOESS]
   nb_values_used = 30 # = fraction of data used * nb frames
   [filtering.median]
   kernel_size = 9


[markerAugmentation] 
## Requires the following markers: ["Neck", "RShoulder", "LShoulder", "RHip", "LHip", "RKnee", "LKnee",
##        "RAnkle", "LAnkle", "RHeel", "LHeel", "RSmallToe", "LSmallToe",
##        "RBigToe", "LBigToe", "RElbow", "LElbow", "RWrist", "LWrist"]
make_c3d = true # save triangulated data in c3d format in addition to trc


[kinematics]
use_augmentation = true  # true or false (lowercase) # Set to true if you want to use the model with augmented markers
right_left_symmetry = true # true or false (lowercase) # Set to false only if you have good reasons to think the participant is not symmetrical (e.g. prosthetic limb)
remove_individual_scaling_setup = true # true or false (lowercase) # If true, the individual scaling setup files are removed to avoid cluttering
remove_individual_IK_setup = true # true or false (lowercase) # If true, the individual IK setup files are removed to avoid cluttering



# CUSTOM skeleton, if you trained your own model from DeepLabCut or MMPose for example. 
# Make sure the node ids correspond to the column numbers of the 2D pose file, starting from zero.
# 
# If you want to perform inverse kinematics, you will also need to create an OpenSim model
# and add to its markerset the location where you expect the triangulated keypoints to be detected.
# 
# In this example, CUSTOM reproduces the HALPE_26 skeleton (default skeletons are stored in skeletons.py).
# You can create as many custom skeletons as you want, just add them further down and rename them.
# 
# Check your model hierarchy with:  for pre, _, node in RenderTree(model): 
#                                      print(f'{pre}{node.name} id={node.id}')
[pose.CUSTOM]
name = "Hip"
id = "19"
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
           id = 9 """

        # Initialize language selection
        self.select_language()
        
        # Define the order of tabs
        self.tab_order = [
            'calibration_config',
            'prepare_video',
            'pose_model',
            'synchronization',
            'activation',
            'advanced_configuration'
                    ]

        # Initialize tab names with red indicators (unsaved)
        self.tab_names = {
            'calibration_config': '🔴 Calibration Configuration',
            'prepare_video': '🔴 Prepare Video',
            'pose_model': '🔴 Pose Estimation',
            'synchronization': '🔴 Synchronization',
            'activation': '🔴 Activation',
            'advanced_configuration': '🔴 Advanced Configuration',
            
        }
        if self.process_mode == 'batch':
            self.tab_order.append('batch_configuration')
            self.tab_names['batch_configuration'] = '🔴 Batch Configuration'
        
        # Initialize dictionaries to hold content and border frames
        self.tab_frames = {}
        self.tab_border_frames = {}
   
        # Initialize progress tracking
        self.progress_var = ctk.DoubleVar()
        self.progress_var.set(0)
        self.progress_steps = {
            'calibration': 25,
            'prepare_video': 50,
            'pose': 75,
            'synchronization': 100
        }
        # Create progress bar
        self.progress_frame = ctk.CTkFrame(self.root)
        self.progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill='x', padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(self.progress_frame, text="Overall Progress: 0%")
        self.progress_label.pack(pady=2)
        

        self.current_camera_index = 0
        self.camera_image_list = []
        self.image_vars = []

    def select_language(self):
        # Create a frame for language selection
        lang_frame = ctk.CTkFrame(self.root)
        lang_frame.pack(expand=True, fill='both')

        def set_language(lang):
            self.language = lang
            # Prompt the user to enter the participant name
            participant_name = simpledialog.askstring(
                title="Participant Name",
                prompt="Please enter the participant name:"
            )
            if not participant_name:
                # If no name is entered, set a default name
                participant_name = 'Participant'
            self.participant_name = participant_name.strip()
            
            # Create the participant folder structure based on process mode
            self.choose_process_mode(lang_frame)

        # Language selection buttons
        ctk.CTkLabel(lang_frame, text="Select Language / Choisir la langue",
                     font=('Helvetica', 20, 'bold')).pack(pady=40)
        ctk.CTkButton(lang_frame, text="English", command=lambda: set_language('en'),
                      width=200, height=50, font=('Helvetica', 18)).pack(pady=20)
        ctk.CTkButton(lang_frame, text="Français (coming soon)", command=lambda: set_language('fr'),
                      width=200, height=50, font=('Helvetica', 18)).pack(pady=20)

    def choose_process_mode(self, lang_frame):
        # Clear language selection frame
        lang_frame.destroy()

        # Create a new frame for process mode selection
        mode_frame = ctk.CTkFrame(self.root)
        mode_frame.pack(expand=True, fill='both')

        def set_mode(mode):
            self.process_mode = mode
            mode_frame.destroy()
            if self.process_mode == 'batch':
                self.tab_order.append('batch_configuration')
                self.tab_names['batch_configuration'] = '🔴 Batch Configuration'
            self.initial_config_check()  # Start with the initial configuration check

        # Process mode selection buttons
        ctk.CTkLabel(mode_frame, text="Select Process Mode / Sélectionnez le mode de traitement",
                     font=('Helvetica', 20, 'bold')).pack(pady=40)
        ctk.CTkButton(mode_frame, text="Single", command=lambda: set_mode('single'),
                      width=200, height=50, font=('Helvetica', 18)).pack(pady=20)
        ctk.CTkButton(mode_frame, text="Batch", command=lambda: set_mode('batch'),
                      width=200, height=50, font=('Helvetica', 18)).pack(pady=20)

    def initial_config_check(self):
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        if self.process_mode == 'single':
            # Create single process folder structure
            self.create_single_folder_structure()
            if os.path.exists(config_file_path):
                response = messagebox.askyesno("Configuration Exists",
                                               "Configuration already exists, would you like to configure Pose2Sim again?")
                if response:
                    self.init_app()  # Proceed with reconfiguration
                else:
                    # Proceed to Pose Estimation tab
                    self.init_app(skip_to_pose_estimation=True)
            else:
                self.init_app()
        elif self.process_mode == 'batch':
            # Prompt for number of trials
            self.prompt_number_of_trials()

    def prompt_number_of_trials(self):
        num_trials = simpledialog.askinteger("Batch Processing",
                                             "Enter the number of trials:",
                                             minvalue=1)
        if not num_trials:
            messagebox.showerror("Input Error", "Number of trials must be at least 1.")
            self.root.destroy()
            return
        self.num_trials = num_trials
        self.create_batch_folder_structure()
        self.init_app()

    def create_single_folder_structure(self):
        # Create participant directory directly
        participant_path = os.path.join(self.participant_name)
    
        # Create calibration and videos subdirectories in the participant folder
        calibration_path = os.path.join(participant_path, 'calibration')
        videos_path = os.path.join(participant_path, 'videos')
    
        # Create all directories
        os.makedirs(calibration_path, exist_ok=True)
        os.makedirs(videos_path, exist_ok=True)
    
          
    def create_batch_folder_structure(self):
        # Create participant directory
        participant_path = os.path.join(self.participant_name)
        os.makedirs(participant_path, exist_ok=True)
        
        # Create calibration directory
        calibration_path = os.path.join(participant_path, 'calibration')
        os.makedirs(calibration_path, exist_ok=True)
                
    def init_app(self, skip_to_pose_estimation=False):
        # Initialize variables
        self.init_variables()
    
        # Create the notebook and tabs
        self.create_tabs()
    
        if skip_to_pose_estimation:
            if self.process_mode == 'batch':
                # For batch mode, handle trials
                response = messagebox.askyesno("Delete Old Videos?",
                                            "Would you like to delete old videos from all trials?")
                if response:
                    # Delete all trial folders if they exist
                    for i in range(1, self.num_trials + 1):
                        trial_path = os.path.join(self.participant_name, f'Trial_{i}')
                        if os.path.exists(trial_path):
                            shutil.rmtree(trial_path)
    
                    # Recreate trial structure
                    for i in range(1, self.num_trials + 1):
                        trial_path = os.path.join(self.participant_name, f'Trial_{i}')
                        os.makedirs(trial_path, exist_ok=True)
                        videos_path = os.path.join(trial_path, 'videos')
                        os.makedirs(videos_path, exist_ok=True)
    
            else:
                # Single mode handling
                videos_path = os.path.join(self.participant_name, 'videos')
                if os.path.exists(videos_path):
                    if messagebox.askyesno("Delete Old Videos?",
                                        "Would you like to delete old Pose videos?"):
                        shutil.rmtree(videos_path)
                os.makedirs(videos_path, exist_ok=True)
    
            # Switch to Pose Estimation tab
            self.notebook.select(self.pose_model_frame)
            
            if self.process_mode == 'batch':
                messagebox.showinfo("Select New Videos",
                                f"You will be asked to select videos for {self.num_trials} trials.\n\n"
                                f"For each trial, you will need to select {self.num_cameras_var.get()} camera videos.")
            else:
                messagebox.showinfo("Select New Videos",
                                "Please select new videos for each camera.")
    
            # Proceed to video selection
            if not self.process_mode == 'batch':
                # For single mode, proceed directly to video input
                num_cameras = int(self.num_cameras_var.get()) if self.num_cameras_var.get() else 0
                video_extension = self.video_extension_var.get()
                if not self.input_videos(os.path.join(self.participant_name, 'videos')):
                    return
        else:
            # Normal initialization without skipping
            config_file_path = os.path.join(self.participant_name, 'Config.toml')
            if self.process_mode == 'batch':
                # Create batch session structure
                self.create_batch_folder_structure()
            else:
                # Create single trial structure
                self.create_single_folder_structure()
    
            # Generate initial config file if it doesn't exist
            self.generate_config_toml()
    
            # Start with the first tab
            self.notebook.select(0)
    
            # Reset progress tracking
            self.current_step = 0
            self.progress_var.set(0)
            self.progress_bar.set(0)
            self.progress_label.configure(text="Progress: 0%")
    
    def get_text(self, key):
        texts = {
            'en': {
                'calibration_config': "Calibration Configuration",
                'prepare_video': "Checkerboard only?",
                'pose_model': "Pose Estimation",
                'synchronization': "Synchronization",
                'activation': "Activation",
                'gait_analysis': "Gait Analysis",
                'advanced_configuration': "Advanced Configuration",
                'participant_name_title': "Participant Name",
                'participant_name_prompt': "Please enter the participant name:",
                'calibration_type': "Calibration Type:",
                'calculate': "Calculate",
                'convert': "Convert",
                'config_exists_message': "Configuration already exists, would you like to config Pose2Sim again?",
                'config_exists_title': "Configuration exists",
                'num_cameras': "Number of Cameras:",
                'checkerboard_width': "Checkerboard Width:",
                'checkerboard_height': "Checkerboard Height:",
                'square_size': "Square Size (mm):",
                'video_extension': "Video/Image Extension (e.g., mp4, png):",
                'proceed_calibration': "Proceed with Calibration",
                'participant_height': "Participant Height (m):",
                'participant_mass': "Participant Mass (kg):",
                'pose_model_selection': "Pose Model Selection:",
                'mode': "Mode:",
                'proceed_pose_estimation': "Proceed with Pose Estimation",
                'activate_pose2sim': "Activate Pose2Sim",
                'input_error': "Input Error",
                'invalid_input': "Invalid input: ",
                'error': "Error",
                'calibration_done_camera': "The first step of calibration configuration is done.",
                'calibration_done_scene': "Calibration configuration now is done for scene, please continue with pose configuration.",
                'calibration_complete': "To complete the calibration configuration, please select 8 points and either measure their distances or use their known coordinates. Note that the first point will serve as the origin.",
                'calibration_complete_finale': "The calibration configuration is now complete. When you run Pose2Sim, you will be prompted to confirm checkerboard detection by clicking either Y or N. You will also need to click on the same 8 points in the scene, so please pay attention to the order in which you entered them. You can always return to this tab to see where to click. Instructions will appear in the top left corner of each image to guide you through the process. Please continue to the next tab",
                'pose_estimation_complete': "Pose estimation configuration updated successfully.",
                'select_checkerboard': "Please select the checkerboard videos for each camera, make sure it's similar to the one shown here.",
                'select_scene_images': "Please select the scene images for each camera.",
                'select_videos': "Please select the videos/images for each camera.",
                'input_checkerboard': "Input Checkerboard Videos",
                'input_scene_images': "Please select videos of the scene where your calibration object is clearly seen.",
                'input_videos': "Input Videos/Images",
                'no_file_selected': "No file selected for camera ",
                'select_image_camera': "Now we need to enter some distances using coordinates. Please choose a scene video that clearly shows the calibration object (refer to the documentation for assistance).",
                'scene_coordinates': "Scene Coordinates",
                'select_language': "Select Language / Choisir la langue",
                'ok': "OK",
                'point': "Point",
                'enter_coordinates': "Enter coordinates relative to Point 1:",
                'x_meters': "X (meters):",
                'y_meters': "Y (meters):",
                'z_meters': "Z (meters):",
                'coordinates_input': "Coordinates Input",
                'save_pdf': "Save as PDF",
                'checkerboard_image_saved': "Checkerboard image saved as ",
                'saved': "Saved",
                'number_of_points': "Enter the number of points (minimum 8):",
                'number_of_points_title': "Number of Points",
                'must_enter_points': "You must enter at least 8 points.",
                'click_image': "Click on the image to select 8 points",
                'enter_distance': "Now enter the distances for each point.",
                'language_selection': "Select Language / Choisir la langue",
                'convert_from': "Convert From:",
                'select_convert_file': "Please select the calibration file to convert.",
                'skip_sync': "Skip synchronization part? (only if videos are already synchronized)",
                'yes': "Yes",
                'no': "No",
                'select_keypoints': "Select keypoints to consider for synchronization:",
                'approx_time': "Do you want to specify approximate times of movement? (Recommended)",
                'auto': "Auto",
                'enter_times': "Enter approximate times of the synchronisation movement for each camera:",
                'advanced_management': "Do you want to proceed with advanced management?",
                'skip_part': "Skip that part, no need :)",
                'min_cameras_warning': "Using only two cameras is the minimum for Pose2Sim.",
                'select_segment': "Select a segment that had a fast movement or 'all' if you are not sure.",
                'play_videos': "Play videos to select timing.",
                'activate': "Activate",
                'next': "Next",
                'delete_old_videos_title': "Delete the old videos?",
                'delete_old_videos_message': "Would you like to delete old Pose videos?",
                'save_synchronization': "Save Synchronization Settings",
                'save_advanced_settings': "Save Advanced Settings",
                'frame_rate': "Frame Rate (fps):",
                'frame_range': "Frame Range (e.g., [10, 300]):",
                'person_association': "Person Association",
                'triangulation': "Triangulation",
                'filtering': "Filtering",
                'marker_augmentation': "Marker Augmentation",
                'kinematics': "Kinematics",
                'tips_placeholder': "",
                'gait_analysis': "Gait Analysis",
                'under_construction': "You’re now ready to activate Pose2Sim. If you’d like to access advanced configuration options, please go to the next tab.",
                'calibration_final_message': "Calibration configuration is now complete, please proceed to pose estimation configuration.",
                'skip_prepare_video': "Do you want to skip preparing videos?",
                'only_checkerboard': "Do your videos contain only checkerboard images?",
                'enter_time_seconds': "Enter time interval in seconds for image extraction:",
                'extrinsic_format_prompt': "Enter the image format (e.g., png, jpg):",
                'proceed_prepare_video': "Proceed with Prepare Video",
                'time_interval_error': "Please enter a valid time interval.",
                'invalid_time_interval': "Invalid time interval entered.",
                'select_calibration_videos': "Select Calibration Videos",
                'no_videos_selected': "No videos were selected.",
                'review_images': "Review Extracted Images",
                'delete': "Delete",
                'save_images': "Save Selected Images",
                'images_reviewed': "Review Completed",
                'images_saved': "Selected images have been saved.",
                'multiple_persons': "Multiple Persons:",
                'single_person': "Single Person:",
                'number_of_people': "Number of People:",
                'submit_number': "Submit",
                'invalid_number_of_people': "Invalid number of people entered.",
                'invalid_height_mass': "Invalid height or mass entered.",
                'submitted': "Submitted",
                'people_details_saved': "Participant details have been saved successfully.",
                'previous': "Previous",
                'keep': "Keep",
                'participant': 'Participant',
                'height': 'Height (m):',
                'batch_configuration' :'batch configuration',
                'mass': 'Mass (kg):',
                'submit': 'Submit',
            },
            'fr': {
                'calibration_config': "Configuration de Calibration",
                'prepare_video': "Préparer la Vidéo",
                'pose_model': "Estimation de Pose",
                'synchronization': "Synchronisation",
                'activation': "Activation",
                'gait_analysis': "Analyse de la Marche",
                'advanced_configuration': "Configuration Avancée",
                'participant_name_title': "Nom du Participant",
                'participant_name_prompt': "Veuillez entrer le nom du participant :",
                'calibration_type': "Type de Calibration :",
                'calculate': "Calculer",
                'convert': "Convertir",
                'config_exists_message': "La configuration existe déjà, souhaitez-vous configurer à nouveau Pose2Sim ?",
                'config_exists_title': "Configuration existante",
                'num_cameras': "Nombre de Caméras :",
                'checkerboard_width': "Largeur de l'échiquier :",
                'checkerboard_height': "Hauteur de l'échiquier :",
                'square_size': "Taille du carré (mm) :",
                'video_extension': "Extension Vidéo/Image (ex: mp4, png) :",
                'proceed_calibration': "Procéder à la Calibration",
                'participant_height': "Taille du Participant (m) :",
                'participant_mass': "Masse du Participant (kg) :",
                'pose_model_selection': "Sélection du Modèle de Pose :",
                'mode': "Mode :",
                'proceed_pose_estimation': "Procéder à l'Estimation de Pose",
                'activate_pose2sim': "Activer Pose2Sim",
                'input_error': "Erreur d'Entrée",
                'invalid_input': "Entrée invalide : ",
                'error': "Erreur",
                'calibration_done_camera': "La première partie de configuration de la calibration est terminée.",
                'calibration_done_scene': "La configuration de la calibration est terminée, veuillez continuer avec la configuration de la pose.",
                'calibration_complete': "Afin de compléter la configuration de la calibration, vous devez choisir 8 points dont vous mesurerez la distance, ou dont vous connaissez les distances en mètres. Le premier point sera l'origine",
                'calibration_complete_finale': "La configuration de la calibration est maintenant terminée. Veuillez noter qu'une fois que vous lancez Pose2Sim, vous devrez confirmer la détection du damier (en cliquant sur O ou N) et cliquer sur les mêmes 8 points pour la scène. .",
                'pose_estimation_complete': "Configuration de l'estimation de pose mise à jour avec succès.",
                'select_checkerboard': "Veuillez sélectionner les vidéos de l'échiquier pour chaque caméra, assurez-vous qu'il est similaire à celui montré ici.",
                'select_scene_images': "Veuillez sélectionner les images de la scène pour chaque caméra.",
                'select_videos': "Veuillez sélectionner les vidéos/images pour chaque caméra.",
                'input_checkerboard': "Importer les Vidéos de l'Échiquier",
                'input_scene_images': "Veuillez sélectionner des vidéos de la scène où votre objet est clairement visible",
                'input_videos': "Importer les Vidéos/Images",
                'no_file_selected': "Aucun fichier sélectionné pour la caméra ",
                'select_image_camera': "Nous devons maintenant entrer des distances en termes de coordonnées. Veuillez sélectionner une vue d'un objet présent dans la scène.",
                'scene_coordinates': "Coordonnées de la Scène",
                'select_language': "Sélectionnez la Langue / Choose Language",
                'ok': "OK",
                'point': "Point",
                'enter_coordinates': "Entrez les coordonnées relatives au Point 1 :",
                'x_meters': "X (mètres) :",
                'y_meters': "Y (mètres) :",
                'z_meters': "Z (mètres) :",
                'coordinates_input': "Saisie des Coordonnées",
                'save_pdf': "Enregistrer en PDF",
                'checkerboard_image_saved': "Image de l'échiquier enregistrée sous ",
                'saved': "Enregistré",
                'number_of_points': "Entrez le nombre de points (minimum 8) :",
                'number_of_points_title': "Nombre de Points",
                'must_enter_points': "Vous devez entrer au moins 8 points.",
                'click_image': "Cliquez sur l'image pour sélectionner 8 points",
                'enter_distance': "Maintenant, entrez les distances pour chaque point.",
                'language_selection': "Sélectionnez la Langue / Choose Language",
                'convert_from': "Convertir Depuis :",
                'select_convert_file': "Veuillez sélectionner le fichier de calibration à convertir.",
                'skip_sync': "Passer la partie synchronisation ? (Les vidéos sont déjà synchronisées)",
                'yes': "Oui",
                'no': "Non",
                'select_keypoints': "Sélectionnez les points clés à considérer pour la synchronisation :",
                'approx_time': "Voulez-vous spécifier des temps approximatifs de mouvement ? (Recommandé)",
                'auto': "Auto",
                'enter_times': "Entrez les temps approximatifs de mouvement pour chaque caméra :",
                'advanced_management': "Voulez-vous procéder à une gestion avancée ?",
                'skip_part': "Passez cette partie, pas besoin :)",
                'min_cameras_warning': "Utiliser seulement deux caméras est le minimum pour Pose2Sim.",
                'select_segment': "Sélectionnez un segment qui a eu un mouvement rapide ou 'all' si vous n'êtes pas sûr.",
                'play_videos': "Lisez les vidéos pour sélectionner le timing.",
                'activate': "Activer",
                'next': "Suivant",
                'delete_old_videos_title': "Supprimer les anciennes vidéos ?",
                'delete_old_videos_message': "Voulez-vous supprimer les anciennes vidéos de Pose ?",
                'save_synchronization': "Enregistrer les paramètres de synchronisation",
                'save_advanced_settings': "Enregistrer les paramètres avancés",
                'frame_rate': "Fréquence d'images (fps) :",
                'frame_range': "Plage d'images (ex: [10, 300]) :",
                'person_association': "Association de Personne",
                'triangulation': "Triangulation",
                'filtering': "Filtrage",
                'marker_augmentation': "Augmentation de Marqueurs",
                'kinematics': "Cinématique",
                'tips_placeholder': "",
                'gait_analysis': "Analyse de la Marche",
                'under_construction': "En construction",
                'calibration_final_message': "La configuration de la calibration est maintenant terminée, veuillez passer à la configuration de l'estimation de la pose.",
                'skip_prepare_video': "Voulez-vous passer la préparation des vidéos ?",
                'only_checkerboard': "Vos vidéos contiennent-elles uniquement des images de damier ?",
                'enter_time_seconds': "Entrez l'intervalle de temps en secondes pour l'extraction des images :",
                'extrinsic_format_prompt': "Entrez le format d'image extrinsèque (ex : png, jpg) :",
                'proceed_prepare_video': "Procéder à la Préparation de la Vidéo",
                'time_interval_error': "Veuillez entrer un intervalle de temps valide.",
                'invalid_time_interval': "Intervalle de temps invalide entré.",
                'select_calibration_videos': "Sélectionnez les Vidéos de Calibration",
                'no_videos_selected': "Aucune vidéo n'a été sélectionnée.",
                'review_images': "Revoir les Images Extraites",
                'delete': "Supprimer",
                'save_images': "Enregistrer les Images Sélectionnées",
                'images_reviewed': "Revue Terminée",
                'images_saved': "Les images sélectionnées ont été enregistrées.",
                'multiple_persons': "Multiples Personnes :",
                'single_person': "Personne Unique :",
                'number_of_people': "Nombre de Personnes :",
                'submit_number': "Soumettre",
                'invalid_number_of_people': "Nombre de personnes invalide entré.",
                'invalid_height_mass': "Hauteur ou masse invalide entrée.",
                'submitted': "Soumis",
                'people_details_saved': "Les détails du participant ont été enregistrés avec succès.",
                'previous': "Précédent",
                'keep': "Garder",
                'batch configuration': 'configuration de batch',
                'participant': 'Participant',
                'height': 'Taille (m) :',
                'mass': 'Masse (kg) :',
                'submit': 'Soumettre',
            }
        }
        if self.language in texts:
            return texts[self.language].get(key, key)
        else:
            return key
    

    def init_variables(self):
        # Initialize variables to hold user inputs
        # Load from Config.toml if exists
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            config = parse(config_content)
        else:
            config = {}

        # Initialize variables as per your configuration structure
        
        self.calibration_type_var = ctk.StringVar(
            value=config.get('calibration', {}).get('calibration_type', 'calculate'))
        self.num_cameras_var = ctk.StringVar(value='2')  # Default value
        self.checkerboard_width_var = ctk.StringVar(
            value=str(config.get('calibration', {}).get('calculate', {}).get('intrinsics', {}).get('intrinsics_corners_nb', [7, 5])[0]))
        self.checkerboard_height_var = ctk.StringVar(
            value=str(config.get('calibration', {}).get('calculate', {}).get('intrinsics', {}).get('intrinsics_corners_nb', [7, 5])[1]))
        self.square_size_var = ctk.StringVar(
            value=str(config.get('calibration', {}).get('calculate', {}).get('intrinsics', {}).get('intrinsics_square_size', 45.0)))
        self.video_extension_var = ctk.StringVar(
            value=config.get('pose', {}).get('vid_img_extension', 'mp4'))
        self.participant_height_var = ctk.StringVar(
            value=str(config.get('project', {}).get('participant_height', 1.72)))
        self.participant_mass_var = ctk.StringVar(
            value=str(config.get('project', {}).get('participant_mass', 70.0)))
        self.pose_model_var = ctk.StringVar(
            value=config.get('pose', {}).get('pose_model', 'HALPE_26'))
        self.mode_var = ctk.StringVar(value=config.get('pose', {}).get('mode', 'balanced'))
        self.convert_from_var = ctk.StringVar(
            value=config.get('calibration', {}).get('convert', {}).get('convert_from', 'qualisys'))
        self.binning_factor_var = ctk.StringVar(
            value=str(config.get('calibration', {}).get('convert', {}).get('qualisys', {}).get('binning_factor', 1)))
        self.sync_videos_var = ctk.StringVar(value='no')  # Default to 'no'
        self.keypoints_var = ctk.StringVar(value='all')
        self.approx_time_var = ctk.StringVar(value='auto')
        self.time_range_var = ctk.StringVar(value='2.0')
        self.likelihood_threshold_var = ctk.StringVar(value='0.4')
        self.filter_cutoff_var = ctk.StringVar(value='6')
        self.filter_order_var = ctk.StringVar(value='4')
        self.approx_time_entries = []
        self.approx_times = []

        # Advanced Configuration Variables
        self.frame_rate_var = ctk.StringVar(value=config.get('project', {}).get('frame_rate', 'auto'))
        self.frame_range_var = ctk.StringVar(value=str(config.get('project', {}).get('frame_range', [])))

        # personAssociation Variables
        self.likelihood_threshold_association_var = ctk.StringVar(
            value=str(config.get('personAssociation', {}).get('likelihood_threshold_association', 0.3)))
        self.reproj_error_threshold_association_var = ctk.StringVar(
            value=str(config.get('personAssociation', {}).get('single_person', {}).get('reproj_error_threshold_association', 20)))
        self.tracked_keypoint_var = ctk.StringVar(
            value=config.get('personAssociation', {}).get('single_person', {}).get('tracked_keypoint', 'Neck'))

        # Triangulation Variables
        self.reproj_error_threshold_triangulation_var = ctk.StringVar(
            value=str(config.get('triangulation', {}).get('reproj_error_threshold_triangulation', 15)))
        self.likelihood_threshold_triangulation_var = ctk.StringVar(
            value=str(config.get('triangulation', {}).get('likelihood_threshold_triangulation', 0.3)))
        self.min_cameras_for_triangulation_var = ctk.StringVar(
            value=str(config.get('triangulation', {}).get('min_cameras_for_triangulation', 2)))

        # Filtering Variables
        self.filter_type_var = ctk.StringVar(
            value=config.get('filtering', {}).get('type', 'butterworth'))

        # Butterworth Variables
        self.filter_cutoff_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('butterworth', {}).get('cut_off_frequency', 6)))
        self.filter_order_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('butterworth', {}).get('order', 4)))

        # Kalman Variables
        self.kalman_trust_ratio_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('kalman', {}).get('trust_ratio', 100)))
        self.kalman_smooth_var = ctk.BooleanVar(
            value=config.get('filtering', {}).get('kalman', {}).get('smooth', True))

        # Butterworth on Speed Variables
        self.butterworth_on_speed_order_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('butterworth_on_speed', {}).get('order', 4)))
        self.butterworth_on_speed_cut_off_frequency_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('butterworth_on_speed', {}).get('cut_off_frequency', 10)))

        # Gaussian Variables
        self.gaussian_sigma_kernel_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('gaussian', {}).get('sigma_kernel', 2)))

        # LOESS Variables
        self.LOESS_nb_values_used_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('LOESS', {}).get('nb_values_used', 30)))

        # Median Variables
        self.median_kernel_size_var = ctk.StringVar(
            value=str(config.get('filtering', {}).get('median', {}).get('kernel_size', 9)))

        # Marker Augmentation Variables  
        self.make_c3d_var = ctk.BooleanVar(
            value=config.get('markerAugmentation', {}).get('make_c3d', True))

        # Kinematics Variables
        self.use_augmentation_var = ctk.BooleanVar(
            value=config.get('kinematics', {}).get('use_augmentation', True))
        self.right_left_symmetry_var = ctk.BooleanVar(
            value=config.get('kinematics', {}).get('right_left_symmetry', True))
        self.remove_individual_scaling_setup_var = ctk.BooleanVar(
            value=config.get('kinematics', {}).get('remove_individual_scaling_setup', True))
        self.remove_individual_IK_setup_var = ctk.BooleanVar(
            value=config.get('kinematics', {}).get('remove_individual_IK_setup', True))

        # other var here**
        self.extrinsic_format_var = ctk.StringVar(
            value=config.get('calibration', {}).get('calculate', {}).get('extrinsics', {}).get('scene', {}).get('extrinsics_extension', 'png'))
        self.time_interval_var = ctk.StringVar(value='1')
        self.num_people_var = ctk.StringVar(value='2')  
        self.time_interval_var = ctk.StringVar(value='1') 

    def create_tabs(self):
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
    
        # Create frames for each tab and map them
        for tab_key in self.tab_order:
            # Create the actual content frame using CustomTkinter
            content_frame = ctk.CTkFrame(self.notebook)
            content_frame.pack(expand=True, fill='both')
    
            # Store references to frames
            self.tab_frames[tab_key] = content_frame
    
            # Add the frame to the notebook with the updated tab name
            self.notebook.add(content_frame, text=self.tab_names[tab_key])
    
        # Define frame attributes to match expected names
        self.calibration_frame = self.tab_frames['calibration_config']
        self.prepare_video_frame = self.tab_frames['prepare_video']
        self.pose_model_frame = self.tab_frames['pose_model']
        self.synchronization_frame = self.tab_frames['synchronization']
        self.activation_frame = self.tab_frames['activation']
        self.advanced_configuration_frame = self.tab_frames['advanced_configuration']
    
        # Ensure to build the batch tab if the mode is set to batch
        if self.process_mode == 'batch':
            # Initialize the batch configuration tab if in batch mode
            self.batch_configuration_frame = self.tab_frames.get('batch_configuration', None)
            if self.batch_configuration_frame is None:
                self.batch_configuration_frame = ctk.CTkFrame(self.notebook)
                self.tab_frames['batch_configuration'] = self.batch_configuration_frame
                self.notebook.add(self.batch_configuration_frame, text=self.tab_names['batch_configuration'])
        
        # Assign each tab's build method
        self.build_calibration_tab()
        self.build_prepare_video_tab()
        self.build_pose_model_tab()
        self.build_synchronization_tab()
        self.build_activation_tab()
        self.build_advanced_configuration_tab()
    
        if self.process_mode == 'batch':
            self.build_batch_configuration_tab()
    
        self.notebook.bind('<<NotebookTabChanged>>', self.handle_tab_change)

    def update_tab_indicator(self, tab_key, saved=True):
        """
        Update the tab title indicator based on whether the step is completed.
        """
        if saved:
            indicator = '🟢 '
        else:
            indicator = '🔴 '

        # Update the tab_names dictionary
        base_title = self.get_text(tab_key)
        self.tab_names[tab_key] = indicator + base_title

        # Find the index of the tab
        tab_index = self.tab_order.index(tab_key)

        # Update the tab text in the notebook
        self.notebook.tab(tab_index, text=self.tab_names[tab_key])
        
    def build_calibration_tab(self):
        frame = ctk.CTkFrame(self.calibration_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)
    
        # Calibration Configuration Heading
        ctk.CTkLabel(frame, text="Calibration Configuration",
                    font=('Helvetica', 20, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky='w')
    
        # Calibration Type Selection
        ctk.CTkLabel(frame, text="Calibration Type:").grid(
            row=1, column=0, sticky='w', pady=5)
        calibration_type_frame = ctk.CTkFrame(frame, fg_color="transparent")
        calibration_type_frame.grid(row=1, column=1, sticky='w')
        
        self.type_confirmed = False  # Add this flag
        
        ctk.CTkRadioButton(calibration_type_frame, text="Convert", variable=self.calibration_type_var,
                        value='convert', command=self.on_calibration_type_change).pack(side='left', padx=10)
        ctk.CTkRadioButton(calibration_type_frame, text="Calculate", variable=self.calibration_type_var,
                        value='calculate', command=self.on_calibration_type_change).pack(side='left', padx=10)
    
        # Frame for calculate options
        self.calculate_options_frame = ctk.CTkFrame(frame)
        self.calculate_options_frame.grid(row=2, column=0, columnspan=2, sticky='w', pady=10)
    
        # Number of Cameras (moved outside specific frames to be available for both options)
        self.cameras_frame = ctk.CTkFrame(frame)
        self.cameras_frame.grid(row=3, column=0, columnspan=2, sticky='w', pady=5)
        ctk.CTkLabel(self.cameras_frame, text="Number of Cameras:").grid(
            row=0, column=0, sticky='w', pady=3)
        self.camera_entry = ctk.CTkEntry(self.cameras_frame, textvariable=self.num_cameras_var, width=100)
        self.camera_entry.grid(row=0, column=1, sticky='w')
    
        # Calculate specific options
        # Checkerboard Width
        ctk.CTkLabel(self.calculate_options_frame, text="Checkerboard Width:").grid(
            row=1, column=0, sticky='w', pady=3)
        self.width_entry = ctk.CTkEntry(self.calculate_options_frame, textvariable=self.checkerboard_width_var, width=100)
        self.width_entry.grid(row=1, column=1, sticky='w')
    
        # Checkerboard Height
        ctk.CTkLabel(self.calculate_options_frame, text="Checkerboard Height:").grid(
            row=2, column=0, sticky='w', pady=3)
        self.height_entry = ctk.CTkEntry(self.calculate_options_frame, textvariable=self.checkerboard_height_var, width=100)
        self.height_entry.grid(row=2, column=1, sticky='w')
    
        # Square Size
        ctk.CTkLabel(self.calculate_options_frame, text="Square Size (mm):").grid(
            row=3, column=0, sticky='w', pady=3)
        self.square_entry = ctk.CTkEntry(self.calculate_options_frame, textvariable=self.square_size_var, width=100)
        self.square_entry.grid(row=3, column=1, sticky='w')
    
        # Video Extension
        ctk.CTkLabel(self.calculate_options_frame, text="Video/Image Extension:").grid(
            row=4, column=0, sticky='w', pady=3)
        self.extension_entry = ctk.CTkEntry(self.calculate_options_frame, textvariable=self.video_extension_var, width=100)
        self.extension_entry.grid(row=4, column=1, sticky='w')
    
        # Convert From Options Frame
        self.convert_frame = ctk.CTkFrame(frame)
        self.convert_frame.grid(row=2, column=0, columnspan=2, sticky='w', pady=10)
        ctk.CTkLabel(self.convert_frame, text="Convert From:").grid(row=0, column=0, sticky='w')
        convert_options = ['qualisys', 'optitrack', 'vicon', 'opencap',
                        'easymocap', 'biocv', 'anipose', 'freemocap']
        self.convert_menu = ctk.CTkOptionMenu(self.convert_frame, variable=self.convert_from_var,
                                            values=convert_options)
        self.convert_menu.grid(row=0, column=1, sticky='w')
    
        # Confirm Type Button
        self.confirm_type_button = ctk.CTkButton(frame, text="Confirm Selection",
                                            command=self.confirm_calibration_type, width=200, height=40)
        self.confirm_type_button.grid(row=4, column=0, columnspan=2, pady=10)
    
        # Proceed Button (initially hidden)
        self.proceed_button = ctk.CTkButton(frame, text="Proceed with Calibration",
                                        command=self.proceed_calibration, width=200, height=40)
        self.proceed_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.proceed_button.grid_remove()  # Hide initially
    
        # Frame for checkerboard image
        self.checkerboard_display_frame = ctk.CTkFrame(frame)
        self.checkerboard_display_frame.grid(row=0, column=2, rowspan=6, padx=20, pady=10)
    
        # Initialize with current calibration type
        self.on_calibration_type_change()
    
    def confirm_calibration_type(self):
        """Confirm the calibration type selection and lock inputs"""
        try:
            # Validate number of cameras
            num_cameras = int(self.num_cameras_var.get())
            if num_cameras < 2:
                messagebox.showerror("Error", "Number of cameras must be at least 2")
                return
                
            if self.calibration_type_var.get() == 'calculate':
                # Validate calculate-specific inputs
                if not all([self.checkerboard_width_var.get(),
                        self.checkerboard_height_var.get(),
                        self.square_size_var.get(),
                        self.video_extension_var.get()]):
                    messagebox.showerror("Error", "All fields must be filled")
                    return
                    
            # Disable all inputs
            self.camera_entry.configure(state='disabled')
            self.convert_menu.configure(state='disabled')
            if self.calibration_type_var.get() == 'calculate':
                self.width_entry.configure(state='disabled')
                self.height_entry.configure(state='disabled')
                self.square_entry.configure(state='disabled')
                self.extension_entry.configure(state='disabled')
                
                # Enable Prepare Video tab for Calculate
                prepare_video_index = self.tab_order.index('prepare_video')
                self.notebook.tab(prepare_video_index, state='normal')
            else:
                # Disable Prepare Video tab for Convert
                prepare_video_index = self.tab_order.index('prepare_video')
                self.notebook.tab(prepare_video_index, state='disabled')
                # Update tab indicators to skip prepare_video
                self.update_tab_indicator('prepare_video', saved=True)
                self.update_progress('prepare_video')
                
            # Update button visibility
            self.confirm_type_button.grid_remove()
            self.proceed_button.grid()
            
            # Set confirmation flag
            self.type_confirmed = True
            
            messagebox.showinfo("Success", "Configuration confirmed. Proceed with calibration when ready.")
            
        except ValueError:
            messagebox.showerror("Error", "Invalid number of cameras")
        
    def on_calibration_type_change(self):
        # If already confirmed, ask user if they want to modify settings
        if hasattr(self, 'type_confirmed') and self.type_confirmed:
            response = messagebox.askyesno("Confirm Changes", 
                                        "Do you want to modify the configuration? This will require reconfirmation.")
            if response:
                # Enable all inputs
                self.camera_entry.configure(state='normal')
                self.convert_menu.configure(state='normal')
                self.width_entry.configure(state='normal')
                self.height_entry.configure(state='normal')
                self.square_entry.configure(state='normal')
                self.extension_entry.configure(state='normal')
                
                # Show confirm button and hide proceed button
                self.confirm_type_button.grid()
                self.proceed_button.grid_remove()
                
                # Reset prepare_video tab state
                prepare_video_index = self.tab_order.index('prepare_video')
                self.notebook.tab(prepare_video_index, state='normal')
                
                # Reset confirmation flag
                self.type_confirmed = False
            else:
                # Revert radio button selection if user doesn't want to change
                self.calibration_type_var.set('calculate' if self.calibration_type_var.get() == 'convert' else 'convert')
                return
                
        # Show/hide appropriate frames based on selection
        if self.calibration_type_var.get() == 'calculate':
            self.calculate_options_frame.grid()
            self.convert_frame.grid_remove()
        else:
            self.calculate_options_frame.grid_remove()
            self.convert_frame.grid()
    def handle_tab_change(self, event):
        """Handle tab change events"""
        current_tab = self.notebook.select()
        current_tab_index = self.notebook.index(current_tab)
        
        # If trying to access prepare_video tab when disabled
        if self.tab_order[current_tab_index] == 'prepare_video':
            if self.calibration_type_var.get() == 'convert' and self.type_confirmed:
                # Prevent access and show message
                messagebox.showinfo("Info", "Prepare Video step is not needed for Convert calibration type.")
                # Switch to next available tab
                next_tab_index = current_tab_index + 1
                if next_tab_index < len(self.tab_order):
                    self.notebook.select(next_tab_index)
    def proceed_calibration(self):
        if not self.type_confirmed:
            messagebox.showerror("Error", "Please confirm your configuration first")
            return
      
        calibration_type = self.calibration_type_var.get()
        try:
            num_cameras = int(self.num_cameras_var.get()) if self.num_cameras_var.get() else 0
            if calibration_type == 'calculate' and num_cameras < 1:
                messagebox.showerror("Error", "Invalid number of cameras entered.")
                return
            elif calibration_type == 'calculate' and num_cameras == 2:
                messagebox.showwarning("Warning", "Using only two cameras is the minimum for Pose2Sim.")
    
            video_extension = self.video_extension_var.get()
    
            if calibration_type == 'calculate':
                checkerboard_width = int(self.checkerboard_width_var.get())
                checkerboard_height = int(self.checkerboard_height_var.get())
                square_size = float(self.square_size_var.get())
    
                # Generate and display checkerboard
                checkerboard_image = self.generate_checkerboard_image(
                    checkerboard_width, checkerboard_height, square_size)
                self.display_checkerboard_image(checkerboard_image)
    
                # Handle folder creation based on process mode
                if self.process_mode == 'single':
                    calibration_path = os.path.join(self.participant_name, 'calibration')
                else:
                    calibration_path = os.path.join(self.participant_name,'calibration')
    
                # Create calibration folders
                for cam in range(1, num_cameras + 1):
                    intrinsics_folder_name = f'int_cam{cam}_img'
                    extrinsics_folder_name = f'ext_cam{cam}_img'
                    intrinsics_path = os.path.join(calibration_path, 'intrinsics', intrinsics_folder_name)
                    extrinsics_path = os.path.join(calibration_path, 'extrinsics', extrinsics_folder_name)
                    os.makedirs(intrinsics_path, exist_ok=True)
                    os.makedirs(extrinsics_path, exist_ok=True)
    
                if not self.input_checkerboard_videos(num_cameras, video_extension):
                    return
                messagebox.showinfo("Calibration Step", "The first step of calibration configuration is done.")
    
                if not self.input_scene_images(num_cameras, video_extension):
                    return
                if not self.input_scene_coordinates():
                    return
    
                config_file_path = os.path.join(self.participant_name, 'Config.toml')
                if self.process_mode == 'batch':
                    config_file_path = os.path.join(self.participant_name,'Config.toml')
    
                self.generate_config_toml()
                self.update_config_toml(config_file_path, section='calibration')
                messagebox.showinfo("Calibration Complete",
                                "To complete the calibration configuration, you need to choose 8 points and measure their distances or use known coordinates. The first point will be the origin.")
    
            else:
                messagebox.showinfo("Convert Calibration",
                                "Convert option selected. Please select calibration file.")
                file_path = filedialog.askopenfilename(
                    title="Select calibration file to convert")
                if not file_path:
                    messagebox.showerror("Error", "No file selected for conversion.")
                    return
    
                # Copy file to appropriate calibration folder
                if self.process_mode == 'single':
                    dest_dir = os.path.join(self.participant_name, 'calibration')
                else:
                    dest_dir = os.path.join(self.participant_name,'calibration')
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(file_path))
                shutil.copy(file_path, dest_path)
    
                # Update Config.toml
                config_file_path = os.path.join(self.participant_name, 'Config.toml')
                if self.process_mode == 'batch':
                    config_file_path = os.path.join(self.participant_name,'Config.toml')
                self.generate_config_toml()
                self.update_config_toml(config_file_path, section='calibration')
                messagebox.showinfo("Calibration Complete",
                                "Calibration configuration is complete. Please proceed to pose estimation configuration.")
    
            # Update the tab indicator
            self.update_tab_indicator('calibration_config', saved=True)
            self.update_progress('calibration')
            
    
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def input_checkerboard_videos(self, num_cameras, video_extension):
        messagebox.showinfo("Input Videos",
                        "Please select the checkerboard videos for each camera.")
        base_path = self.participant_name
        if self.process_mode == 'batch':
            base_path = os.path.join(self.participant_name)
    
        for cam in range(1, num_cameras + 1):
            file_path = filedialog.askopenfilename(
                title=f"Select Checkerboard Video for Camera {cam}")
            if not file_path:
                messagebox.showerror("Error", f"No file selected for camera {cam}")
                return False
    
            intrinsics_folder_name = f'int_cam{cam}_img'
            dest_dir = os.path.join(base_path, 'calibration', 'intrinsics', intrinsics_folder_name)
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
    
        return True
    
    def input_scene_images(self, num_cameras, video_extension):
        messagebox.showinfo("Scene Images", 
                        "Please select scene images for each camera.")
        base_path = self.participant_name
        if self.process_mode == 'batch':
            base_path = os.path.join(self.participant_name)
    
        for cam in range(1, num_cameras + 1):
            file_path = filedialog.askopenfilename(
                title=f"Select Scene Image for Camera {cam}")
            if not file_path:
                messagebox.showerror("Error", f"No file selected for camera {cam}")
                return False
    
            extrinsics_folder_name = f'ext_cam{cam}_img'
            dest_dir = os.path.join(base_path, 'calibration', 'extrinsics', extrinsics_folder_name)
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)
    
        return True
    
    
    
    def display_checkerboard_image(self, image):
        max_size = 300
        img_width, img_height = image.size
        scale = min(max_size / img_width, max_size / img_height, 1)
        display_image = image.resize(
            (int(img_width * scale), int(img_height * scale)), Image.Resampling.LANCZOS)
    
        ctki = CTkImage(dark_image=display_image, light_image=display_image, 
                        size=(int(img_width * scale), int(img_height * scale)))
    
        for widget in self.checkerboard_display_frame.winfo_children():
            widget.destroy()
    
        panel = ctk.CTkLabel(self.checkerboard_display_frame, image=ctki, text="")
        panel.image = ctki
        panel.pack(padx=10, pady=10)
    
        def save_as_pdf():
            file_path = filedialog.asksaveasfilename(
                defaultextension='.pdf', 
                filetypes=[('PDF files', '*.pdf')])
            if file_path:
                image.save(file_path, 'PDF')
                messagebox.showinfo("Saved", f"Checkerboard image saved as {file_path}")
    
        button_frame = ctk.CTkFrame(self.checkerboard_display_frame)
        button_frame.pack(pady=5)
        ctk.CTkButton(button_frame, text="Save as PDF", 
                    command=save_as_pdf).pack(padx=5, pady=5)
    
    def generate_checkerboard_image(self, checkerboard_width, checkerboard_height, square_size):
        num_rows = checkerboard_height + 1
        num_cols = checkerboard_width + 1
        square_size = int(square_size)
    
        pattern = np.zeros((num_rows * square_size, num_cols * square_size), dtype=np.uint8)
        for row in range(num_rows):
            for col in range(num_cols):
                if (row + col) % 2 == 0:
                    pattern[row*square_size:(row+1)*square_size,
                        col*square_size:(col+1)*square_size] = 255
    
        return Image.fromarray(pattern)
    
    def input_scene_coordinates(self):
        """Handle scene coordinate input with color-coded points"""
        messagebox.showinfo("Scene Coordinates", 
                        "Please select a scene image or video showing the calibration object.")
        
        file_path = filedialog.askopenfilename(
            title="Select Scene Image/Video",
            filetypes=[("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov")])
        
        if not file_path:
            return False
    
        # Load image/video frame
        if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                messagebox.showerror("Error", "Failed to read video frame")
                return False
        else:
            frame = cv2.imread(file_path)
            if frame is None:
                messagebox.showerror("Error", "Failed to load image")
                return False
    
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Create matplotlib figure for point selection
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.imshow(frame_rgb)
        self.ax.set_title("Click to select 8 points")
    
        self.points_2d = []
        self.point_markers = []  # Store point markers for color updates
    
        def onclick(event):
            if len(self.points_2d) < 8:
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    self.points_2d.append((x, y))
                    # Plot point in red initially
                    point = self.ax.plot(x, y, 'ro')[0]
                    self.point_markers.append(point)
                    # Add point number
                    self.ax.text(x + 5, y + 5, str(len(self.points_2d)), color='white')
                    self.fig.canvas.draw()
                    
                    if len(self.points_2d) == 8:
                        self.input_coordinates(self.points_2d)
    
        # Create canvas and connect click event
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.calibration_frame)
        self.canvas.get_tk_widget().pack(pady=10)
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
        
        return True
    
    def input_coordinates(self, points_2d):
        self.object_coords_3d = []
        self.current_point_index = 0
    
        # Define config file path based on process mode
        if self.process_mode == 'batch':
            self.config_file_path = os.path.join(self.participant_name, 'Config.toml')
        else:
            self.config_file_path = os.path.join(self.participant_name, 'Config.toml')
    
        def create_coordinate_window():
            if self.current_point_index >= len(points_2d):
                if len(self.object_coords_3d) == len(points_2d):
                    self.update_config_toml(self.config_file_path, section='coordinates')
                    messagebox.showinfo(
                        "Calibration Complete",
                        "The calibration configuration is now complete. When you run Pose2Sim, you will be prompted to confirm checkerboard detection by clicking either Y or N. You will also need to click on the same 8 points in the scene, so please pay attention to the order in which you entered them. You can always return to this tab to see where to click. Instructions will appear in the top left corner of each image to guide you through the process. Please continue to the next tab"
                    )
                return
    
            # Change current point color to yellow when window opens
            if self.current_point_index < len(self.point_markers):
                self.point_markers[self.current_point_index].set_color('yellow')
                self.fig.canvas.draw()
    
            # Predefined coordinates
            predefined_coords = [
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [1.2, 0.8, 0.0],
                [0.0, 0.80, 0.0],
                [0.0, 0.0, 0.75],
                [1.2, 0.0, 0.75],
                [1.2, 0.80, 0.75],
                [0.0, 0.80, 0.75]
            ]
    
            # Create and configure window
            coord_window = ctk.CTkToplevel(self.root)
            coord_window.title(f"Point {self.current_point_index + 1} Coordinates Input")
            coord_window.geometry("400x400")
            coord_window.transient(self.root)  #
            coord_window.grab_set()  
    
            # Create main frame
            main_frame = ctk.CTkFrame(coord_window)
            main_frame.pack(expand=True, fill='both', padx=20, pady=20)
    
            # Title
            title_label = ctk.CTkLabel(
                main_frame,
                text=f"Enter Coordinates for Point {self.current_point_index + 1}",
                font=('Helvetica', 16, 'bold')
            )
            title_label.pack(pady=(0, 20))
    
            # Variables for coordinates
            x_var = ctk.StringVar(value=str(predefined_coords[self.current_point_index][0]))
            y_var = ctk.StringVar(value=str(predefined_coords[self.current_point_index][1]))
            z_var = ctk.StringVar(value=str(predefined_coords[self.current_point_index][2]))
    
            # Create coordinate entry frames
            coords_frame = ctk.CTkFrame(main_frame)
            coords_frame.pack(fill='x', pady=10)
    
            # X coordinate
            x_frame = ctk.CTkFrame(coords_frame)
            x_frame.pack(fill='x', pady=5)
            x_label = ctk.CTkLabel(x_frame, text="X (meters):", width=100)
            x_label.pack(side='left', padx=5)
            x_entry = ctk.CTkEntry(x_frame, textvariable=x_var, width=150)
            x_entry.pack(side='left', padx=5)
    
            # Y coordinate
            y_frame = ctk.CTkFrame(coords_frame)
            y_frame.pack(fill='x', pady=5)
            y_label = ctk.CTkLabel(y_frame, text="Y (meters):", width=100)
            y_label.pack(side='left', padx=5)
            y_entry = ctk.CTkEntry(y_frame, textvariable=y_var, width=150)
            y_entry.pack(side='left', padx=5)
    
            # Z coordinate
            z_frame = ctk.CTkFrame(coords_frame)
            z_frame.pack(fill='x', pady=5)
            z_label = ctk.CTkLabel(z_frame, text="Z (meters):", width=100)
            z_label.pack(side='left', padx=5)
            z_entry = ctk.CTkEntry(z_frame, textvariable=z_var, width=150)
            z_entry.pack(side='left', padx=5)
    
            # Disable entries for first point as it's the origine
            if self.current_point_index == 0:
                x_entry.configure(state='disabled')
                y_entry.configure(state='disabled')
                z_entry.configure(state='disabled')
    
            # Point position indicator
            point_indicator = ctk.CTkLabel(
                main_frame,
                text=f"Point {self.current_point_index + 1} of {len(points_2d)}",
                font=('Helvetica', 12)
            )
            point_indicator.pack(pady=10)
    
            # Image coordinates
            point_pos = points_2d[self.current_point_index]
            position_label = ctk.CTkLabel(
                main_frame,
                text=f"Image coordinates: ({point_pos[0]:.1f}, {point_pos[1]:.1f})",
                font=('Helvetica', 12)
            )
            position_label.pack(pady=5)
    
            def submit():
                try:
                    # Convert string values to float
                    x = float(x_var.get())
                    y = float(y_var.get())
                    z = float(z_var.get())
    
                    # Save coordinates
                    self.object_coords_3d.append([x, y, z])
    
                    # Change point color to green after saving coordinates
                    if self.current_point_index < len(self.point_markers):
                        self.point_markers[self.current_point_index].set_color('green')
                        self.fig.canvas.draw()
    
                    self.current_point_index += 1
    
                    # Destroy current window
                    coord_window.destroy()
    
                    # Create next window
                    self.root.after(100, create_coordinate_window)
    
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid numbers for all coordinates")
                    return
    
            # Next button
            next_button = ctk.CTkButton(
                main_frame,
                text="Next Point",
                command=submit,
                width=200,
                height=40
            )
            next_button.pack(pady=20)
    
            # Bind Enter key
            coord_window.bind('<Return>', lambda e: submit())
    
        # Start the process with the first window
        create_coordinate_window()
        self.update_progress('calibration')
    
                        
    def generate_config_toml(self):
        """Only generates template if config file doesn't exist"""
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        if self.process_mode == 'batch':
            config_file_path = os.path.join(self.participant_name,'Config.toml')
    
        # Only generate template if file doesn't exist
        if not os.path.exists(config_file_path):
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(self.config_template)

    
    def on_calibration_type_change(self):
        # If already confirmed, ask user if they want to modify settings
        if hasattr(self, 'type_confirmed') and self.type_confirmed:
            response = messagebox.askyesno("Confirm Changes", 
                                        "Do you want to modify the configuration? This will require reconfirmation.")
            if response:
                # Enable all inputs
                self.camera_entry.configure(state='normal')
                self.convert_menu.configure(state='normal')
                self.width_entry.configure(state='normal')
                self.height_entry.configure(state='normal')
                self.square_entry.configure(state='normal')
                self.extension_entry.configure(state='normal')
                
                # Show confirm button and hide proceed button
                self.confirm_type_button.grid()
                self.proceed_button.grid_remove()
                
                # Reset confirmation flag
                self.type_confirmed = False
            else:
                # Revert radio button selection if user doesn't want to change
                self.calibration_type_var.set('calculate' if self.calibration_type_var.get() == 'convert' else 'convert')
                return
                
        # Show/hide appropriate frames based on selection
        if self.calibration_type_var.get() == 'calculate':
            self.calculate_options_frame.grid()
            self.convert_frame.grid_remove()
        else:
            self.calculate_options_frame.grid_remove()
            self.convert_frame.grid()
                
    def build_prepare_video_tab(self):
        frame = ctk.CTkFrame(self.prepare_video_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)
    
        # Prepare Video Configuration Heading
        ctk.CTkLabel(frame, text="Prepare Video",
                    font=('Helvetica', 20, 'bold')).pack(pady=(0, 20))
    
        # Only Checkerboard Selection
        checkerboard_frame = ctk.CTkFrame(frame)
        checkerboard_frame.pack(fill='x', pady=10)
        self.only_checkerboard_var = ctk.StringVar(value='yes')
        
        ctk.CTkLabel(checkerboard_frame, text="Do your videos contain only checkerboard images?").pack(side='left', padx=10)
        ctk.CTkRadioButton(checkerboard_frame, text="Yes", variable=self.only_checkerboard_var,
                        value='yes', command=self.on_only_checkerboard_change).pack(side='left', padx=10)
        ctk.CTkRadioButton(checkerboard_frame, text="No", variable=self.only_checkerboard_var,
                        value='no', command=self.on_only_checkerboard_change).pack(side='left', padx=10)
    
        # Time Interval Frame (initially hidden)
        self.time_extraction_frame = ctk.CTkFrame(frame)
        ctk.CTkLabel(self.time_extraction_frame, text="Enter time interval in seconds for image extraction:").pack(side='left', padx=10)
        ctk.CTkEntry(self.time_extraction_frame, textvariable=self.time_interval_var, width=100).pack(side='left', padx=10)
    
        # Extrinsic Format Frame
        format_frame = ctk.CTkFrame(frame)
        format_frame.pack(fill='x', pady=10)
        ctk.CTkLabel(format_frame, text="Enter the image format (e.g., png, jpg):").pack(side='left', padx=10)
        ctk.CTkEntry(format_frame, textvariable=self.extrinsic_format_var, width=100).pack(side='left', padx=10)
    
        # Confirm Button for Yes option
        self.confirm_button = ctk.CTkButton(
            frame,
            text="Confirm",
            command=self.confirm_checkerboard_only,
            width=200,
            height=40
        )
        self.confirm_button.pack(pady=20)
    
        # Proceed Button (initially hidden)
        self.proceed_prepare_video_button = ctk.CTkButton(
            frame,
            text="Proceed with Prepare Video",
            command=self.proceed_prepare_video,
            width=200,
            height=40
        )
        
        # Initially hide the proceed button
        self.proceed_prepare_video_button.pack_forget()
    
    def update_config_toml(self, config_file_path, section=None):
        """
        Update specific sections of the config file without overwriting others.
        section: 'calibration', 'pose', 'synchronization', etc.
        """
        try:
            # Read existing config if it exists
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config = parse(f.read())
            else:
                # If file doesn't exist, start with template
                config = parse(self.config_template)
    
            # Update only specific sections based on what's being configured
            if section == 'calibration':
                calibration_type = self.calibration_type_var.get()
                if 'calibration' not in config:
                    config['calibration'] = {}
                
                config['calibration']['calibration_type'] = calibration_type
                
                if calibration_type == 'calculate':
                    if 'calculate' not in config['calibration']:
                        config['calibration']['calculate'] = {}
                    
                    if 'intrinsics' not in config['calibration']['calculate']:
                        config['calibration']['calculate']['intrinsics'] = {}
                    
                    if 'extrinsics' not in config['calibration']['calculate']:
                        config['calibration']['calculate']['extrinsics'] = {}
                        if 'scene' not in config['calibration']['calculate']['extrinsics']:
                            config['calibration']['calculate']['extrinsics']['scene'] = {}
    
                    # Set extensions based on video preparation choice
                    original_extension = self.video_extension_var.get()
                    intrinsics_extension = "png" if self.change_intrinsics_extension else original_extension
                    
                    config['calibration']['calculate']['intrinsics'].update({
                        'intrinsics_corners_nb': [
                            int(self.checkerboard_width_var.get()),
                            int(self.checkerboard_height_var.get())
                        ],
                        'intrinsics_square_size': float(self.square_size_var.get()),
                        'intrinsics_extension': intrinsics_extension
                    })
                    
                    # Always keep original extension for extrinsics
                    config['calibration']['calculate']['extrinsics']['scene'].update({
                        'extrinsics_extension': original_extension
                    })
    
                else:  # convert
                    if 'convert' not in config['calibration']:
                        config['calibration']['convert'] = {}
                    config['calibration']['convert']['convert_from'] = self.convert_from_var.get()
                    if self.convert_from_var.get() == 'qualisys':
                        if 'qualisys' not in config['calibration']['convert']:
                            config['calibration']['convert']['qualisys'] = {}
                        config['calibration']['convert']['qualisys']['binning_factor'] = int(
                            self.binning_factor_var.get())
                            
                    elif section == 'coordinates':
                        if 'calibration' not in config:
                            config['calibration'] = {}
                        if 'calculate' not in config['calibration']:
                            config['calibration']['calculate'] = {}
                        if 'extrinsics' not in config['calibration']['calculate']:
                            config['calibration']['calculate']['extrinsics'] = {}
                        if 'scene' not in config['calibration']['calculate']['extrinsics']:
                            config['calibration']['calculate']['extrinsics']['scene'] = {}
                            config['calibration']['calculate']['extrinsics']['scene']['object_coords_3d'] = self.object_coords_3d
        
            elif section == 'pose':
                if 'pose' not in config:
                    config['pose'] = {}
                
                config['pose'].update({
                    'vid_img_extension': self.video_extension_var.get(),
                    'pose_model': self.pose_model_var.get()
                })
                
                if self.pose_model_var.get() == 'HALPE_26':
                    config['pose']['mode'] = self.mode_var.get()
    
                # Update project section for participant details
                if 'project' not in config:
                    config['project'] = {}
                
                if self.multiple_persons_var.get() == 'single':
                    config['project'].update({
                        'multi_person': False,
                        'participant_height': float(self.participant_height_var.get()),
                        'participant_mass': float(self.participant_mass_var.get())
                    })
                else:
                    config['project'].update({
                        'multi_person': True,
                        'participant_height': self.participant_heights,
                        'participant_mass': self.participant_masses
                    })
    
            elif section == 'synchronization':
                if self.sync_videos_var.get() == 'no':
                    if 'synchronization' not in config:
                        config['synchronization'] = {}
                    
                    keypoints = self.keypoints_var.get()
                    config['synchronization'].update({
                        'keypoints_to_consider': [keypoints] if keypoints != 'all' else 'all',
                        'time_range_around_maxspeed': float(self.time_range_var.get()),
                        'likelihood_threshold': float(self.likelihood_threshold_var.get()),
                        'filter_cutoff': int(self.filter_cutoff_var.get()),
                        'filter_order': int(self.filter_order_var.get())
                    })
                    
                    if self.approx_time_var.get() == 'yes':
                        config['synchronization']['approx_time_maxspeed'] = [
                            float(entry.get()) for entry in self.approx_time_entries
                        ]
                    else:
                        config['synchronization']['approx_time_maxspeed'] = 'auto'
    
            elif section == 'coordinates':
                if 'calibration' not in config:
                    config['calibration'] = {}
                if 'calculate' not in config['calibration']:
                    config['calibration']['calculate'] = {}
                if 'extrinsics' not in config['calibration']['calculate']:
                    config['calibration']['calculate']['extrinsics'] = {}
                if 'scene' not in config['calibration']['calculate']['extrinsics']:
                    config['calibration']['calculate']['extrinsics']['scene'] = {}
                
                config['calibration']['calculate']['extrinsics']['scene']['object_coords_3d'] = self.object_coords_3d
    
            elif section == 'advanced':
                # Project section
                if 'project' not in config:
                    config['project'] = {}
                
                config['project'].update({
                    'frame_rate': self.frame_rate_var.get()
                })
                
                try:
                    frame_range = ast.literal_eval(self.frame_range_var.get())
                    if isinstance(frame_range, list):
                        config['project']['frame_range'] = frame_range
                except:
                    config['project']['frame_range'] = []
    
                # Person Association section
                if 'personAssociation' not in config:
                    config['personAssociation'] = {}
                
                config['personAssociation']['likelihood_threshold_association'] = float(self.likelihood_threshold_association_var.get())
                
                if 'single_person' not in config['personAssociation']:
                    config['personAssociation']['single_person'] = {}
                
                config['personAssociation']['single_person'].update({
                    'reproj_error_threshold_association': float(self.reproj_error_threshold_association_var.get()),
                    'tracked_keypoint': self.tracked_keypoint_var.get()
                })
    
                # Triangulation section
                if 'triangulation' not in config:
                    config['triangulation'] = {}
                
                config['triangulation'].update({
                    'reproj_error_threshold_triangulation': float(self.reproj_error_threshold_triangulation_var.get()),
                    'likelihood_threshold_triangulation': float(self.likelihood_threshold_triangulation_var.get()),
                    'min_cameras_for_triangulation': int(self.min_cameras_for_triangulation_var.get())
                })
    
                # Filtering section
                if 'filtering' not in config:
                    config['filtering'] = {}
                
                filter_type = self.filter_type_var.get()
                config['filtering']['type'] = filter_type
    
                if filter_type == 'butterworth':
                    if 'butterworth' not in config['filtering']:
                        config['filtering']['butterworth'] = {}
                    config['filtering']['butterworth'].update({
                        'cut_off_frequency': float(self.filter_cutoff_var.get()),
                        'order': int(self.filter_order_var.get())
                    })
                
                elif filter_type == 'kalman':
                    if 'kalman' not in config['filtering']:
                        config['filtering']['kalman'] = {}
                    config['filtering']['kalman'].update({
                        'trust_ratio': float(self.kalman_trust_ratio_var.get()),
                        'smooth': self.kalman_smooth_var.get()
                    })
                
                elif filter_type == 'butterworth_on_speed':
                    if 'butterworth_on_speed' not in config['filtering']:
                        config['filtering']['butterworth_on_speed'] = {}
                    config['filtering']['butterworth_on_speed'].update({
                        'order': int(self.butterworth_on_speed_order_var.get()),
                        'cut_off_frequency': float(self.butterworth_on_speed_cut_off_frequency_var.get())
                    })
                
                elif filter_type == 'gaussian':
                    if 'gaussian' not in config['filtering']:
                        config['filtering']['gaussian'] = {}
                    config['filtering']['gaussian'].update({
                        'sigma_kernel': float(self.gaussian_sigma_kernel_var.get())
                    })
                
                elif filter_type == 'LOESS':
                    if 'LOESS' not in config['filtering']:
                        config['filtering']['LOESS'] = {}
                    config['filtering']['LOESS'].update({
                        'nb_values_used': int(self.LOESS_nb_values_used_var.get())
                    })
                
                elif filter_type == 'median':
                    if 'median' not in config['filtering']:
                        config['filtering']['median'] = {}
                    config['filtering']['median'].update({
                        'kernel_size': int(self.median_kernel_size_var.get())
                    })
    
                # Marker Augmentation section
                if 'markerAugmentation' not in config:
                    config['markerAugmentation'] = {}
                config['markerAugmentation']['make_c3d'] = self.make_c3d_var.get()
    
                # Kinematics section
                if 'kinematics' not in config:
                    config['kinematics'] = {}
                config['kinematics'].update({
                    'use_augmentation': self.use_augmentation_var.get(),
                    'right_left_symmetry': self.right_left_symmetry_var.get(),
                    'remove_individual_scaling_setup': self.remove_individual_scaling_setup_var.get(),
                    'remove_individual_IK_setup': self.remove_individual_IK_setup_var.get()
                })
    
            # Write back to file
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(dumps(config))
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update Config.toml: {str(e)}")
    
    def confirm_checkerboard_only(self):
        """Handle confirmation when 'Yes' is selected for checkerboard-only option"""
        # Keep existing extension for both intrinsics and extrinsics
        self.change_intrinsics_extension = False
        
        # Update Config.toml
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        if self.process_mode == 'batch':
            config_file_path = os.path.join(self.participant_name, 'Config.toml')
        
        self.update_config_toml(config_file_path, section='calibration')
        
        self.update_progress('prepare_video')
        self.update_tab_indicator('prepare_video', saved=True)
        messagebox.showinfo("Complete", "Prepare video step completed.")
        
        # Move to next tab
        next_tab_index = self.tab_order.index('prepare_video') + 1
        if next_tab_index < len(self.tab_order):
            self.notebook.select(next_tab_index)
    
    def proceed_prepare_video(self):
        """Handle video preparation when 'No' is selected"""
        if self.only_checkerboard_var.get() == 'no':
            try:
                time_interval = float(self.time_interval_var.get())
                if time_interval <= 0:
                    raise ValueError
                
                # Set flag to change intrinsics extension to png
                self.change_intrinsics_extension = True
                
                # Update Config.toml
                config_file_path = os.path.join(self.participant_name, 'Config.toml')
                if self.process_mode == 'batch':
                    config_file_path = os.path.join(self.participant_name, 'Config.toml')
                
                self.update_config_toml(config_file_path, section='calibration')
                
                # Disable the Proceed button
                self.proceed_prepare_video_button.configure(state='disabled')
    
                # Start extraction in a separate thread
                extraction_thread = threading.Thread(target=lambda: self.extract_frames(time_interval))
                extraction_thread.start()
                
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid time interval.")
                return
        
    def on_only_checkerboard_change(self):
        if self.only_checkerboard_var.get() == 'no':
            self.time_extraction_frame.pack(fill='x', pady=10)
            self.proceed_prepare_video_button.pack(pady=20)  # Show proceed button
            self.confirm_button.pack_forget()  # Hide confirm button
        else:
            self.time_extraction_frame.pack_forget()
            self.proceed_prepare_video_button.pack_forget()  # Hide proceed button
            self.confirm_button.pack(pady=20)  # Show confirm button
    
  
        
    def extract_frames(self, time_interval):
        base_path = os.path.join(self.participant_name, 'calibration', 'intrinsics')
        if self.process_mode == 'batch':
            base_path = os.path.join(self.participant_name,'calibration', 'intrinsics')
    
        if not os.path.exists(base_path):
            messagebox.showerror("Error", f"Directory '{base_path}' does not exist.")
            self.proceed_prepare_video_button.configure(state='normal')
            return
    
        video_extensions = ('.mp4', '.avi', '.mov', '.mpeg')
        extracted_images = []
        total_videos = 0
    
        # Collect all video files
        video_files = []
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_files.append(os.path.join(root, file))
    
        total_videos = len(video_files)
    
        if not video_files:
            messagebox.showwarning("Warning", "No video files found.")
            self.progress_bar.set(0)
            self.proceed_prepare_video_button.configure(state='normal')
            return
    
        try:
            for idx, video_path in enumerate(video_files):
                video_dir = os.path.dirname(video_path)
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    messagebox.showerror("Error", f"Failed to open video: {video_path}")
                    continue
    
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                interval_frames = int(fps * time_interval)
    
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
    
                    if frame_count % interval_frames == 0:
                        image_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.png"
                        save_path = os.path.join(video_dir, image_name)
                        cv2.imwrite(save_path, frame)
                        extracted_images.append(save_path)
    
                    frame_count += 1
    
                cap.release()
                
              
                # Update progress to 50%
                progress = ((idx + 1) / total_videos)
                self.progress_bar.set(progress)
                self.root.update_idletasks()
    
            if extracted_images:
                self.review_extracted_images(extracted_images, base_path)
            else:
                messagebox.showinfo("Complete", "No frames were extracted.")
    
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.progress_bar.set(0)
            self.proceed_prepare_video_button.configure(state='normal')
            
        
    def review_extracted_images(self, image_paths, base_path):
        images_by_camera = {}
        for img_path in image_paths:
            camera_dir = os.path.basename(os.path.dirname(img_path))
            if camera_dir not in images_by_camera:
                images_by_camera[camera_dir] = []
            images_by_camera[camera_dir].append(img_path)
    
        self.camera_image_list = list(images_by_camera.items())
        self.current_camera_index = 0
    
        if self.camera_image_list:
            camera_dir, imgs = self.camera_image_list[self.current_camera_index]
            self.process_camera_images(camera_dir, imgs)
        else:
            messagebox.showinfo("Complete", "No images to review.")    
        
    def process_camera_images(self, camera_dir, image_paths):
        """
        Creates a review window for a specific camera's images, allowing the user to keep or delete images.
        After processing, it moves to the next camera automatically.
        """
        review_window = ctk.CTkToplevel(self.root)
        review_window.title(f"Review Extracted Images - {camera_dir}")
        review_window.geometry("1000x700")
    
        # Create a main frame
        main_frame = ctk.CTkFrame(review_window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Create scrollable frame
        scrollable_frame = ctk.CTkScrollableFrame(main_frame)
        scrollable_frame.pack(fill="both", expand=True)
    
        # List to hold image references and their corresponding variables
        self.image_vars = []
    
        # Define number of columns in the grid
        num_columns = 4
        padding = 10
    
        # Create frames for images
        for idx, img_path in enumerate(image_paths):
            # Create a frame for each image
            img_frame = ctk.CTkFrame(scrollable_frame)
            img_frame.grid(row=idx//num_columns, column=idx%num_columns, padx=padding, pady=padding)
    
            try:
                # Load and resize the image
                img = Image.open(img_path)
                img.thumbnail((200, 200))  # Resize for display
                ctk_img = CTkImage(dark_image=img, light_image=img, size=(200, 200))
                
                # Create a label to display the image
                img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                img_label.image = ctk_img  # Keep a reference
                img_label.pack(padx=5, pady=5)
    
                # Checkbox to keep the image (unchecked by default)
                var = ctk.BooleanVar(value=False)
                checkbox = ctk.CTkCheckBox(img_frame, text="Keep", variable=var)
                checkbox.pack(pady=(0, 5))
    
                # Store the variable and path
                self.image_vars.append({'var': var, 'path': img_path})
    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image {img_path}: {str(e)}")
                continue
    
        # Function to handle the deletion of unselected images and proceed to next camera
        def delete_unselected_images():
            to_delete = [img['path'] for img in self.image_vars if not img['var'].get()]
            if not to_delete:
                confirm = messagebox.askyesno("No Deletion", "No images selected for deletion. Do you want to proceed?")
                if not confirm:
                    return
            else:
                confirm = messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete {len(to_delete)} unselected images?")
                if confirm:
                    for img_path in to_delete:
                        try:
                            os.remove(img_path)
                        except Exception as e:
                            messagebox.showerror("Deletion Error", f"Failed to delete {img_path}: {str(e)}")
                    messagebox.showinfo("Deletion Complete", f"Deleted {len(to_delete)} images.")
    
            # Close the current review window
            review_window.destroy()
    
            # Move to the next camera
            self.current_camera_index += 1
            if self.current_camera_index < len(self.camera_image_list):
                next_camera_dir, next_imgs = self.camera_image_list[self.current_camera_index]
                self.process_camera_images(next_camera_dir, next_imgs)
            else:
                # All cameras have been processed message
                messagebox.showinfo("Process Completed", "All images have been reviewed and processed.")
                self.update_tab_indicator('prepare_video', saved=True)
                self.progress_bar.set(1.0)  # Set to 100% when complete
                self.root.update_idletasks()
    
        # Button to delete unselected images
        delete_button = ctk.CTkButton(
            review_window,
            text='Delete unselected images',
            command=delete_unselected_images,
            fg_color='#f44336',  # Red color for delete
            text_color='white',
            width=200,
            height=40
        )
        delete_button.pack(pady=10)
        
        self.update_progress('prepare_video')
    def build_pose_model_tab(self):
        """
        Build the Pose Estimation tab.
        """
        frame = ctk.CTkFrame(self.pose_model_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Pose Estimation Configuration Heading
        ctk.CTkLabel(frame, text="Pose Estimation",
                    font=('Helvetica', 20, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky='w')

        # Ask if videos have multiple persons
        ctk.CTkLabel(frame, text="Multiple Persons:").grid(
            row=1, column=0, sticky='w', pady=5)
        multi_person_frame = ctk.CTkFrame(frame, fg_color="transparent")
        multi_person_frame.grid(row=1, column=1, sticky='w')
        self.multiple_persons_var = ctk.StringVar(value='single')  # Default to single
        ctk.CTkRadioButton(multi_person_frame, text="Single Person", variable=self.multiple_persons_var, value='single',
                           command=self.on_multiple_persons_change).pack(side='left', padx=10)
        ctk.CTkRadioButton(multi_person_frame, text="Multiple Persons", variable=self.multiple_persons_var, value='multiple',
                           command=self.on_multiple_persons_change).pack(side='left', padx=10)

        # Frame for single or multiple person inputs
        self.person_details_frame = ctk.CTkFrame(frame)
        self.person_details_frame.grid(row=2, column=0, columnspan=2, sticky='w', pady=10)

        # Initially show single person inputs
        self.build_single_person_inputs()

        # Pose Model Selection
        ctk.CTkLabel(frame, text="Pose Model Selection:").grid(
            row=3, column=0, sticky='w', pady=5)
        pose_model_options = [
            'BODY_25B', 'BODY_135', 'COCO_133', 'COCO_17', 'MPII', 'HALPE_26', 'CUSTOM'
        ]
        self.pose_model_menu = ctk.CTkOptionMenu(
            frame, variable=self.pose_model_var, values=pose_model_options, width=150, command=self.on_pose_model_change)
        self.pose_model_menu.grid(row=3, column=1, sticky='w')

        # Mode Selection
        self.mode_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ctk.CTkLabel(self.mode_frame, text="Mode:").grid(row=0, column=0, sticky='w')
        mode_options = ['lightweight', 'balanced', 'performance']
        self.mode_menu = ctk.CTkOptionMenu(self.mode_frame, variable=self.mode_var,
                                          values=mode_options, width=150)
        self.mode_menu.grid(row=0, column=1, sticky='w')
        self.mode_frame.grid(row=4, column=0, columnspan=2, sticky='w', pady=5)

        # Video Extension (for pose estimation)
        ctk.CTkLabel(frame, text="Video/Image Extension (e.g., mp4, png):").grid(
            row=5, column=0, sticky='w', pady=5)
        ctk.CTkEntry(frame, textvariable=self.video_extension_var, width=100).grid(
            row=5, column=1, sticky='w')

        # Proceed Button
        ctk.CTkButton(frame, text="Proceed with Pose Estimation",
                      command=self.proceed_pose_estimation, width=200, height=40).grid(
            row=6, column=0, columnspan=2, pady=20)

        # Adjust the pose model change
        self.on_pose_model_change(self.pose_model_var.get())

    def build_synchronization_tab(self):
        """
        Build the Synchronization tab.
        """
        frame = ctk.CTkFrame(self.synchronization_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)

        ctk.CTkLabel(frame, text="Synchronization",
                    font=('Helvetica', 20, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky='w')

        # Ask if videos are already synchronized
        ctk.CTkLabel(frame, text="Skip synchronization part? (Videos are already synchronized)").grid(
            row=1, column=0, sticky='w', pady=5)
        ctk.CTkRadioButton(frame, text="Yes", variable=self.sync_videos_var, value='yes',
                           command=self.on_sync_option_change).grid(
            row=1, column=1, sticky='w')
        ctk.CTkRadioButton(frame, text="No", variable=self.sync_videos_var, value='no',
                           command=self.on_sync_option_change).grid(
            row=1, column=2, sticky='w')

        # Frame for synchronization options
        self.sync_options_frame = ctk.CTkFrame(frame)
        self.sync_options_frame.grid(row=2, column=0, columnspan=3, sticky='w')

        # Keypoints to consider
        ctk.CTkLabel(self.sync_options_frame, text="Select keypoints to consider for synchronization:").grid(
            row=0, column=0, sticky='w', pady=5)
        keypoints_options = ['all', 'CHip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe', 'RSmallToe', 'RHeel',
                             'LHip', 'LKnee', 'LAnkle', 'LBigToe', 'LSmallToe', 'LHeel', 'Neck', 'Head',
                             'Nose', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'Custom']
        self.keypoints_menu = ctk.CTkOptionMenu(
            self.sync_options_frame, variable=self.keypoints_var, values=keypoints_options, width=200)
        self.keypoints_menu.grid(row=0, column=1, sticky='w')

        # Approximate time of movement
        ctk.CTkLabel(self.sync_options_frame, text="Do you want to specify approximate times of movement? (Recommended)").grid(
            row=1, column=0, sticky='w', pady=5)
        ctk.CTkRadioButton(self.sync_options_frame, text="Yes", variable=self.approx_time_var, value='yes',
                           command=self.on_approx_time_change).grid(
            row=1, column=1, sticky='w')
        ctk.CTkRadioButton(self.sync_options_frame, text="Auto", variable=self.approx_time_var, value='auto',
                           command=self.on_approx_time_change).grid(
            row=1, column=2, sticky='w')
        # After the approx_time_var radio buttons, add:
        self.approx_time_frame = ctk.CTkFrame(self.sync_options_frame)
        self.approx_time_frame.grid(row=2, column=0, columnspan=3, sticky='nsew', pady=5)
        self.approx_time_frame.grid_remove()  # Initially hidden
        # Time range around max speed
        ctk.CTkLabel(self.sync_options_frame, text="Time interval around max speed (seconds):").grid(
            row=3, column=0, sticky='w', pady=5)
        ctk.CTkEntry(self.sync_options_frame, textvariable=self.time_range_var, width=100).grid(
            row=3, column=1, sticky='w')

        # Likelihood threshold
        ctk.CTkLabel(self.sync_options_frame, text="Likelihood Threshold:").grid(
            row=4, column=0, sticky='w', pady=5)
        ctk.CTkEntry(self.sync_options_frame, textvariable=self.likelihood_threshold_var, width=100).grid(
            row=4, column=1, sticky='w')

        # Filter cutoff
        ctk.CTkLabel(self.sync_options_frame, text="Filter Cutoff (Hz):").grid(
            row=5, column=0, sticky='w', pady=5)
        ctk.CTkEntry(self.sync_options_frame, textvariable=self.filter_cutoff_var, width=100).grid(
            row=5, column=1, sticky='w')

        # Filter order
        ctk.CTkLabel(self.sync_options_frame, text="Filter Order:").grid(
            row=6, column=0, sticky='w', pady=5)
        ctk.CTkEntry(self.sync_options_frame, textvariable=self.filter_order_var, width=100).grid(
            row=6, column=1, sticky='w')

        # Save Button
        ctk.CTkButton(self.sync_options_frame, text="Save Synchronization Settings",
                      command=self.save_synchronization_settings, width=200, height=40).grid(row=7, column=0, columnspan=3, pady=10)

        self.on_sync_option_change()

    

    def build_advanced_configuration_tab(self):
        frame = ctk.CTkFrame(self.advanced_configuration_frame)
        frame.pack(expand=True, fill='both', padx=20, pady=20)
    
        # Header
        header_frame = ctk.CTkFrame(frame)
        header_frame.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(header_frame, 
                    text='Advanced Configuration', 
                    font=('Helvetica', 20, 'bold')).pack(anchor='w')
    
        # Main content scrollable frame
        content_frame = ctk.CTkScrollableFrame(frame)
        content_frame.pack(fill='both', expand=True)
    
        # Frame Rate and Frame Range
        basic_frame = ctk.CTkFrame(content_frame)
        basic_frame.pack(fill='x', pady=10, padx=10)
        
        basic_label = ctk.CTkFrame(basic_frame)
        basic_label.pack(fill='x', pady=5)
        ctk.CTkLabel(basic_label, text='Basic Settings', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        frame_rate_frame = ctk.CTkFrame(basic_frame)
        frame_rate_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(frame_rate_frame, text='Frame Rate:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(frame_rate_frame, textvariable=self.frame_rate_var, width=150).pack(side='left', padx=5)
        
        frame_range_frame = ctk.CTkFrame(basic_frame)
        frame_range_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(frame_range_frame, text='Frame Range:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(frame_range_frame, textvariable=self.frame_range_var, width=150).pack(side='left', padx=5)
    
        # Person Association Section
        person_assoc_frame = ctk.CTkFrame(content_frame)
        person_assoc_frame.pack(fill='x', pady=10, padx=10)
        
        pa_label = ctk.CTkFrame(person_assoc_frame)
        pa_label.pack(fill='x', pady=5)
        ctk.CTkLabel(pa_label, text='Person Association', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        likelihood_frame = ctk.CTkFrame(person_assoc_frame)
        likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(likelihood_frame, text='Likelihood Threshold:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(likelihood_frame, textvariable=self.likelihood_threshold_association_var, width=150).pack(side='left', padx=5)
        
        reproj_frame = ctk.CTkFrame(person_assoc_frame)
        reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(reproj_frame, text='Reprojection Error Threshold:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(reproj_frame, textvariable=self.reproj_error_threshold_association_var, width=150).pack(side='left', padx=5)
        
        tracked_frame = ctk.CTkFrame(person_assoc_frame)
        tracked_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tracked_frame, text='Tracked Keypoint:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(tracked_frame, textvariable=self.tracked_keypoint_var, width=150).pack(side='left', padx=5)
    
        # Triangulation Section
        triangulation_frame = ctk.CTkFrame(content_frame)
        triangulation_frame.pack(fill='x', pady=10, padx=10)
        
        tri_label = ctk.CTkFrame(triangulation_frame)
        tri_label.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_label, text='Triangulation', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        tri_reproj_frame = ctk.CTkFrame(triangulation_frame)
        tri_reproj_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_reproj_frame, text='Reprojection Error Threshold:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(tri_reproj_frame, textvariable=self.reproj_error_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        tri_likelihood_frame = ctk.CTkFrame(triangulation_frame)
        tri_likelihood_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(tri_likelihood_frame, text='Likelihood Threshold:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(tri_likelihood_frame, textvariable=self.likelihood_threshold_triangulation_var, width=150).pack(side='left', padx=5)
        
        min_cameras_frame = ctk.CTkFrame(triangulation_frame)
        min_cameras_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(min_cameras_frame, text='Minimum Cameras:', width=200).pack(side='left', padx=5)
        ctk.CTkEntry(min_cameras_frame, textvariable=self.min_cameras_for_triangulation_var, width=150).pack(side='left', padx=5)
    
        # Filtering Section
        filtering_frame = ctk.CTkFrame(content_frame)
        filtering_frame.pack(fill='x', pady=10, padx=10)
        
        filter_label = ctk.CTkFrame(filtering_frame)
        filter_label.pack(fill='x', pady=5)
        ctk.CTkLabel(filter_label, text='Filtering', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        filter_type_frame = ctk.CTkFrame(filtering_frame)
        filter_type_frame.pack(fill='x', pady=5)
        ctk.CTkLabel(filter_type_frame, text='Filter Type:', width=200).pack(side='left', padx=5)
        filtering_options = ['butterworth', 'kalman', 'gaussian', 'LOESS', 'median', 'butterworth_on_speed']
        self.filter_type_menu = ctk.CTkOptionMenu(
            filter_type_frame, 
            variable=self.filter_type_var, 
            values=filtering_options, 
            width=150,
            command=self.on_filter_type_change
        )
        self.filter_type_menu.pack(side='left', padx=5)
        
        # Frame for filter-specific parameters
        self.filter_specific_frame = ctk.CTkFrame(filtering_frame)
        self.filter_specific_frame.pack(fill='x', pady=5)
        
        # Initialize filter-specific parameters
        self.on_filter_type_change(self.filter_type_var.get())
    
        # Marker Augmentation Section
        marker_frame = ctk.CTkFrame(content_frame)
        marker_frame.pack(fill='x', pady=10, padx=10)
        
        marker_label = ctk.CTkFrame(marker_frame)
        marker_label.pack(fill='x', pady=5)
        ctk.CTkLabel(marker_label, text='Marker Augmentation', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        make_c3d_check = ctk.CTkCheckBox(
            marker_frame,
            text="Make C3D",
            variable=self.make_c3d_var,
            onvalue=True,
            offvalue=False
        )
        make_c3d_check.pack(pady=5)
    
        # Kinematics Section
        kinematics_frame = ctk.CTkFrame(content_frame)
        kinematics_frame.pack(fill='x', pady=10, padx=10)
        
        kin_label = ctk.CTkFrame(kinematics_frame)
        kin_label.pack(fill='x', pady=5)
        ctk.CTkLabel(kin_label, text='Kinematics', font=('Helvetica', 16, 'bold')).pack(anchor='w')
        
        use_augmentation_check = ctk.CTkCheckBox(
            kinematics_frame,
            text="Use Augmentation",
            variable=self.use_augmentation_var,
            onvalue=True,
            offvalue=False
        )
        use_augmentation_check.pack(pady=5)
    
        right_left_symmetry_check = ctk.CTkCheckBox(
            kinematics_frame,
            text="Right-Left Symmetry",
            variable=self.right_left_symmetry_var,
            onvalue=True,
            offvalue=False
        )
        right_left_symmetry_check.pack(pady=5)
    
        remove_scaling_check = ctk.CTkCheckBox(
            kinematics_frame,
            text="Remove Individual Scaling",
            variable=self.remove_individual_scaling_setup_var,
            onvalue=True,
            offvalue=False
        )
        remove_scaling_check.pack(pady=5)
    
        remove_ik_check = ctk.CTkCheckBox(
            kinematics_frame,
            text="Remove Individual IK Setup",
            variable=self.remove_individual_IK_setup_var,
            onvalue=True,
            offvalue=False
        )
        remove_ik_check.pack(pady=5)
        
        # Save Button at the bottom
        save_button = ctk.CTkButton(
            frame,
            text='Save Advanced Settings',
            command=self.save_advanced_settings,
            height=40
        )
        save_button.pack(pady=20)
    
    def on_filter_type_change(self, selected_filter):
        # Clear existing widgets in filter_specific_frame
        for widget in self.filter_specific_frame.winfo_children():
            widget.destroy()

        if selected_filter == 'butterworth':
            # Butterworth filter parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Cutoff Frequency (Hz):').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.filter_cutoff_var, width=100).grid(row=0, column=1, sticky='w')

            ctk.CTkLabel(self.filter_specific_frame, text='Filter Order:').grid(row=1, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.filter_order_var, width=100).grid(row=1, column=1, sticky='w')

        elif selected_filter == 'kalman':
            # Kalman filter parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Trust Ratio:').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.kalman_trust_ratio_var, width=100).grid(row=0, column=1, sticky='w')

            ctk.CTkLabel(self.filter_specific_frame, text='Smooth:').grid(row=1, column=0, sticky='w', pady=2)
            ctk.CTkCheckBox(self.filter_specific_frame, variable=self.kalman_smooth_var).grid(row=1, column=1, sticky='w')

        elif selected_filter == 'butterworth_on_speed':
            # Butterworth on Speed parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Order:').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.butterworth_on_speed_order_var, width=100).grid(row=0, column=1, sticky='w')

            ctk.CTkLabel(self.filter_specific_frame, text='Cut-off Frequency (Hz):').grid(row=1, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.butterworth_on_speed_cut_off_frequency_var, width=100).grid(row=1, column=1, sticky='w')

        elif selected_filter == 'gaussian':
            # Gaussian filter parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Sigma Kernel (px):').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.gaussian_sigma_kernel_var, width=100).grid(row=0, column=1, sticky='w')

        elif selected_filter == 'LOESS':
            # LOESS filter parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Number of Values Used:').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.LOESS_nb_values_used_var, width=100).grid(row=0, column=1, sticky='w')

        elif selected_filter == 'median':
            # Median filter parameters
            ctk.CTkLabel(self.filter_specific_frame, text='Kernel Size:').grid(row=0, column=0, sticky='w', pady=2)
            ctk.CTkEntry(self.filter_specific_frame, textvariable=self.median_kernel_size_var, width=100).grid(row=0, column=1, sticky='w')

        else:
            # Handle unexpected filter types
            ctk.CTkLabel(self.filter_specific_frame, text='No additional parameters.').grid(row=0, column=0, sticky='w', pady=2)

    def on_multiple_persons_change(self):
        # Clear current person details
        for widget in self.person_details_frame.winfo_children():
            widget.destroy()

        if self.multiple_persons_var.get() == 'single':
            self.build_single_person_inputs()
        else:
            self.build_multiple_persons_inputs()

    def build_single_person_inputs(self):
        # Single person inputs for height and mass
        ctk.CTkLabel(self.person_details_frame, text="Participant Height (m):").grid(
            row=0, column=0, sticky='w', pady=2)
        ctk.CTkEntry(self.person_details_frame, textvariable=self.participant_height_var, width=100).grid(
            row=0, column=1, sticky='w')

        ctk.CTkLabel(self.person_details_frame, text="Participant Mass (kg):").grid(
            row=1, column=0, sticky='w', pady=2)
        ctk.CTkEntry(self.person_details_frame, textvariable=self.participant_mass_var, width=100).grid(
            row=1, column=1, sticky='w')

    def build_multiple_persons_inputs(self):
        # Ask for the number of people of interest
        ctk.CTkLabel(self.person_details_frame, text="Number of People:").grid(
            row=0, column=0, sticky='w', pady=2)
        ctk.CTkEntry(self.person_details_frame, textvariable=self.num_people_var, width=100).grid(
            row=0, column=1, sticky='w')
        ctk.CTkButton(self.person_details_frame, text="Submit",
                     command=self.build_people_details_inputs, width=100, height=30).grid(
            row=0, column=2, padx=5)

    def build_people_details_inputs(self):
        try:
            num_people = int(self.num_people_var.get())
            if num_people < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Invalid number of people entered.")
            return

        # Clear previous widgets
        for widget in self.person_details_frame.winfo_children():
            widget.destroy()

        self.people_details_vars = []
        for i in range(num_people):
            # Participant Height
            height_label_text = f"Participant {i+1} Height (m):"
            ctk.CTkLabel(self.person_details_frame, text=height_label_text).grid(
                row=i*2, column=0, sticky='w', pady=2)
            height_var = ctk.DoubleVar()
            height_entry = ctk.CTkEntry(self.person_details_frame, textvariable=height_var, width=100)
            height_entry.grid(row=i*2, column=1, sticky='w')

            # Participant Mass
            mass_label_text = f"Participant {i+1} Mass (kg):"
            ctk.CTkLabel(self.person_details_frame, text=mass_label_text).grid(
                row=i*2+1, column=0, sticky='w', pady=2)
            mass_var = ctk.DoubleVar()
            mass_entry = ctk.CTkEntry(self.person_details_frame, textvariable=mass_var, width=100)
            mass_entry.grid(row=i*2+1, column=1, sticky='w')

            self.people_details_vars.append((height_var, mass_var))

        # Add an explicit Submit button
        submit_button = ctk.CTkButton(self.person_details_frame, text="Submit",
                                command=self.submit_people_details, width=100, height=30)
        submit_button.grid(row=num_people*2, column=0, columnspan=2, pady=10)

    def submit_people_details(self, event=None):
        heights = []
        masses = []
    
        for height_var, mass_var in self.people_details_vars:
            try:
                # Get and strip the input values
                height_input = str(height_var.get()).strip()
                mass_input = str(mass_var.get()).strip()
    
                # Convert input to float
                height = float(height_input)
                mass = float(mass_input)
    
                # Ensure valid non-negative values
                if height <= 0 or mass <= 0:
                    raise ValueError("Height and mass must be positive numbers.")
    
                # Append to lists
                heights.append(height)
                masses.append(mass)
            except ValueError as e:
                messagebox.showerror("Input Error", f"Invalid height or mass entered: {e}")
                return
    
        # Store the values
        self.participant_heights = heights
        self.participant_masses = masses
    
        # Update Config.toml
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        if self.process_mode == 'batch':
            config_file_path = os.path.join(self.participant_name, 'Config.toml')
        
        try:
            # Read existing config
            if os.path.exists(config_file_path):
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config = parse(f.read())
            else:
                config = parse(self.config_template)
    
            # Update project section
            if 'project' not in config:
                config['project'] = {}
            
            config['project'].update({
                'multi_person': True,
                'participant_height': self.participant_heights,  # Save as list
                'participant_mass': self.participant_masses      # Save as list
            })
    
            # Write back to file
            with open(config_file_path, 'w', encoding='utf-8') as f:
                f.write(dumps(config))
    
            # Show success message
            messagebox.showinfo("Submitted", "Participant details have been saved successfully.")
    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def save_synchronization_settings(self):
        # Update the synchronization settings in the Config.toml file
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        self.update_config_toml(config_file_path, section='synchronization')
        self.update_progress('synchronization')
        messagebox.showinfo("Saved", "Synchronization settings saved.")
        # Update the Synchronization tab indicator to green
        self.update_tab_indicator('synchronization', saved=True)

    def save_advanced_settings(self):
        # Update the advanced settings in the Config.toml file
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        self.update_config_toml(config_file_path, section='advanced')
        messagebox.showinfo("Saved", "Advanced Configuration settings saved.")
        # Update the Advanced Configuration tab indicator to green
        self.update_tab_indicator('advanced_configuration', saved=True)

   

    def on_pose_model_change(self, value):
        if value == 'HALPE_26':
            self.mode_frame.grid()
        else:
            self.mode_frame.grid_remove()

    def on_sync_option_change(self):
        if self.sync_videos_var.get() == 'yes':
            self.sync_options_frame.grid_remove()
            # Set Synchronization tab indicator to green since synchronization is skipped
            self.update_tab_indicator('synchronization', saved=True)
        else:
            self.sync_options_frame.grid()
            # Set Synchronization tab indicator to red since synchronization needs to be configured
            self.update_tab_indicator('synchronization', saved=False)

    def on_approx_time_change(self):
        """Handle time input for each camera when 'yes' is selected"""
        # Clear previous widgets
        for widget in self.approx_time_frame.winfo_children():
            widget.destroy()
    
        if self.approx_time_var.get() == 'yes':
            # Show the frame
            self.approx_time_frame.grid()
            
            # Create scrollable frame for camera times
            time_scroll_frame = ctk.CTkScrollableFrame(self.approx_time_frame, width=400, height=200)
            time_scroll_frame.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=10, pady=5)
            
            # Header label
            ctk.CTkLabel(time_scroll_frame, 
                        text="Enter approximate times (in seconds) of movement for each camera:",
                        wraplength=380).grid(row=0, column=0, columnspan=2, pady=(0,10), sticky='w')
            
            # Create entry for each camera
            self.approx_time_entries = []
            num_cameras = int(self.num_cameras_var.get()) if self.num_cameras_var.get() else 0
            
            for cam in range(1, num_cameras + 1):
                # Frame for each camera's input
                cam_frame = ctk.CTkFrame(time_scroll_frame)
                cam_frame.grid(row=cam, column=0, columnspan=2, sticky='ew', pady=2)
                
                # Label and entry
                ctk.CTkLabel(cam_frame, text=f"Camera {cam}:", width=100).pack(side='left', padx=5)
                time_var = ctk.StringVar(value="0.0")
                entry = ctk.CTkEntry(cam_frame, textvariable=time_var, width=100)
                entry.pack(side='left', padx=5)
                self.approx_time_entries.append(time_var)
            
            # Add confirmation button
            confirm_button = ctk.CTkButton(
                self.approx_time_frame,
                text="Confirm Times",
                command=self.confirm_camera_times,
                width=150
            )
            confirm_button.grid(row=1, column=0, columnspan=2, pady=10)
            
        else:
            # Hide the frame in auto mode
            self.approx_time_frame.grid_remove()
    
    def confirm_camera_times(self):
        """Validate and save camera times"""
        try:
            # Validate all entries are valid numbers
            times = []
            for i, time_var in enumerate(self.approx_time_entries, 1):
                try:
                    time_value = float(time_var.get())
                    if time_value < 0:
                        raise ValueError
                    times.append(time_value)
                except ValueError:
                    messagebox.showerror("Error", f"Invalid time value for Camera {i}")
                    return
    
            # Update Config.toml
            config_file_path = os.path.join(self.participant_name, 'Config.toml')
            try:
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config = parse(f.read())
                
                # Update synchronization section
                if 'synchronization' not in config:
                    config['synchronization'] = {}
                
                config['synchronization']['approx_time_maxspeed'] = times
                
                # Write back to file
                with open(config_file_path, 'w', encoding='utf-8') as f:
                    f.write(dumps(config))
                
                # Show success message
                messagebox.showinfo("Success", "Camera times have been saved successfully")
                
                # Disable entries after confirmation
                for entry in self.approx_time_frame.winfo_children():
                    if isinstance(entry, ctk.CTkEntry):
                        entry.configure(state='disabled')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update configuration: {str(e)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
    def proceed_pose_estimation(self):
        """
        Handle pose estimation configuration and video input for both single and batch modes.
        Includes validation, error handling, and proper UI updates.
        """
        try:
            # Initial validation
            if not self.video_extension_var.get():
                messagebox.showerror("Error", "Please specify a video extension.")
                return
                
            if self.multiple_persons_var.get() == 'multiple':
                if not hasattr(self, 'participant_heights') or not hasattr(self, 'participant_masses'):
                    messagebox.showerror("Error", 
                        "Please enter participant details for multiple persons or switch back to single person mode.")
                    return
            else:
                # Validate single person inputs
                try:
                    height = float(self.participant_height_var.get())
                    mass = float(self.participant_mass_var.get())
                    if height <= 0 or mass <= 0:
                        raise ValueError("Height and mass must be positive numbers.")
                except ValueError as ve:
                    messagebox.showerror("Input Error", 
                        f"Please enter valid height and mass values: {str(ve)}")
                    return
    
            # Handle single mode
            if self.process_mode == 'single':
                # Create videos directory
                target_path = os.path.join(self.participant_name, 'videos')
                os.makedirs(target_path, exist_ok=True)
    
                # Check if directory already has videos
                existing_videos = [f for f in os.listdir(target_path) 
                                if f.endswith(self.video_extension_var.get())]
                if existing_videos:
                    response = messagebox.askyesno("Existing Videos",
                        "Existing videos found. Do you want to replace them?")
                    if response:
                        for video in existing_videos:
                            try:
                                os.remove(os.path.join(target_path, video))
                            except Exception as e:
                                messagebox.showerror("Error", 
                                    f"Could not remove existing video {video}: {str(e)}")
                                return
                    else:
                        return
    
                # Input new videos
                if not self.input_videos(target_path):
                    return
    
                # Update configuration
                config_file_path = os.path.join(self.participant_name, 'Config.toml')
                self.update_config_toml(config_file_path, section='pose')
    
                # Show completion message and update UI
                messagebox.showinfo("Pose Estimation Complete",
                                "Pose estimation configuration updated successfully.")
                self.update_tab_indicator('pose_model', saved=True)
                self.update_progress('pose')
    
            # Handle batch mode
            else:
                try:
                    # Update parent configuration first
                    parent_config_path = os.path.join(self.participant_name, 'Config.toml')
                    self.update_config_toml(parent_config_path, section='pose')
    
                    # Process each trial
                    for trial in range(1, self.num_trials + 1):
                        # Create trial directory structure
                        trial_path = os.path.join(self.participant_name, f'Trial_{trial}')
                        videos_path = os.path.join(trial_path, 'videos')
                        os.makedirs(videos_path, exist_ok=True)
    
                        # Check for existing videos in trial
                        existing_videos = [f for f in os.listdir(videos_path) 
                                        if f.endswith(self.video_extension_var.get())]
                        if existing_videos:
                            response = messagebox.askyesno("Existing Videos",
                                f"Existing videos found in Trial_{trial}. Do you want to replace them?")
                            if response:
                                for video in existing_videos:
                                    try:
                                        os.remove(os.path.join(videos_path, video))
                                    except Exception as e:
                                        messagebox.showerror("Error", 
                                            f"Could not remove existing video {video} in Trial_{trial}: {str(e)}")
                                        return
                            else:
                                continue
    
                        # Prompt for video selection
                        messagebox.showinfo("Trial Input", 
                            f"Please select videos for Trial {trial}")
    
                        # Input videos for this trial
                        if not self.input_videos(videos_path):
                            return
    
                    # Finalize batch configuration
                    self.finalize_configuration()
    
                    # Show completion message and update UI
                    messagebox.showinfo("Pose Estimation Complete",
                                    "Pose estimation configuration updated successfully for all trials.")
                    self.update_tab_indicator('pose_model', saved=True)
                    self.update_progress('pose')
    
                except Exception as e:
                    messagebox.showerror("Batch Processing Error",
                        f"Error during batch processing: {str(e)}")
                    return
    
        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {str(ve)}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
        finally:
            # Ensure progress bar is updated even if an error occurs
            self.update_progress('pose')
        
    def update_progress(self, section):
        """
        Update the progress bar and label
        section: The section that was just completed
        """
        if section in self.progress_steps:
            progress = self.progress_steps[section] / 100
            self.progress_var.set(progress)
            self.progress_bar.set(progress)
            
            # Determine color based on progress
            if progress <= 0.25:
                color = "#1f538d"  # Light red
            elif progress <= 0.50:
                color = "#1f538d"  # Light orange
            elif progress <= 0.75:
                color = "#1f538d"  # Light yellow
            else:
                color = "#1f538d"  # colors to change later now all blue 
                
            self.progress_bar.configure(progress_color=color)
            
            if section == 'calibration':
                status_text = f"Progress: 25% - Calibration completed"
            elif section == 'prepare_video':
                status_text = f"Progress: 50% - Video preparation completed"
            elif section == 'pose':
                status_text = f"Progress: 75% - Pose estimation configured"
            elif section == 'synchronization':
                status_text = f"Progress: 100% - Configuration complete, ready to launch Pose2Sim"
            else:
                status_text = f"Progress: {int(progress * 100)}%"
            
            self.progress_label.configure(text=status_text)
    def activate_pose2sim(self):
        # Determine if synchronization should be enabled based on the user's choice
        do_synchronization = self.sync_videos_var.get().lower() == 'no'

        # Generate the Python code to run Pose2Sim with the appropriate settings
        python_script_content = f"""
from Pose2Sim import Pose2Sim
Pose2Sim.runAll(do_calibration=True, 
                do_poseEstimation=True, 
                do_synchronization={do_synchronization}, 
                do_personAssociation=True, 
                do_triangulation=True, 
                do_filtering=True, 
                do_markerAugmentation=True, 
                do_kinematics=True)
"""

        # Save the Python script to the participant's directory
        script_path = os.path.join(self.participant_name, 'run_pose2sim.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(python_script_content)

        # Generate the CMD script to activate Conda environment and run the Python script
        cmd_script_content = f"""
@echo off
setlocal EnableDelayedExpansion

REM Activate Conda environment
call conda activate Pose2Sim

REM Change to the specified directory
cd "{os.path.abspath(self.participant_name)}"

REM Launch the Python script and keep the command prompt open
python run_pose2sim.py

REM Pause the command prompt to prevent it from closing
pause

endlocal
"""

        cmd_script_path = os.path.join(self.participant_name, 'activate_pose2sim.cmd')
        with open(cmd_script_path, 'w', encoding='utf-8') as f:
            f.write(cmd_script_content)

        # Run the CMD script
        os.system(f'start cmd /k "{cmd_script_path}"')

    def generate_checkerboard_image(self, checkerboard_width, checkerboard_height, square_size):
        """
        Generates a full checkerboard image including the outer borders.
        """
        # Increase the number of squares by 1 to include the outer borders
        num_rows = checkerboard_height + 1
        num_cols = checkerboard_width + 1
        square_size = int(square_size)  # Ensure square size is an integer

        # Create a black and white checkerboard image
        pattern = np.zeros(
            (num_rows * square_size, num_cols * square_size), dtype=np.uint8)
        for row in range(num_rows):
            for col in range(num_cols):
                if (row + col) % 2 == 0:
                    pattern[row*square_size:(row+1)*square_size,
                            col*square_size:(col+1)*square_size] = 255  # White square

        # Convert to PIL Image
        image = Image.fromarray(pattern)

        return image

    def display_checkerboard_image(self, image):
        # Resize image for display if it's too large
        max_size = 300
        img_width, img_height = image.size
        scale = min(max_size / img_width, max_size / img_height, 1)
        display_image = image.resize(
            (int(img_width * scale), int(img_height * scale)), Image.Resampling.LANCZOS)

        # Convert the image to CTkImage
        ctki = CTkImage(dark_image=display_image, light_image=display_image, size=(int(img_width * scale), int(img_height * scale)))

        # Clear previous image if any
        for widget in self.checkerboard_display_frame.winfo_children():
            widget.destroy()

        panel = ctk.CTkLabel(self.checkerboard_display_frame, image=ctki, text="")
        panel.image = ctki  # Keep a reference to avoid garbage collection
        panel.pack()

        # Add buttons to save or print the image
        button_frame = ctk.CTkFrame(self.checkerboard_display_frame, fg_color="transparent")
        button_frame.pack(pady=(10, 0))

        def save_as_pdf():
            # Ask the user for a file path to save the PDF
            file_path = filedialog.asksaveasfilename(
                defaultextension='.pdf', filetypes=[('PDF files', '*.pdf')])
            if file_path:
                image.save(file_path, 'PDF')
                messagebox.showinfo("Saved", f"Checkerboard image saved as {file_path}")

        ctk.CTkButton(button_frame, text="Save as PDF", command=save_as_pdf,
                     fg_color="#4CAF50", text_color='white', width=150, height=40).pack(side='left', padx=5)

    

    
            
    def build_activation_tab(self):
        """
        Build the Activation tab with three activation options.
        """
        frame = ctk.CTkFrame(self.activation_frame)
        frame.pack(expand=True, fill='both', padx=10, pady=10)
    
        # Title
        ctk.CTkLabel(frame, text="Activation",
                    font=('Helvetica', 20, 'bold')).pack(pady=20)
        
        # Description
        ctk.CTkLabel(frame, 
                    text="Choose how you want to launch Pose2Sim:",
                    font=('Helvetica', 14)).pack(pady=10)
    
        # Button Frame for better organization
        button_frame = ctk.CTkFrame(frame)
        button_frame.pack(pady=20)
    
        # CMD Button
        cmd_button = ctk.CTkButton(
            button_frame,
            text="Launch with CMD",
            command=lambda: self.activate_pose2sim_with(type='cmd'),
            width=200,
            height=40
        )
        cmd_button.pack(pady=10)
    
        # Anaconda Prompt Button
        conda_button = ctk.CTkButton(
            button_frame,
            text="Launch with Anaconda Prompt",
            command=lambda: self.activate_pose2sim_with(type='conda'),
            width=200,
            height=40
        )
        conda_button.pack(pady=10)
    
        # PowerShell Button
        powershell_button = ctk.CTkButton(
            button_frame,
            text="Launch with PowerShell",
            command=lambda: self.activate_pose2sim_with(type='powershell'),
            width=200,
            height=40
        )
        powershell_button.pack(pady=10)
    
    def activate_pose2sim_with(self, type='cmd'):
        """
        Activate Pose2Sim with specified terminal type
        type: 'cmd', 'conda', or 'powershell'
        """
        # Check pose model
        pose_model = None
        config_file_path = os.path.join(self.participant_name, 'Config.toml')
        try:
            with open(config_file_path, 'r', encoding='utf-8') as f:
                config = parse(f.read())
                pose_model = config.get('pose', {}).get('pose_model', 'HALPE_26')
        except Exception:
            pose_model = 'HALPE_26'  # default if can't read config
    
        # Show warning if not using HALPE_26
        do_pose_estimation = True
        if pose_model != 'HALPE_26':
            response = messagebox.showwarning(
                "Pose Model Warning",
                f"The selected pose model '{pose_model}' is not yet integrated in Pose2Sim. "
                "You will need to run pose estimation separately and then use Pose2Sim for the remaining steps.\n\n"
                "Pose estimation will be skipped in this run.",
                icon='warning'
            )
            do_pose_estimation = False
    
        # Generate the Python code to run Pose2Sim
        python_script_content = f"""
from Pose2Sim import Pose2Sim
Pose2Sim.runAll(do_calibration=True, 
                do_poseEstimation={str(do_pose_estimation)}, 
                do_synchronization={self.sync_videos_var.get().lower() == 'no'}, 
                do_personAssociation=True, 
                do_triangulation=True, 
                do_filtering=True, 
                do_markerAugmentation=True, 
                do_kinematics=True)
    """
    
        # Save the Python script
        script_path = os.path.join(self.participant_name, 'run_pose2sim.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(python_script_content)
    
        # Generate launch script based on type
        if type == 'cmd':
            launch_script = f"""
    @echo off
    setlocal EnableDelayedExpansion
    
    REM Activate Conda environment
    call conda activate Pose2Sim
    
    REM Change to the specified directory
    cd "{os.path.abspath(self.participant_name)}"
    
    REM Launch the Python script and keep the command prompt open
    python run_pose2sim.py
    
    REM Pause the command prompt to prevent it from closing
    pause
    
    endlocal
    """
            script_ext = 'cmd'
            launch_command = 'cmd /k'
    
        elif type == 'conda':
            launch_script = f"""
    @echo off
    setlocal EnableDelayedExpansion
    
    REM Change to the specified directory
    cd "{os.path.abspath(self.participant_name)}"
    
    REM Launch the Python script
    call conda activate Pose2Sim && python run_pose2sim.py
    
    REM Pause to keep the window open
    pause
    
    endlocal
    """
            script_ext = 'bat'
            launch_command = 'conhost'
    
        else:  # powershell
            launch_script = f"""
    # Change to the specified directory
    cd "{os.path.abspath(self.participant_name)}"
    
    # Activate Conda environment and run script
    conda activate Pose2Sim; python run_pose2sim.py
    
    # Pause to keep the window open
    Write-Host "Press any key to continue..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    """
            script_ext = 'ps1'
            launch_command = 'powershell'
    
        # Save and execute the launch script
        script_name = f'activate_pose2sim_{type}.{script_ext}'
        launch_script_path = os.path.join(self.participant_name, script_name)
        with open(launch_script_path, 'w', encoding='utf-8') as f:
            f.write(launch_script)
    
        # Launch the script
        if type == 'cmd':
            os.system(f'start {launch_command} "{launch_script_path}"')
        elif type == 'conda':
            os.system(f'start {launch_command} "{launch_script_path}"')
        else:  # powershell
            os.system(f'start {launch_command} -NoExit -Command "& \'{launch_script_path}\'"')
        
        
    def input_videos(self, target_path):
        """Handle video input for both single and batch modes"""
        try:
            num_cameras = int(self.num_cameras_var.get())
            video_extension = self.video_extension_var.get()
            
            for cam in range(1, num_cameras + 1):
                file_path = filedialog.askopenfilename(
                    title=f"Select video for Camera {cam}",
                    filetypes=[(f"Video files", f"*.{video_extension}")])
                
                if not file_path:
                    messagebox.showerror("Error", f"No file selected for camera {cam}")
                    return False
    
                # Copy and rename the file
                dest_filename = f"cam{cam}.{video_extension}"
                dest_path = os.path.join(target_path, dest_filename)
                
                # Create directory if it doesn't exist
                os.makedirs(target_path, exist_ok=True)
                
                # Copy the file
                shutil.copy(file_path, dest_path)
    
            return True
    
        except Exception as e:
            messagebox.showerror("Error", f"Error processing videos: {str(e)}")
            return False
    
   
    
   
    def finalize_configuration(self):
        """Create Config.toml files for trials based on parent configuration"""
        participant_path = os.path.join(self.participant_name)
        
        # Get the parent Config.toml content
        parent_config_path = os.path.join(participant_path, 'Config.toml')
        if os.path.exists(parent_config_path):
            with open(parent_config_path, 'r', encoding='utf-8') as f:
                parent_config = f.read()
            
            # Copy the parent config to each trial
            for i in range(1, self.num_trials + 1):
                trial_path = os.path.join(participant_path, f'Trial_{i}')
                trial_config_path = os.path.join(trial_path, 'Config.toml')
                
                # Create trial folder if it doesn't exist
                os.makedirs(trial_path, exist_ok=True)
                
                # Copy the parent config to the trial
                with open(trial_config_path, 'w', encoding='utf-8') as f:
                    f.write(parent_config)
    
    def build_batch_configuration_tab(self):
        """Build the batch configuration tab with trial-specific settings only"""
        if self.process_mode != 'batch':
            return
        
        frame = ctk.CTkFrame(self.tab_frames['batch_configuration'])
        frame.pack(expand=True, fill='both', padx=20, pady=20)
    
        # Header
        header_frame = ctk.CTkFrame(frame)
        header_frame.pack(fill='x', pady=(0, 20))
        ctk.CTkLabel(header_frame, 
                    text='Trial-Specific Configuration', 
                    font=('Helvetica', 20, 'bold')).pack(anchor='w')
    
        # Information label
        info_label = ctk.CTkLabel(frame,
                            text="Configure trial-specific parameters. Other settings will be inherited from the main configuration.",
                            wraplength=500)
        info_label.pack(pady=10)
    
        # Buttons for trials
        self.trials_frame = ctk.CTkFrame(frame)
        self.trials_frame.pack(fill='x', pady=10)
        
        for i in range(1, self.num_trials + 1):
            trial_button = ctk.CTkButton(
                self.trials_frame,
                text=f"Trial_{i}",
                command=lambda x=i: self.configure_trial(x),
                fg_color="#3B8ED0",
                width=200,
                height=40
            )
            trial_button.pack(pady=5)
    def get_config_value(self, config, setting):
        """Helper function to get value from nested config"""
        try:
            # Project settings
            if setting in ['frame_rate', 'frame_range', 'multi_person', 'participant_height', 'participant_mass']:
                return str(config.get('project', {}).get(setting, ''))
            
            # Pose settings
            elif setting in ['pose_model', 'mode', 'vid_img_extension', 'det_frequency', 'display_detection', 'save_video']:
                return str(config.get('pose', {}).get(setting, ''))
            
            # Synchronization settings
            elif setting in ['keypoints_to_consider', 'approx_time_maxspeed', 'time_range_around_maxspeed',
                            'likelihood_threshold', 'filter_cutoff', 'filter_order']:
                return str(config.get('synchronization', {}).get(setting, ''))
            
            # Person Association settings
            elif setting in ['likelihood_threshold_association']:
                return str(config.get('personAssociation', {}).get(setting, ''))
            elif setting in ['reproj_error_threshold_association', 'tracked_keypoint']:
                return str(config.get('personAssociation', {}).get('single_person', {}).get(setting, ''))
            
            # Triangulation settings
            elif setting in ['reproj_error_threshold_triangulation', 'likelihood_threshold_triangulation',
                            'min_cameras_for_triangulation', 'interpolation', 'interp_if_gap_smaller_than']:
                return str(config.get('triangulation', {}).get(setting, ''))
            
            # Filtering settings
            elif setting in ['type', 'display_figures']:
                return str(config.get('filtering', {}).get(setting, ''))
            elif setting in ['cut_off_frequency', 'order']:
                return str(config.get('filtering', {}).get('butterworth', {}).get(setting, ''))
            elif setting in ['trust_ratio', 'smooth']:
                return str(config.get('filtering', {}).get('kalman', {}).get(setting, ''))
            
            # Marker Augmentation settings
            elif setting == 'make_c3d':
                return str(config.get('markerAugmentation', {}).get(setting, True))
            
            # Kinematics settings
            elif setting in ['use_augmentation', 'right_left_symmetry', 
                            'remove_individual_scaling_setup', 'remove_individual_IK_setup']:
                return str(config.get('kinematics', {}).get(setting, True))
            
            return ''
        except:
            return ''
            
    def configure_trial(self, trial_number):
        """Open configuration window for trial-specific settings"""
        config_window = ctk.CTkToplevel(self.root)
        config_window.title(f"Configure Trial_{trial_number}")
        config_window.geometry("800x600")
        
        main_frame = ctk.CTkFrame(config_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scroll_frame = ctk.CTkScrollableFrame(main_frame)
        scroll_frame.pack(fill='both', expand=True)
        
        # Load trial configuration
        config_path = os.path.join(self.participant_name, f'Trial_{trial_number}', 'Config.toml')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = parse(f.read())
        except Exception as e:
            messagebox.showerror("Error", f"Could not load configuration for Trial_{trial_number}: {str(e)}")
            return
    
        settings_vars = {}
        
        # Expanded sections with more parameters
        sections = {
            'Project Settings': [
                'frame_range',
                'multi_person',
                'participant_height',
                'participant_mass'
            ],
            'Pose Estimation': [
                'pose_model',
                'mode',
                'vid_img_extension',
                'det_frequency',
                'display_detection',
                'save_video'
            ],
            'Synchronization': [
                'keypoints_to_consider',
                'approx_time_maxspeed',
                'time_range_around_maxspeed',
                'likelihood_threshold',
                'filter_cutoff',
                'filter_order'
            ],
            'Person Association': [
                'likelihood_threshold_association',
                'reproj_error_threshold_association',
                'tracked_keypoint'
            ],
            'Triangulation': [
                'reproj_error_threshold_triangulation',
                'likelihood_threshold_triangulation',
                'min_cameras_for_triangulation',
                'interpolation',
                'interp_if_gap_smaller_than'
            ],
            'Filtering': [
                'type',
                'display_figures',
                'cut_off_frequency',
                'order',
                'trust_ratio',
                'smooth'
            ],
            'Marker Augmentation': [
                'make_c3d'
            ],
            'Kinematics': [
                'use_augmentation',
                'right_left_symmetry',
                'remove_individual_scaling_setup',
                'remove_individual_IK_setup'
            ]
        }
    
        row = 0
        for section_name, settings in sections.items():
            # Section header
            ctk.CTkLabel(scroll_frame, text=section_name, 
                        font=('Helvetica', 16, 'bold')).grid(row=row, column=0, columnspan=2, pady=(15,5), sticky='w')
            row += 1
            
            for setting in settings:
                current_value = self.get_config_value(config, setting)
                
                # Handle boolean values
                if setting in ['multi_person', 'display_detection', 'make_c3d', 'use_augmentation', 
                            'right_left_symmetry', 'remove_individual_scaling_setup', 'remove_individual_IK_setup']:
                    var = ctk.BooleanVar(value=current_value.lower() == 'true' if isinstance(current_value, str) else bool(current_value))
                    ctk.CTkLabel(scroll_frame, text=setting.replace('_', ' ').title() + ':').grid(
                        row=row, column=0, pady=2, padx=5, sticky='w')
                    ctk.CTkCheckBox(scroll_frame, text="", variable=var).grid(
                        row=row, column=1, pady=2, padx=5, sticky='w')
                
                # Handle dropdown menus for specific settings
                elif setting in ['pose_model', 'mode', 'type', 'interpolation']:
                    var = ctk.StringVar(value=current_value)
                    options = {
                        'pose_model': ['HALPE_26', 'COCO_133', 'COCO_17', 'BODY_25B', 'BODY_25', 'BODY_135'],
                        'mode': ['lightweight', 'balanced', 'performance'],
                        'type': ['butterworth', 'kalman', 'gaussian', 'LOESS', 'median', 'butterworth_on_speed'],
                        'interpolation': ['linear', 'slinear', 'quadratic', 'cubic', 'none']
                    }
                    ctk.CTkLabel(scroll_frame, text=setting.replace('_', ' ').title() + ':').grid(
                        row=row, column=0, pady=2, padx=5, sticky='w')
                    ctk.CTkOptionMenu(scroll_frame, variable=var, values=options[setting]).grid(
                        row=row, column=1, pady=2, padx=5, sticky='w')
                
                # Default text entry for other settings
                else:
                    var = ctk.StringVar(value=str(current_value))
                    ctk.CTkLabel(scroll_frame, text=setting.replace('_', ' ').title() + ':').grid(
                        row=row, column=0, pady=2, padx=5, sticky='w')
                    ctk.CTkEntry(scroll_frame, textvariable=var, width=200).grid(
                        row=row, column=1, pady=2, padx=5, sticky='w')
                
                settings_vars[setting] = var
                row += 1
    
            def save_trial_configuration():
                try:
                    # Create nested structure if it doesn't exist
                    if 'project' not in config:
                        config['project'] = {}
                    if 'pose' not in config:
                        config['pose'] = {}
                    if 'synchronization' not in config:
                        config['synchronization'] = {}
                    if 'personAssociation' not in config:
                        config['personAssociation'] = {'single_person': {}}
                    if 'triangulation' not in config:
                        config['triangulation'] = {}
                    if 'filtering' not in config:
                        config['filtering'] = {'butterworth': {}, 'kalman': {}}
                    if 'markerAugmentation' not in config:
                        config['markerAugmentation'] = {}
                    if 'kinematics' not in config:
                        config['kinematics'] = {}
    
                    # Update configuration with new values
                    for setting, var in settings_vars.items():
                        value = var.get()
                        
                        # Project settings
                        if setting in ['frame_range', 'multi_person', 'participant_height', 'participant_mass']:
                            config['project'][setting] = value
                        
                        # Pose settings
                        elif setting in ['pose_model', 'mode', 'vid_img_extension', 'det_frequency', 
                                    'display_detection', 'save_video']:
                            config['pose'][setting] = value
                        
                        # Synchronization settings
                        elif setting in ['keypoints_to_consider', 'approx_time_maxspeed', 
                                    'time_range_around_maxspeed', 'likelihood_threshold', 
                                    'filter_cutoff', 'filter_order']:
                            config['synchronization'][setting] = value
                        
                        # Person Association settings
                        elif setting == 'likelihood_threshold_association':
                            config['personAssociation'][setting] = value
                        elif setting in ['reproj_error_threshold_association', 'tracked_keypoint']:
                            config['personAssociation']['single_person'][setting] = value
                        
                        # Triangulation settings
                        elif setting in ['reproj_error_threshold_triangulation', 
                                    'likelihood_threshold_triangulation',
                                    'min_cameras_for_triangulation', 'interpolation', 
                                    'interp_if_gap_smaller_than']:
                            config['triangulation'][setting] = value
                        
                        # Filtering settings
                        elif setting in ['type', 'display_figures']:
                            config['filtering'][setting] = value
                        elif setting in ['cut_off_frequency', 'order']:
                            if 'butterworth' not in config['filtering']:
                                config['filtering']['butterworth'] = {}
                            config['filtering']['butterworth'][setting] = value
                        elif setting in ['trust_ratio', 'smooth']:
                            if 'kalman' not in config['filtering']:
                                config['filtering']['kalman'] = {}
                            config['filtering']['kalman'][setting] = value
                        
                        # Marker Augmentation settings
                        elif setting == 'make_c3d':
                            config['markerAugmentation'][setting] = value
                        
                        # Kinematics settings
                        elif setting in ['use_augmentation', 'right_left_symmetry', 
                                    'remove_individual_scaling_setup', 'remove_individual_IK_setup']:
                            config['kinematics'][setting] = value
    
                    # Save the updated configuration
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(dumps(config))
    
                    messagebox.showinfo("Success", f"Configuration for Trial_{trial_number} has been saved successfully!")
                    config_window.destroy()
    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
        # Create save button frame at the bottom
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill='x', pady=10, padx=10)
    
        # Add the save button
        save_button = ctk.CTkButton(
            button_frame,
            text="Save Trial Configuration",
            command=save_trial_configuration,
            width=200,
            height=40
        )
        save_button.pack(pady=10)
    def set_config_value(self, config, setting, value):
        """Helper function to set value in nested config"""
        try:
            if setting in ['frame_rate', 'frame_range']:
                if 'project' not in config:
                    config['project'] = {}
                config['project'][setting] = value
            elif setting in ['pose_model', 'mode', 'vid_img_extension']:
                if 'pose' not in config:
                    config['pose'] = {}
                config['pose'][setting] = value
            elif setting in ['likelihood_threshold_association', 'tracked_keypoint']:
                if 'personAssociation' not in config:
                    config['personAssociation'] = {'single_person': {}}
                config['personAssociation']['single_person'][setting] = value
            elif setting in ['reproj_error_threshold_triangulation', 'likelihood_threshold_triangulation',
                            'min_cameras_for_triangulation']:
                if 'triangulation' not in config:
                    config['triangulation'] = {}
                config['triangulation'][setting] = value
            elif setting in ['type', 'cut_off_frequency', 'order']:
                if 'filtering' not in config:
                    config['filtering'] = {}
                config['filtering'][setting] = value
        except Exception as e:
            print(f"Error setting {setting}: {str(e)}")
    
def main():
    root = ctk.CTk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
