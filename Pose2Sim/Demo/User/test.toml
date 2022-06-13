###############################################################################
## PROJECT PARAMETERS                                                        ##
###############################################################################

# Configure your project parameters here

[project]
project_dir = '' # BETWEEN SINGLE QUOTES! # If empty, project dir is current dir
frames_range = [0,100] #For example [10,300], or [] if all frames.
frame_rate = 60 #Hz

rawImg_folder_name = 'raw-2d'
calib_folder_name = 'calib-2d'
pose_folder_name = 'pose-2d'
pose_json_folder_extension = 'json'
pose_img_folder_extension = 'img'
poseTracked_folder_name = 'pose-2d-tracked'
pose3d_folder_name = 'pose-3d'
opensim_folder_name = 'opensim'

[pose-2d]
openpose_model = 'BODY_25B' #BODY_25B, BODY_25, COCO, MPI are available.
# set your own model in skeleton.py if you don't use openpose.

[calibration]
type = 'qca' # 'qca', 'checkerboard', 'arucoboard', or 'charucoboard'
   [calibration.qca]
   binning_factor = 1 # Usually 1

   [calibration.checkerboard]
   corners_nb = [7,12] # [H,W] rather than [w,h]
   square_size = 80 # mm # [h,w] if square is actually a rectangle
   frame_for_origin = -1 # starting from zero. -1 if board is at origin on last frame.
   show_corner_detection = false # /!\ Beware that corners must be detected on all frames, 
   # or else extrinsic parameters may be wrong. Set show_corner_detection to 1 to verify.
   from_vid_or_img = 'img' # 'vid' or 'img'
   vid_snapshot_every_N_frames = 100
   vid_extension = 'mp4'
   img_extension = 'jpg' # 'png', 'jpg', etc

[2d-tracking]
tracked_keypoint = 'Neck' # Check skeleton.py and choose a stable point for tracking the person of interest.
error_threshold_tracking = 20 # px

[3d-triangulation]
error_threshold_triangulation = 15 # px
likelihood_threshold = 0.3
min_cameras_for_triangulation = 2
interpolation = 'cubic' #linear, slinear, quadratic, cubic

[3d-filtering]
type = 'butterworth' # butterworth, butterworth_on_speed, gaussian, LOESS, median.
display_figures = 'False' # 'True' or 'False'

   [3d-filtering.butterworth]
   type = 'low'
   order = 4 
   cut_off_frequency = 6 # Hz
   [3d-filtering.butterworth_on_speed]
   type = 'low'
   order = 4 
   cut_off_frequency = 10 # Hz
   [3d-filtering.gaussian]
   sigma_kernel = 2 #px
   [3d-filtering.LOESS]
   nb_values_used = 30 # = fraction of data used * nb frames
   [3d-filtering.median]
   kernel_size = 9

[opensim]
