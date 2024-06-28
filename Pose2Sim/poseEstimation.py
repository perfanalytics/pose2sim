import time
import cv2
import json
import os
import toml
from tqdm import tqdm
from rtmlib import PoseTracker, Body, Wholebody, Body_and_Feet, draw_skeleton
import glob
import re

def natural_sort_key(s):
    """
    Key for natural sorting of strings containing numbers.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', s)]

def process_video(video_path, output_path, json_output_dir, pose_tracker, save_video, realtime_vis, openpose_skeleton):
    """
    Process a video file for pose estimation.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output video (if save_video is True)pip
        json_output_dir (str): Directory to save JSON output files
        pose_tracker (PoseTracker): Initialized pose tracker object
        save_video (bool): Whether to save the output video
        realtime_vis (bool): Whether to show real-time visualization
    """
    cap = cv2.VideoCapture(video_path)

    if save_video:
        # Set up video writer if saving video is enabled
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate from the raw video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Get the width from the raw video
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the height from the raw video
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # Create the output video file

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    if realtime_vis:
        # Create a window for real-time visualization
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL) # window name can be changed "Pose Estimation" to anything else

    with tqdm(total=total_frames, desc="Processing Frames", leave=False, ncols=100) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # Perform pose estimation on the frame
            keypoints, scores = pose_tracker(frame)
            
            # Prepare keypoints with confidence scores for JSON output
            keypoints_with_confidence = []
            for kp, score in zip(keypoints[0], scores[0]):
                keypoints_with_confidence.extend([kp[0].item(), kp[1].item(), score.item()])

            # Create JSON output structure
            json_output = {
                "version": 1.3,
                "people": [
                    {
                        "person_id": [-1],
                        "pose_keypoints_2d": keypoints_with_confidence,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": []
                    }
                ]
            }
            
            # Save JSON output for each frame
            json_file_path = os.path.join(json_output_dir, f"frame_{frame_idx:04d}.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)

            # Draw skeleton on the frame
            img_show = frame.copy()
            img_show = draw_skeleton(img_show,
                                     keypoints,
                                     scores,
                                     kpt_thr=0.1, # maybe change this value if 0.1 is too low.
                                     openpose_skeleton=openpose_skeleton)

            if save_video:
                out.write(img_show)

            if realtime_vis:
                # Show real-time visualization
                cv2.imshow("Pose Estimation", img_show)
                # Break the loop if 'q' is pressed (also closes the window and stops the process)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

    cap.release()
    if save_video:
        out.release()
    if realtime_vis:
        cv2.destroyAllWindows()

def process_images(image_folder, json_output_dir, pose_tracker, realtime_vis, openpose_skeleton):
    """
    Process a folder of image files for pose estimation.
    
    Args:
        image_folder (str): Path to the folder containing input images
        json_output_dir (str): Directory to save JSON output files
        pose_tracker (PoseTracker): Initialized pose tracker object
        realtime_vis (bool): Whether to show real-time visualization
        openpose_skeleton (bool): Whether to use OpenPose skeleton format
    """
    image_files = glob.glob(os.path.join(image_folder, '*.[jp][pn]g'))  # supports jpg, jpeg, png
    image_files.sort(key=natural_sort_key)
    
    if realtime_vis:
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL) # window name can be changed "Pose Estimation" to anything else

    for idx, image_file in enumerate(tqdm(image_files, desc="Processing Images", ncols=100)):
        frame = cv2.imread(image_file)
        
        # Perform pose estimation on the image
        keypoints, scores = pose_tracker(frame)
        
        # Prepare keypoints with confidence scores for JSON output
        keypoints_with_confidence = []
        for kp, score in zip(keypoints[0], scores[0]):
            keypoints_with_confidence.extend([kp[0].item(), kp[1].item(), score.item()])

        # Create JSON output structure
        json_output = {
            "version": 1.3,
            "people": [
                {
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_with_confidence,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                }
            ]
        }
        
        # Extract frame number from the filename
        frame_number = int(re.search(r'\d+', os.path.basename(image_file)).group())
        
        # Save JSON output for each image
        json_file_path = os.path.join(json_output_dir, f"frame_{frame_number:04d}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_output, json_file, indent=4)

        # Draw skeleton on the image
        img_show = frame.copy()
        img_show = draw_skeleton(img_show,
                                 keypoints,
                                 scores,
                                 kpt_thr=0.1, # maybe change this value if 0.1 is too low.
                                 openpose_skeleton=openpose_skeleton)

        if realtime_vis:
            cv2.imshow("Pose Estimation", img_show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if realtime_vis:
        cv2.destroyAllWindows()

def rtm_estimator(config_dict):
    """
    Main function to run the pose estimation process based on the configuration.
    
    Args:
        config_dict (dict): Configuration dictionary loaded from TOML file
    """
    # Read configuration
    project_dir = config_dict['project']['project_dir']
    raw_path = os.path.join(project_dir, 'pose_raw') # 'pose_raw' could be changed to something else. It is just demo.
    output_base_dir = os.path.abspath(os.path.join(raw_path, '..', 'pose'))
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Extract configuration parameters
    """
    Should refine the alignment of the parameters with the configuration file.
    """
    data_type = config_dict['pose_demo']['data_type'] # video, image
    save_video = config_dict['pose_demo']['save_video']
    device = config_dict['pose_demo']['device'] # cpu, gpu
    backend = config_dict['pose_demo']['backend'] # onnxruntime, openvino, opencv(It seems not supported yet.)
    det_frequency = config_dict['pose_demo']['det_frequency']
    skeleton_type = config_dict['pose_demo']['skeleton_type'] # Body, Wholebody, Body_and_Feet
    mode = config_dict['pose_demo']['mode'] # performance, balanced, lightweight
    tracking = config_dict['pose_demo']['tracking']
    openpose_skeleton = config_dict['pose_demo']['to_openpose']
    realtime_vis = config_dict['pose_demo']['realtime_vis']

    # Select the appropriate model based on the model_type
    if skeleton_type == 'Body' or skeleton_type == 'body':
        ModelClass = Body # 17 keypoints
    elif skeleton_type == 'Wholebody' or skeleton_type == 'wholebody':
        ModelClass = Wholebody # 133 keypoints
    elif skeleton_type == 'Body_and_Feet' or skeleton_type == 'body_and_feet':
        ModelClass = Body_and_Feet # 26 keypoints(halpe26)
    else:
        raise ValueError(f"Invalid model_type: {skeleton_type}. Must be 'Body', 'Wholebody', or 'Body_and_Feet'.")

    # Initialize the pose tracker
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=openpose_skeleton)

    if data_type == "video":
        # Process video files
        video_files = glob.glob(os.path.join(raw_path, '*.mp4')) + glob.glob(os.path.join(raw_path, '*.avi'))
        for video_file in video_files:
            video_name = os.path.basename(video_file)
            video_name_wo_ext = os.path.splitext(video_name)[0]
            output_video_path = os.path.join(output_base_dir, f"{video_name_wo_ext}_pose.avi")
            json_output_dir = os.path.join(output_base_dir, video_name_wo_ext)

            if not os.path.exists(json_output_dir):
                os.makedirs(json_output_dir)

            process_video(video_file, output_video_path, json_output_dir, pose_tracker, save_video, realtime_vis, openpose_skeleton)

    elif data_type == "image":
        # Process image folders
        image_folders = [f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]
        for image_folder in image_folders:
            image_folder_path = os.path.join(raw_path, image_folder)
            json_output_dir = os.path.join(output_base_dir, image_folder)
            # Could add a save image option here if needed!

            if not os.path.exists(json_output_dir):
                os.makedirs(json_output_dir)

            process_images(image_folder_path, json_output_dir, pose_tracker, realtime_vis, openpose_skeleton)
    else:
        raise ValueError("Invalid data type. Must be 'video' or 'image'.")