import time
import cv2
import json
import os
import toml
from tqdm import tqdm
from Pose2Sim.Poselib import PoseTracker, Wholebody, draw_skeleton
import glob

def process_video(video_path, output_path, json_output_dir, wholebody, save_video, openpose_skeleton, realtime_vis):
    """
    Process a video file for pose estimation.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path to save the output video (if save_video is True)
        json_output_dir (str): Directory to save JSON output files
        wholebody (PoseTracker): Initialized pose tracker object
        save_video (bool): Whether to save the output video
        openpose_skeleton (bool): Whether to use OpenPose skeleton format
        realtime_vis (bool): Whether to show real-time visualization
    """
    cap = cv2.VideoCapture(video_path)

    if save_video:
        # Set up video writer if saving video is enabled
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    if realtime_vis:
        # Create a window for real-time visualization
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

    with tqdm(total=total_frames, desc="Processing Frames", leave=False, ncols=100) as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            frame_idx += 1

            if not success:
                break

            # Perform pose estimation on the frame
            keypoints, scores = wholebody(frame)
            
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
            json_file_path = os.path.join(json_output_dir, f"frame_{frame_idx:05d}.json")
            with open(json_file_path, 'w') as json_file:
                json.dump(json_output, json_file, indent=4)

            # Draw skeleton on the frame
            img_show = frame.copy()
            img_show = draw_skeleton(img_show,
                                     keypoints,
                                     scores,
                                     openpose_skeleton=openpose_skeleton,
                                     kpt_thr=0.1)

            if save_video:
                out.write(img_show)

            if realtime_vis:
                # Show real-time visualization
                cv2.imshow("Pose Estimation", img_show)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)

    cap.release()
    if save_video:
        out.release()
    if realtime_vis:
        cv2.destroyAllWindows()

def process_images(image_dir, json_output_dir, wholebody, openpose_skeleton, realtime_vis):
    """
    Process a directory of images for pose estimation.
    
    Args:
        image_dir (str): Directory containing input images
        json_output_dir (str): Directory to save JSON output files
        wholebody (PoseTracker): Initialized pose tracker object
        openpose_skeleton (bool): Whether to use OpenPose skeleton format
        realtime_vis (bool): Whether to show real-time visualization
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.*')))
    frame_idx = 0

    if realtime_vis:
        # Create a window for real-time visualization
        cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)

    for image_file in tqdm(image_files, desc="Processing Images", ncols=100):
        frame = cv2.imread(image_file)
        frame_idx += 1

        # Perform pose estimation on the image
        keypoints, scores = wholebody(frame)
        
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
        
        # Save JSON output for each image
        json_file_path = os.path.join(json_output_dir, f"frame_{frame_idx:05d}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json_output, json_file, indent=4)

        # Draw skeleton on the image
        img_show = frame.copy()
        img_show = draw_skeleton(img_show,
                                 keypoints,
                                 scores,
                                 openpose_skeleton=openpose_skeleton,
                                 kpt_thr=0.1)

        if realtime_vis:
            # Show real-time visualization
            cv2.imshow("Pose Estimation", img_show)
            # Break the loop if 'q' is pressed
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
    raw_path = os.path.join(project_dir, 'pose_raw')
    output_base_dir = os.path.abspath(os.path.join(raw_path, '..', 'pose'))
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # Extract configuration parameters
    data_type = config_dict['pose_demo']['data_type']
    save_video = config_dict['pose_demo']['save_video']
    device = config_dict['pose_demo']['device']
    backend = config_dict['pose_demo']['backend']
    det_frequency = config_dict['pose_demo']['det_frequency']
    to_openpose = config_dict['pose_demo']['to_openpose']
    mode = config_dict['pose_demo']['mode']
    tracking = config_dict['pose_demo']['tracking']
    realtime_vis = config_dict['pose_demo']['realtime_vis']

    # Initialize pose tracker
    wholebody = PoseTracker(
        Wholebody,
        det_frequency=det_frequency,
        to_openpose=to_openpose,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking)

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

            process_video(video_file, output_video_path, json_output_dir, wholebody, save_video, to_openpose, realtime_vis)

    elif data_type == "image":
        # Process image folders
        image_folders = [f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))]
        for image_folder in image_folders:
            image_folder_path = os.path.join(raw_path, image_folder)
            json_output_dir = os.path.join(output_base_dir, image_folder)

            if not os.path.exists(json_output_dir):
                os.makedirs(json_output_dir)

            process_images(image_folder_path, json_output_dir, wholebody, to_openpose, realtime_vis)
    else:
        raise ValueError("Invalid data type. Must be 'video' or 'image'.")