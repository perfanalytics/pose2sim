#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ######################################################
    ###### Face Blurring with polygon or rectangle #######
    ######################################################

    Detect faces in videos and blur or mask them.
    Beware that if a person is not detected in one or several frames, their face will not be blurred.
    Contributions are welcome if you can provide a fix!

    Usage:
        If you want to blur faces in a video:
        from Pose2Sim.Utilities import face_blurring; face_blurring.face_blurring_func(r'<input_video_file>')
        OR face_blurring -i input_video_file
        OR face_blurring -i input_video_file -o output_video_file

        Other arguments are available, type face_blurring -h for a list.


        # TODO: Add support for batch processing of multiple videos in a root folder.
        If you want to blur faces many videos in a root folder:
        from Pose2Sim.Utilities import face_blurring; face_blurring.face_blurring_func(r'<root_folder>')
        OR face_blurring -d video_directory
'''


## INIT
import cv2
import numpy as np
import logging
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from rtmlib import Body, PoseTracker
from Pose2Sim.poseEstimation import setup_backend_device, save_to_openpose


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2023, BlendOSim & Sim2Blend"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# Define face-related keypoint indices based on standard COCO-17/MMPose order
# nose, left_eye, right_eye, left_ear, right_ear
# (Refer to https://github.com/Tau-J/rtmlib/blob/main/rtmlib/visualization/skeleton/coco17.py)
FACE_KEYPOINT_INDICES = [0, 1, 2, 3, 4]

# Default settings
DEFAULT_VISUALIZE = False # Display the processed video in real-time (set default to False)
DEFAULT_BLUR_TYPE = 'blur' # 'blur' or 'black'
DEFAULT_BLUR_INTENSITY = "medium" # low, medium, high
DEFAULT_BLUR_SHAPE = "rectangle" # Shape of the blurred area: 'polygon' or 'rectangle'
DEFAULT_BLUR_SIZE = "small" # 'small', 'medium', 'large'
DEFAULT_BACKEND = 'auto'
DEFAULT_DEVICE = 'auto'
DEFAULT_MODEL_TYPE = 'rtmpose' # 'rtmpose' or 'rtmo'
DEFAULT_MODE = 'lightweight'
DEFAULT_DET_FREQUENCY = 10 
DEFAULT_CONFIDENCE_THRESHOLD = 0.0 # Face keypoint confidence threshold; face keypoints basically have low confidence.
DEFAULT_SAVE_JSON = False # Save detected face keypoints to JSON files
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', 
    '.m4v', '.mpg', '.mpeg', '.3gp', '.ts', '.mts', '.m2ts'}

def face_blurring_func(**args):
    """Core logic for face blurring.
    Takes arguments as a dictionary.
    """

    input_vid_path = args.get('input_vid')
    input_dir_path = args.get('input_dir')
    output_video_path = args.get('output', None) # Use .get for defaults
    visualize = args.get('visualize', DEFAULT_VISUALIZE)
    blur_type = args.get('blur_type', DEFAULT_BLUR_TYPE)
    blur_intensity = args.get('blur_intensity', DEFAULT_BLUR_INTENSITY) # Retrieve blur_intensity
    blur_shape = args.get('blur_shape', DEFAULT_BLUR_SHAPE)
    blur_size = args.get('blur_size', DEFAULT_BLUR_SIZE) # Retrieve blur_size
    model_type = args.get('model_type', DEFAULT_MODEL_TYPE)
    mode = args.get('mode', DEFAULT_MODE)
    det_frequency = args.get('det_frequency', DEFAULT_DET_FREQUENCY)
    backend = args.get('backend', DEFAULT_BACKEND)
    device = args.get('device', DEFAULT_DEVICE)
    conf_threshold = args.get('conf', DEFAULT_CONFIDENCE_THRESHOLD)
    save_json = args.get('save_json', DEFAULT_SAVE_JSON)


    # Setup basic logging
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Determine input videos
    if input_dir_path:
        # Find vide files recursively, excluding those starting with '_blurred' <- to avoid processing already processed videos
        video_files = [file for file in Path(input_dir_path).glob('**/*')
                        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS]
       
        # Filter out videos whose stem ends with _blurred
        input_videos = [v for v in video_files if not v.stem.endswith('_blurred')]
        if not input_videos:
            logging.error(f"Error: No video files found in {input_dir_path}")
            return
        logging.info(f"Found {len(input_videos)} videos in {input_dir_path}. Processing...")
    elif input_vid_path:
        input_videos = [Path(input_vid_path)]
        logging.info(f"Processing single video: {input_vid_path}")
    else:
        logging.error("Error: Neither root path nor input video path was provided.")
        return

    start_time_total = time.time()

    # Setup pose estimator
    backend, device = setup_backend_device(backend=backend, device=device) # Use default values

    if model_type == 'rtmpose':
        pose_sol = PoseTracker(Body,
                            det_frequency=det_frequency,
                            tracking=False,
                            mode=mode,
                            to_openpose=False,
                            backend=backend,
                            device=device)
    elif model_type == 'rtmo':
        pose_sol = Body(pose='rtmo',
                        mode=mode,
                        to_openpose=False,
                        backend=backend,
                        device=device)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Please use 'rtmpose' or 'rtmo'.")

    # --- Loop through each input video --- 
    for current_input_vid_path in input_videos:
        logging.info(f"--- Processing video: {current_input_vid_path} ---")
        start_time_video = time.time()

        # Open video capture for the current video
        cap = cv2.VideoCapture(str(current_input_vid_path)) # Should be str
        if not cap.isOpened():
            logging.error(f"Error: Could not open video file {current_input_vid_path}. Skipping.")
            continue # Skip to the next video

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer for the current video
        # Determine output path based on whether an explicit output was given
        if output_video_path is None:
            # Default: save next to input video with '_blurred' prefix
            current_output_video_path = current_input_vid_path.parent / f"{current_input_vid_path.stem}_blurred{current_input_vid_path.suffix}"
        elif len(input_videos) > 1 and input_dir_path:
            # Multiple videos from root, explicit output is a directory
            output_dir = Path(output_video_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            current_output_video_path = output_dir / f"{current_input_vid_path.stem}_blurred{current_input_vid_path.suffix}"
        else:
            # Single video or specific output file provided
            current_output_video_path = Path(output_video_path)
            # Ensure parent directory exists if it's a specific file path
            current_output_video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(current_output_video_path), fourcc, fps, (frame_width, frame_height))
        logging.info(f"Output video will be saved to: {current_output_video_path}")

        # Setup JSON output directory if needed for the current video
        current_json_output_dir = None
        if save_json:
            json_output_parent_dir = current_output_video_path.parent
            current_json_output_dir = json_output_parent_dir / f'{current_output_video_path.stem}_face_json'
            current_json_output_dir.mkdir(exist_ok=True)
            logging.info(f"Face keypoints JSON will be saved to: {current_json_output_dir}")

        if visualize:
            window_name = f"Face Blurring - {current_input_vid_path.name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Process frames for the current video
        frame_idx = 0
        with tqdm(total=total_frames, desc=f"Blurring {current_input_vid_path.name}") as pbar:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                # 1. Detect poses (keypoints and scores)
                keypoints, scores = pose_sol(frame) # keypoints: [N_persons, N_kpts, 2], scores: [N_persons, N_kpts]
                processed_frame = frame.copy() # Copy original frame for processing

                # Prepare data for JSON saving (only face keypoints)
                keypoints_for_json = np.zeros_like(keypoints)
                scores_for_json = np.zeros_like(scores)

                # 2. Iterate through detected persons to apply obscuration and prepare JSON data
                for person_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
                    # 3. Extract face keypoints and scores
                    face_kpts = person_kpts[FACE_KEYPOINT_INDICES]
                    face_scores = person_scores[FACE_KEYPOINT_INDICES]

                    # 4. Filter valid keypoints by confidence threshold
                    valid_indices = np.where(face_scores >= conf_threshold)[0]
                    valid_detected_kpts = face_kpts[valid_indices]
                    # valid_detected_scores = face_scores[valid_indices] # Not used currently

                    # --- Estimate Forehead and Chin Points --- 
                    points_for_hull = valid_detected_kpts.copy().astype(int) # Start with valid detected points, ensure int

                    # Check if we have enough points for estimation (at least eyes and nose)
                    # Get indices relative to FACE_KEYPOINT_INDICES
                    nose_idx_in_face = 0 
                    leye_idx_in_face = 1
                    reye_idx_in_face = 2

                    # Check if these keypoints are valid based on their original indices in face_scores
                    nose_valid = face_scores[nose_idx_in_face] >= conf_threshold
                    leye_valid = face_scores[leye_idx_in_face] >= conf_threshold
                    reye_valid = face_scores[reye_idx_in_face] >= conf_threshold

                    if nose_valid and leye_valid and reye_valid:
                        nose_pt = face_kpts[nose_idx_in_face]
                        leye_pt = face_kpts[leye_idx_in_face]
                        reye_pt = face_kpts[reye_idx_in_face]
                        
                        # Calculate center points and distance
                        y_eye_avg = (leye_pt[1] + reye_pt[1]) / 2
                        x_center = (leye_pt[0] + reye_pt[0]) / 2 # Use eye center for x
                        y_nose = nose_pt[1]

                        d_face = abs(y_nose - y_eye_avg) # Use abs for safety

                        if d_face > 1: # Ensure the distance is meaningful (e.g., more than 1 pixel)
                            
                            # NOTE: Since we don't have full face keypoints, we used the distance between eyes and nose to estimate the forehead and chin points.
                            # From my rule of thumb for 'polygon' blur_shape, the forehead is usually 2.5x and the chin is 3x the distance between eyes and nose.
                            #                       for 'rectangle' blur_shape, the forehead is usually 2x and the chin is 2.5x the distance between eyes and nose.
                            if blur_size == "small":
                                factor_chin = 2.5 # Add some padding downwards
                                factor_forehead = 2.0 # Add a bit more padding upwards
                            elif blur_size == "medium":
                                factor_chin = 3.0 # Add some padding downwards
                                factor_forehead = 2.5 # Add a bit more padding upwards
                            elif blur_size == "large":
                                factor_chin = 4.0 # Add some padding downwards
                                factor_forehead = 3.0 # Add a bit more padding upwards
                            else: # Default to medium if unknown value
                                logging.warning(f"Unknown blur_size '{blur_size}', please select from 'small', 'medium', 'large'. For now, blur_size is set to 'medium'.")
                                factor_chin = 3.0 
                                factor_forehead = 2.5

                            # Estimate chin
                            y_chin = y_nose + d_face * factor_chin
                            chin_pt = np.array([[int(x_center), int(y_chin)]])

                            # Estimate forehead top
                            y_forehead = y_eye_avg - d_face * factor_forehead
                            forehead_pt = np.array([[int(x_center), int(y_forehead)]])

                            # Add estimated points to the list for hull calculation
                            # Ensure points_for_hull is 2D before vstack if it was empty
                            if points_for_hull.shape[0] > 0:
                                points_for_hull = np.vstack([points_for_hull, chin_pt, forehead_pt])
                            else: # Handle case where no initial points were valid
                                points_for_hull = np.vstack([chin_pt, forehead_pt])
                    # --- End Estimation ---

                    # 5. Apply blurring/black polygon using combined points if enough points exist
                    # Need at least 3 points (original or combined) to form a hull/ROI
                    # if len(valid_face_kpts_obscuration) >= 3: # Old check
                    if points_for_hull.shape[0] >= 3:
                        # Pass blur_intensity to apply_face_obscuration
                        processed_frame = apply_face_obscuration(processed_frame, points_for_hull, blur_type, blur_shape, blur_intensity)

                    # --- Prepare filtered data for save_to_openpose --- 
                    if save_json and current_json_output_dir:
                        for i, kpt_idx in enumerate(FACE_KEYPOINT_INDICES):
                            if face_scores[i] >= conf_threshold:
                                # Copy only valid face keypoints and scores to the arrays for saving
                                keypoints_for_json[person_idx, kpt_idx] = person_kpts[kpt_idx]
                                scores_for_json[person_idx, kpt_idx] = person_scores[kpt_idx]
                    # --- End JSON data preparation for this person --- 

                # 6. Save face keypoints to JSON using save_to_openpose (once per frame)
                if save_json and current_json_output_dir:
                    json_file_path = current_json_output_dir / f'{current_output_video_path.stem}_{frame_idx:06d}_face.json'
                    save_to_openpose(str(json_file_path), keypoints_for_json, scores_for_json)

                # 7. Visualize if enabled
                if visualize:
                    cv2.imshow(window_name, processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # 8. Write processed frame to output video
                out.write(processed_frame)

                frame_idx += 1
                pbar.update(1)
            
        # Release resources
        cap.release()
        out.release()
        if visualize:
            cv2.destroyWindow(window_name)

        logging.info(f"Finished processing {current_input_vid_path}.")
        end_time_video = time.time()
        logging.info(f"Processing {current_input_vid_path.name} took {end_time_video - start_time_video:.2f} seconds.")
    # --- End loop through videos ---

    logging.info("Face blurring process finished.")
    end_time_total = time.time()
    logging.info(f"Total processing finished in {end_time_total - start_time_total:.2f} seconds.")


def apply_face_obscuration(frame: np.ndarray, face_keypoints: np.ndarray, blur_type: str, blur_shape: str, blur_intensity: int) -> np.ndarray:
    """Applies blurring or a black polygon to the face region defined by keypoints using ROI region <- if not, very slow in my case.

    Args:
        frame: The input image frame.
        face_keypoints: A NumPy array of shape (N, 2) containing valid face keypoints (x, y).
                       N must be >= 3.
        blur_type: The type of obscuration ('blur' or 'black').
        blur_shape: The shape of the obscuration area ('polygon' or 'rectangle').
        blur_intensity: Divisor for kernel size calculation (smaller is more intense blur).

    Returns:
        The frame with the face region obscured.
    """

    try:
        # 1. Calculate the convex hull
        hull_points = face_keypoints.astype(int)
        hull = cv2.convexHull(hull_points)

        # 2. Calculate Bounding Box for ROI with padding
        x, y, w, h = cv2.boundingRect(hull)
        padding = int(max(w, h) * 0.2) # Add 20% padding relative to the larger dimension
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        roi_w = x2 - x1
        roi_h = y2 - y1

        # Ensure ROI dimensions are valid
        if roi_w <= 0 or roi_h <= 0:
            return frame 

        # 3. Extract ROI from the frame
        roi = frame[y1:y2, x1:x2]

        # 4. Adjust hull points relative to ROI origin
        hull_roi = hull_points - [x1, y1]

        # 5. Create mask based on blur_shape within ROI
        mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
        if blur_shape == 'polygon':
            # Ensure hull_roi has at least 3 points before calling convexHull again
            if len(hull_roi) >= 3:
                 cv2.fillConvexPoly(mask_roi, cv2.convexHull(hull_roi), 255) # Use adjusted hull points
            else:
                 logging.warning("Not enough points for polygon mask after ROI adjustment, using rectangle.")
                 mask_roi.fill(255) # Fallback to rectangle if hull is invalid in ROI
        elif blur_shape == 'rectangle':
            mask_roi.fill(255) # Fill the entire ROI for rectangle shape
        else:
             logging.warning(f"Unknown blur_shape: {blur_shape}. Using polygon.")
             if len(hull_roi) >= 3:
                  cv2.fillConvexPoly(mask_roi, cv2.convexHull(hull_roi), 255)
             else:
                  mask_roi.fill(255)
        

        # 6. Process based on blur_type using the generated mask_roi
        if blur_type == 'blur':
            # Determine the blur intensity
            if blur_intensity == "low":
                blur_intensity = 6
            elif blur_intensity == "medium":
                blur_intensity = 4
            elif blur_intensity == "high":
                blur_intensity = 2

            # Apply Gaussian blur ONLY to the ROI
            kernel_w = max(5, int(roi_w / blur_intensity) * 2 + 1) # Dynamic kernel based on ROI width
            kernel_h = max(5, int(roi_h / blur_intensity) * 2 + 1) # Dynamic kernel based on ROI height
            # Ensure kernel size is odd
            kernel_size = (kernel_w, kernel_h)
            blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
            
            # Blend within the ROI using the mask
            processed_roi = roi.copy()
            processed_roi[mask_roi == 255] = blurred_roi[mask_roi == 255]

        elif blur_type == 'black':
            # Apply black color within the ROI using the mask
            processed_roi = roi.copy()
            processed_roi[mask_roi == 255] = (0, 0, 0)
            
        else:
            logging.warning(f"Unknown blur_type: {blur_type}. Returning original frame.")
            return frame

        # 7. Put the processed ROI back into the original frame
        obscured_frame = frame.copy() # Start with a copy of the original
        obscured_frame[y1:y2, x1:x2] = processed_roi
        
        return obscured_frame

    except Exception as e:
        logging.error(f"Error during face obscuration: {e}")
        # Return original frame in case of error during polygon processing
        return frame

# --- Argument Parser and Main Function --- 

def main():
    parser = argparse.ArgumentParser(description='Detect and blur faces in a video.')
    parser.add_argument('-i', '--input_vid', required=False, help='Path to the input video file if single video edit (string).')
    parser.add_argument('-d', '--input_dir', required=False, help='Path to the input video directory if multiple video edit (string).')
    parser.add_argument('-o', '--output', default=None, help='Path to the output video file (string). Defaults to "blurred_<input_name>".')
    parser.add_argument('-v', '--visualize', action='store_true', help=f'Enable real-time visualization (boolean, default: {DEFAULT_VISUALIZE}).')
    parser.add_argument('--blur_type', default=DEFAULT_BLUR_TYPE, choices=['blur', 'black'], help=f'Type of obscuration (string, default: {DEFAULT_BLUR_TYPE}).')
    parser.add_argument('--blur_intensity', default=DEFAULT_BLUR_INTENSITY, choices=['low', 'medium', 'high'], help=f'Intensity of Gaussian blur (string: low, medium, high. default: {DEFAULT_BLUR_INTENSITY}).')
    parser.add_argument('--blur_size', default=DEFAULT_BLUR_SIZE, choices=['small', 'medium', 'large'], help=f'Size of the estimated face area for padding (string, default: {DEFAULT_BLUR_SIZE}).')
    parser.add_argument('--blur_shape', default=DEFAULT_BLUR_SHAPE, choices=['polygon', 'rectangle'], help=f'Shape of the blurred area (string, default: {DEFAULT_BLUR_SHAPE}).')
    parser.add_argument('--backend', default=DEFAULT_BACKEND, help="Backend if you don't want it to be determined automatically ('onnxruntime', 'openvino', ...)")
    parser.add_argument('--device', default=DEFAULT_DEVICE, help="Device if you don't want it to be determined automatically ('cpu', 'cuda', 'mps', ...)")
    parser.add_argument('--model_type', default=DEFAULT_MODEL_TYPE, help='Model type (string, "rtmo" or "rtmpose". Default: "rtmpose").')
    parser.add_argument('--mode', default=DEFAULT_MODE, choices=list(vars(Body)['MODE'].keys()), help=f'Pose estimation mode (string among {list(vars(Body)["MODE"].keys())}, default: {DEFAULT_MODE}).')
    parser.add_argument('--det_frequency', type=int, default=DEFAULT_DET_FREQUENCY, help=f'Detection frequency. The higher, the faster the process will go but potentially at the expense of a lesser accuracy (int, default: {DEFAULT_DET_FREQUENCY}).')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f'Minimum confidence threshold for face keypoints (float, default: {DEFAULT_CONFIDENCE_THRESHOLD}).')
    parser.add_argument('--save_json', action='store_true', dest='save_json', help='Saving face keypoints to JSON files (boolean, default: {DEFAULT_SAVE_JSON}).')

    args = vars(parser.parse_args())

    # Confirm that exactly one of root or input is provided
    if not (args['input_dir'] or args['input_vid']):
        parser.error('Either --input_dir or --input_vid must be provided.')
    if args['input_dir'] and args['input_vid']:
        parser.error('Provide either --input_dir or --input_vid, not both.')

    face_blurring_func(**args)

    # Example usage from terminal:

    # If you want to blur faces in a video:
    # python Utilities/face_blurring.py -i path/to/your/video.mp4 --blur_type black --visualize

    # If you want to blur faces in a root folder:
    # python Utilities/face_blurring.py -r path/to/your/root/folder --blur_type black --visualize

if __name__ == '__main__':
    main()