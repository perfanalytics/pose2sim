#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ######################################################
    ###### Face Blurring with polygon or rectangle #######
    ######################################################

    Detect faces in videos and blur them.
    If you concern privacy more seriously, you can replace the blurring by a black polygon.

    Usage:
        If you want to blur faces in a video:
        from Pose2Sim.Utilities import face_blurring; face_blurring.face_blurring_func(r'<input_video_file>')
        OR face_blurring -i input_video_file
        OR face_blurring -i input_video_file -o output_video_file

        # TODO: Add support for batch processing of multiple videos in a root folder.
        If you want to blur faces many videos in a root folder:
        from Pose2Sim.Utilities import face_blurring; face_blurring.face_blurring_func(r'<root_folder>')
        OR face_blurring -r root_folder
'''

## INIT
import cv2
import numpy as np
import os
import logging
import argparse
import time
from tqdm import tqdm
from functools import partial
from rtmlib import PoseTracker, Custom
from Pose2Sim.poseEstimation import setup_backend_device, save_to_openpose

# Define face-related keypoint indices based on standard COCO-17/MMPose order
# nose, left_eye, right_eye, left_ear, right_ear
# (Refer to https://github.com/Tau-J/rtmlib/blob/main/rtmlib/visualization/skeleton/coco17.py)
FACE_KEYPOINT_INDECES = [0, 1, 2, 3, 4]

# Default settings
DEFAULT_MODE = 'lightweight'
DEFAULT_CONFIDENCE_THRESHOLD = 0.1 # Face keypoint confidence threshold; face keypoints basically have low confidence.
DEFAULT_BLUR_TYPE = 'blur' # 'blur' or 'black'
DEFAULT_BLUR_SHAPE = "rectangle" # Shape of the blurred area: 'polygon' or 'rectangle'
DEFAULT_BLUR_INTENCITY = "medium" # low, medium, high
DEFAULT_BLUR_SIZE = "small" # 'small', 'medium', 'large'
DEFAULT_SAVE_JSON = True # Save detected face keypoints to JSON files
DEFAULT_VISUALIZE = False # Display the processed video in real-time (set default to False)

# RTMO model information (refer to https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body.py)
RTMO_MODELS = {
    'lightweight': {
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip',
        'pose_input_size': (640, 640),
     },
     'balanced': {
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip',
        'pose_input_size': (640, 640),
     },
     'performance': {
        'pose': 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip',
        'pose_input_size': (640, 640),
     }
}

def face_blurring_func(**args):
    """Core logic for face blurring.
    Takes arguments as a dictionary.
    """
    input_video_path = args.get('input')
    output_video_path = args.get('output', None) # Use .get for defaults
    mode = args.get('mode', DEFAULT_MODE)
    conf_threshold = args.get('conf', DEFAULT_CONFIDENCE_THRESHOLD)
    blur_type = args.get('blur_type', DEFAULT_BLUR_TYPE)
    blur_shape = args.get('blur_shape', DEFAULT_BLUR_SHAPE)
    blur_size = args.get('blur_size', DEFAULT_BLUR_SIZE) # Retrieve blur_size
    blur_intensity = args.get('blur_intensity', DEFAULT_BLUR_INTENCITY) # Retrieve blur_intensity
    save_json = args.get('save_json', DEFAULT_SAVE_JSON)
    visualize = args.get('visualize', DEFAULT_VISUALIZE)

    # Setup basic logging
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(f"Processing video: {input_video_path}")
    start_time = time.time()

    # Setup backend and device automatically
    backend, device = setup_backend_device() # Use default values

    # --- RTMO Model Setup and Direct PoseTracker Initialization ---
    rtmo_pose_path = RTMO_MODELS[mode]['pose']
    rtmo_input_size = RTMO_MODELS[mode]['pose_input_size']

    # --- Use Custom solution with partial for RTMO --- 
    RTMO_Solution = partial(Custom,
                            pose_class='RTMO', # Specify RTMO class
                            pose=rtmo_pose_path, # Provide model path
                            pose_input_size=rtmo_input_size, # Provide input size
                            to_openpose=False, # RTMO uses MMPose format by default
                            backend=backend,
                            device=device)

    # Initialize PoseTracker with the configured Custom solution
    pose_tracker = PoseTracker(solution=RTMO_Solution, # Pass the partial object
                               tracking=False, # Tracking is not needed for face blurring
                               to_openpose=False # Consistent with solution setting
                               )

    # --- Model Setup Complete ---


    # Open video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {input_video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    if output_video_path is None:
        base_name = os.path.basename(input_video_path)
        name, ext = os.path.splitext(base_name)
        output_video_path = os.path.join(os.path.dirname(input_video_path), f"blurred_{name}{ext}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Setup JSON output directory if needed
    json_output_dir = None
    if save_json:
        video_name_wo_ext = os.path.splitext(os.path.basename(input_video_path))[0]
        # Change save location to a 'pose_blurred' subfolder within the original video folder
        pose_blurred_dir = os.path.join(os.path.dirname(input_video_path), 'pose_blurred')
        if not os.path.exists(pose_blurred_dir):
             os.makedirs(pose_blurred_dir)
        json_output_dir = os.path.join(pose_blurred_dir, f'{video_name_wo_ext}_face_json')
        if not os.path.exists(json_output_dir):
            os.makedirs(json_output_dir)
        # logging.info(f"Face keypoints JSON will be saved to: {json_output_dir}")

    if visualize:
        cv2.namedWindow("Face Blurring", cv2.WINDOW_NORMAL)

    # Process frames
    frame_idx = 0
    with tqdm(total=total_frames, desc="Blurring faces") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # 1. Detect poses (keypoints and scores)
            keypoints, scores = pose_tracker(frame) # keypoints: [N_persons, N_kpts, 2], scores: [N_persons, N_kpts]

            processed_frame = frame.copy() # Copy original frame for processing

            # Prepare data for JSON saving (only face keypoints)
            keypoints_for_json = np.zeros_like(keypoints)
            scores_for_json = np.zeros_like(scores)

            # 2. Iterate through detected persons to apply obscuration and prepare JSON data
            for person_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
                # 3. Extract face keypoints and scores
                face_kpts = person_kpts[FACE_KEYPOINT_INDECES]
                face_scores = person_scores[FACE_KEYPOINT_INDECES]

                # 4. Filter valid keypoints by confidence threshold
                valid_indices = np.where(face_scores >= conf_threshold)[0]
                valid_detected_kpts = face_kpts[valid_indices]
                # valid_detected_scores = face_scores[valid_indices] # Not used currently

                # --- Estimate Forehead and Chin Points --- 
                points_for_hull = valid_detected_kpts.copy().astype(int) # Start with valid detected points, ensure int

                # Check if we have enough points for estimation (at least eyes and nose)
                # Get indices relative to FACE_KEYPOINT_INDECES
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
                            logging.warning(f"Unknown blur_size '{blur_size}', plase select from 'small', 'medium', 'large'.")
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
                if save_json:
                    for i, kpt_idx in enumerate(FACE_KEYPOINT_INDECES):
                        if face_scores[i] >= conf_threshold:
                            # Copy only valid face keypoints and scores to the arrays for saving
                            keypoints_for_json[person_idx, kpt_idx] = person_kpts[kpt_idx]
                            scores_for_json[person_idx, kpt_idx] = person_scores[kpt_idx]
                # --- End JSON data preparation for this person --- 

            # 6. Save face keypoints to JSON using save_to_openpose (once per frame)
            if save_json:
                json_file_path = os.path.join(json_output_dir, f'{os.path.splitext(os.path.basename(input_video_path))[0]}_{frame_idx:06d}_face.json')
                # Ensure the directory exists before saving
                if not os.path.exists(os.path.dirname(json_file_path)):
                     os.makedirs(os.path.dirname(json_file_path))
                save_to_openpose(json_file_path, keypoints_for_json, scores_for_json)

            # 7. Visualize if enabled
            if visualize:
                cv2.imshow("Face Blurring", processed_frame)
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
        cv2.destroyAllWindows()

    logging.info("Face blurring process finished.")

    end_time = time.time()
    logging.info(f"Processing finished in {end_time - start_time:.2f} seconds.")


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
    parser = argparse.ArgumentParser(description='Detect and blur faces in a video using RTMO.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input video file (string).')
    parser.add_argument('-o', '--output', default=None, help='Path to the output video file (string). Defaults to "blurred_<input_name>".')
    parser.add_argument('--mode', default=DEFAULT_MODE, choices=RTMO_MODELS.keys(), help=f'Pose estimation mode (string, default: {DEFAULT_MODE}).')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help=f'Minimum confidence threshold for face keypoints (float, default: {DEFAULT_CONFIDENCE_THRESHOLD}).')
    parser.add_argument('--blur-type', default=DEFAULT_BLUR_TYPE, choices=['blur', 'black'], help=f'Type of obscuration (string, default: {DEFAULT_BLUR_TYPE}).')
    parser.add_argument('--blur-shape', default=DEFAULT_BLUR_SHAPE, choices=['polygon', 'rectangle'], help=f'Shape of the blurred area (string, default: {DEFAULT_BLUR_SHAPE}).')
    parser.add_argument('--blur-size', default=DEFAULT_BLUR_SIZE, choices=['small', 'medium', 'large'], help=f'Size of the estimated face area for padding (string, default: {DEFAULT_BLUR_SIZE}).')
    parser.add_argument('--blur-intensity', default=DEFAULT_BLUR_INTENCITY, choices=['low', 'medium', 'high'], help=f'Intensity of Gaussian blur (string: low, medium, high. default: {DEFAULT_BLUR_INTENCITY}).')
    parser.add_argument('--no-json', action='store_false', dest='save_json', help='Disable saving face keypoints to JSON files (boolean, default: {DEFAULT_SAVE_JSON}).')
    parser.add_argument('--visualize', action='store_true', help=f'Enable real-time visualization (boolean, default: {DEFAULT_VISUALIZE}).')

    args = parser.parse_args()

    face_blurring_func(
        input=args.input,
        output=args.output,
        mode=args.mode,
        conf=args.conf,
        blur_type=args.blur_type,
        blur_shape=args.blur_shape,
        blur_size=args.blur_size,
        blur_intensity=args.blur_intensity,
        save_json=args.save_json,
        visualize=args.visualize
    )

    # Example usage from terminal:
    # python Utilities/face_blurring.py -i path/to/your/video.mp4 -o path/to/output/blurred_video.mp4 --blur-type black --visualize

if __name__ == '__main__':
    main()