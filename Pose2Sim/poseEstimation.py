#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE ESTIMATION                                                       ##
###########################################################################

    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
'''


## INIT
import os
import glob
import json
import re
import logging
import ast
from tqdm import tqdm
from anytree import RenderTree
import numpy as np
import cv2

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton
from rtmlib.tools.object_detection.post_processings import nms
from Pose2Sim.common import natural_sort_key, sort_people_sports2d, sort_people_deepsort,\
                        colors, thickness, draw_bounding_box, draw_keypts, draw_skel, bbox_xyxy_compute, \
                        get_screen_size, calculate_display_size, is_video_file, is_image_file, get_max_workers
from Pose2Sim.skeletons import *

np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
import warnings # Silence numpy and CoreML warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN slice encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message=".*Input.*has a dynamic shape.*but the runtime shape.*has zero elements.*")

# Not safe, but to be used until OpenMMLab/RTMlib's SSL certificates are updated
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def setup_pose_tracker(ModelClass, det_frequency, mode, tracking, backend, device):
    '''
    Set up the RTMLib pose tracker with the appropriate model and backend.
    If CUDA is available, use it with ONNXRuntime backend; else use CPU with openvino

    INPUTS:
    - ModelClass: class. The RTMlib model class to use for pose detection (Body, BodyWithFeet, Wholebody)
    - det_frequency: int. The frequency of pose detection (every N frames)
    - mode: str. The mode of the pose tracker ('lightweight', 'balanced', 'performance')
    - tracking: bool. Whether to track persons across frames with RTMlib tracker
    - backend: str. The backend to use for pose detection (onnxruntime, openvino, opencv)
    - device: str. The device to use for pose detection (cpu, cuda, rocm, mps)

    OUTPUTS:
    - pose_tracker: PoseTracker. The initialized pose tracker object    
    '''

    backend, device = setup_backend_device(backend=backend, device=device)

    # Initialize the pose tracker with Halpe26 model
    pose_tracker = PoseTracker(
        ModelClass,
        det_frequency=det_frequency,
        mode=mode,
        backend=backend,
        device=device,
        tracking=tracking,
        to_openpose=False)
        
    return pose_tracker


def setup_model_class_mode(pose_model, mode, config_dict={}):
    '''
    Set up the pose model class and mode for the pose tracker.
    '''

    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation in {mode} mode.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation in {mode} mode.")
    elif pose_model.upper() =='HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info(f"Using FACE_106 model for pose estimation in {mode} mode.")
    elif pose_model.upper() =='ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info(f"Using ANIMAL2D_17 model for pose estimation in {mode} mode.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation in {mode} mode.")
    try:
        pose_model = eval(model_name)
    except:
        try: # from Config.toml
            from anytree.importer import DictImporter
            model_name = pose_model.upper()
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model)[0])
            if pose_model.id == 'None':
                pose_model.id = None
            logging.info(f"Using model {model_name} for pose estimation.")
        except:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
            from functools import partial
            try:
                mode = ast.literal_eval(mode)
            except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
                mode = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', mode) # changes "[640", "640]" to [640,640]
                mode = json.loads(mode)
            det_class = mode.get('det_class')
            det = mode.get('det_model')
            det_input_size = mode.get('det_input_size')
            pose_class = mode.get('pose_class')
            pose = mode.get('pose_model')
            pose_input_size = mode.get('pose_input_size')

            ModelClass = partial(Custom,
                        det_class=det_class, det=det, det_input_size=det_input_size,
                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size)
            logging.info(f"Using model {model_name} with the following custom parameters: {mode}.")

            if pose_class == 'RTMO' and model_name != 'COCO_17':
                logging.warning("RTMO currently only supports 'Body' pose_model. Switching to 'Body'.")
                pose_model = eval('COCO_17')
            
        except (json.JSONDecodeError, TypeError):
            logging.warning("Invalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'

    return pose_model, ModelClass, mode


def setup_backend_device(backend='auto', device='auto'):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device!='auto' and backend!='auto':
        device = device.lower()
        backend = backend.lower()

    if device=='auto' or backend=='auto':
        if device=='auto' and backend!='auto' or device!='auto' and backend=='auto':
            logging.warning(f"If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch
            import onnxruntime as ort
            if torch.cuda.is_available() == True and 'CUDAExecutionProvider' in ort.get_available_providers():
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info(f"Valid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info(f"Valid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise 
        except:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info(f"Valid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise
            except:
                device = 'cpu'
                backend = 'openvino'
                logging.info(f"No valid CUDA installation found: using OpenVINO backend with CPU.")

    else:
        logging.info(f"Using {device} device with {backend} backend.")    
    
    return backend, device


def save_to_openpose(json_file_path, keypoints, scores):
    '''
    Save the keypoints and scores to a JSON file in the OpenPose format

    INPUTS:
    - json_file_path: Path to save the JSON file
    - keypoints: Detected keypoints
    - scores: Confidence scores for each keypoint

    OUTPUTS:
    - JSON file with the detected keypoints and confidence scores in the OpenPose format
    '''

    # Prepare keypoints with confidence scores for JSON output
    nb_detections = len(keypoints)
    # print('results: ', keypoints, scores)
    detections = []
    for i in range(nb_detections): # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
        detections.append({
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_with_confidence_i,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                })
            
    # Create JSON output structure
    json_output = {"version": 1.3, "people": detections}
    
    # Save JSON output for each frame
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    os.makedirs(json_output_dir, exist_ok=True)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def process_video(video_path, pose_tracker, pose_model, frame_range, average_likelihood_threshold_pose, output_format, save_video, save_images, display_detection, tracking_mode, max_distance_px, deepsort_tracker):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - frame_range: list. Range of frames to process
    - average_likelihood_threshold_pose: float. If the average confidence score of the detected keypoints for a person is below this threshold, the person will be dropped (default: 0.5)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - max_distance_px: int. The maximum distance in pixels for associating detections across frames in sports2d tracking mode (default: 100)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    cap = cv2.VideoCapture(video_path)
    cap.read()
    if cap.read()[0] == False:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    os.makedirs(pose_dir, exist_ok=True)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the width and height from the raw video

    if save_video: # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = round(cap.get(cv2.CAP_PROP_FPS)) # Get the frame rate from the raw video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file
        
    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(video_path)}", display_width, display_height)

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[0,total_frames] if frame_range in ('all', 'auto', []) else frame_range][0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_range[0])
    frame_idx = f_range[0]

    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
    kpt_id_max = max(keypoints_ids)+1

    with tqdm(iterable=range(*f_range), desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            if frame_idx in range(*f_range):
                # print('\nFrame ', frame_idx)
                success, frame = cap.read()
                if not success:
                    break
            
                try: # Frames with no detection cause errors on MacOS CoreMLExecutionProvider
                    # Detect poses
                    keypoints, scores = pose_tracker(frame)

                    # Non maximum suppression (at pose level, not detection, and only using likely keypoints)
                    frame_shape = frame.shape
                    mask_scores = np.mean(scores, axis=1) > 0.2

                    likely_keypoints = np.where(mask_scores[:, np.newaxis, np.newaxis], keypoints, np.nan)
                    likely_scores = np.where(mask_scores[:, np.newaxis], scores, np.nan)
                    likely_bboxes = bbox_xyxy_compute(frame_shape, likely_keypoints, padding=0)
                    score_likely_bboxes = np.nanmean(np.where(np.isnan(likely_scores), 0, likely_scores), axis=1)

                    valid_indices = np.where(score_likely_bboxes > average_likelihood_threshold_pose)[0]
                    if len(valid_indices) > 0:
                        valid_bboxes = likely_bboxes[valid_indices]
                        valid_scores = score_likely_bboxes[valid_indices]
                        keep_valid = nms(valid_bboxes, valid_scores, nms_thr=0.45)
                        keep = valid_indices[keep_valid]
                    else:
                        keep = []
                    keypoints, scores = likely_keypoints[keep], likely_scores[keep]

                    # Track poses across frames
                    if tracking_mode == 'deepsort':
                        keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                    if tracking_mode == 'sports2d': 
                        if 'prev_keypoints' not in locals(): 
                            prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)
                    else:
                        pass

                except:
                    keypoints = np.full((1,kpt_id_max,2), fill_value=np.nan)
                    scores = np.full((1,kpt_id_max), fill_value=np.nan)
                    
                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton on the frame
                if display_detection or save_video or save_images:
                    # try:
                    #     # MMPose skeleton
                    #     img_show = frame.copy()
                    #     img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                    # except:
                        # Sports2D skeleton
                        valid_X, valid_Y, valid_scores = [], [], []
                        for person_keypoints, person_scores in zip(keypoints, scores):
                            person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                            valid_X.append(person_X)
                            valid_Y.append(person_Y)
                            valid_scores.append(person_scores)
                        img_show = frame.copy()
                        img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                        img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                        img_show = draw_skel(img_show, valid_X, valid_Y, pose_model)
                
                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.jpg'), img_show)

                frame_idx += 1
                pbar.update(1)

            if frame_idx >= f_range[1]:
                break

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def process_video_worker(video_path, ModelClass, det_frequency, mode, backend, device,
                           pose_model, frame_range, average_likelihood_threshold_pose, 
                           output_format, save_video, save_images, display_detection, 
                           tracking_mode, max_distance_px, deepsort_params, init_lock=None):
    '''
    Worker function for parallel pose estimation. Creates its own PoseTracker
    and optional DeepSort tracker, then processes one video independently.
    '''
    if init_lock is not None:
        with init_lock:
            pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
    else:
        pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
    deepsort_tracker = None
    if tracking_mode == 'deepsort' and multi_person:
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deepsort_tracker = DeepSort(**deepsort_params)
    process_video(video_path, pose_tracker, pose_model, frame_range, average_likelihood_threshold_pose, 
                  output_format, save_video, save_images, display_detection,
                  tracking_mode, max_distance_px, deepsort_tracker)


def process_images(image_folder_path, pose_tracker, pose_model, output_format, fps, save_video, save_images, display_detection, frame_range, tracking_mode, max_distance_px, deepsort_tracker):
    '''
    Estimate pose estimation from a folder of images
    
    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''    

    pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    os.makedirs(pose_dir, exist_ok=True)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = sorted([f for f in glob.glob(os.path.join(image_folder_path, '*')) if is_image_file(f)], key=natural_sort_key)
    if len(image_files) == 0:
        raise NameError(f'No image files found in {image_folder_path}.')

    if save_video or display_detection:
        first_frame = cv2.imread(image_files[0])
        H, W = first_frame.shape[:2]

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file

    if display_detection:
        screen_width, screen_height = get_screen_size()
        display_width, display_height = calculate_display_size(W, H, screen_width, screen_height, margin=50)
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", display_width, display_height)
    
    # Retrieve keypoint names from model
    keypoints_ids = [node.id for _, _, node in RenderTree(pose_model) if node.id!=None]
    kpt_id_max = max(keypoints_ids)+1
    
    f_range = [[0,len(image_files)] if frame_range in ('all', 'auto', []) else frame_range][0]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            try:
                frame = cv2.imread(image_file)
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            try:
                # Detect poses
                keypoints, scores = pose_tracker(frame)

                # Track poses across frames
                if tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                if tracking_mode == 'sports2d': 
                    if 'prev_keypoints' not in locals(): 
                        prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores, max_dist=max_distance_px)
            except:
                keypoints = np.full((1,kpt_id_max,2), fill_value=np.nan)
                scores = np.full((1,kpt_id_max), fill_value=np.nan)

            # Extract frame number from the filename
            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            # Draw skeleton on the image
            if display_detection or save_video or save_images:
                try:
                    # MMPose skeleton
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                except:
                    # Sports2D skeleton
                    valid_X, valid_Y, valid_scores = [], [], []
                    for person_keypoints, person_scores in zip(keypoints, scores):
                        person_X, person_Y = person_keypoints[:, 0], person_keypoints[:, 1]
                        valid_X.append(person_X)
                        valid_Y.append(person_Y)
                        valid_scores.append(person_scores)
                    img_show = frame.copy()
                    img_show = draw_bounding_box(img_show, valid_X, valid_Y, colors=colors, fontSize=2, thickness=thickness)
                    img_show = draw_keypts(img_show, valid_X, valid_Y, valid_scores, cmap_str='RdYlGn')
                    img_show = draw_skel(img_show, valid_X, valid_Y, pose_model)

            if display_detection:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                out.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def estimate_pose_all(config_dict):
    '''
    Estimate pose from a video file or a folder of images and 
    write the results to JSON files, videos, and/or images.
    Results can optionally be displayed in real time.

    Supported models: HALPE_26 (default, body and feet), COCO_133 (body, feet, hands), COCO_17 (body)
    Supported modes: lightweight, balanced, performance (edit paths at rtmlib/tools/solutions if you 
    need nother detection or pose models)

    Optionally gives consistent person ID across frames (slower but good for 2D analysis)
    Optionally runs detection every n frames and inbetween tracks points (faster but less accurate).

    If a valid cuda installation is detected, uses the GPU with the ONNXRuntime backend. Otherwise, 
    uses the CPU with the OpenVINO backend.

    INPUTS:
    - videos or image folders from the video directory
    - a Config.toml file

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - Optionally, videos and/or image files with the detected keypoints 
    '''

    # Read config
    project_dir = config_dict.get('project', {}).get('project_dir', '.')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    frame_range = config_dict.get('project', {}).get('frame_range', 'auto')
    multi_person = config_dict.get('project', {}).get('multi_person', False)
    max_workers_input = config_dict.get('pose', {}).get('parallel_workers_pose', False)
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    pose_model = config_dict.get('pose', {}).get('pose_model', 'Body_with_feet')
    mode = config_dict.get('pose', {}).get('mode', 'balanced')
    
    output_format = config_dict.get('pose', {}).get('output_format', 'openpose')
    save_video = True if 'to_video' in config_dict.get('pose', {}).get('save_video', 'to_video') else False
    save_images = True if 'to_images' in config_dict.get('pose', {}).get('save_video', 'to_video') else False
    det_frequency = config_dict.get('pose', {}).get('det_frequency', 4)
    backend = config_dict.get('pose', {}).get('backend', 'auto')
    device = config_dict.get('pose', {}).get('device', 'auto')
    display_detection = config_dict.get('pose', {}).get('display_detection', True)
    overwrite_pose = config_dict.get('pose', {}).get('overwrite_pose', False)
    average_likelihood_threshold_pose = config_dict.get('pose', {}).get('average_likelihood_threshold_pose', 0.5)
    tracking_mode = config_dict.get('pose', {}).get('tracking_mode', 'sports2d')
    max_distance_px = config_dict.get('pose', {}).get('max_distance_px', 100)
    if tracking_mode == 'deepsort' and multi_person:
        deepsort_params = config_dict.get('pose', {}).get('deepsort_params', '{}')
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params) # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        from deep_sort_realtime.deepsort_tracker import DeepSort
        deepsort_tracker = DeepSort(**deepsort_params)
    else:
        deepsort_tracker = None
        deepsort_params = {}

    # Determine frame rate
    video_files = sorted([f for f in glob.glob(os.path.join(video_dir, '*')) if is_video_file(f)], key=natural_sort_key)
    frame_rate = config_dict.get('project', {}).get('frame_rate', 'auto')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            logging.warning(f'Cannot read video. Frame rate will be set to 30 fps.')
            frame_rate = 30  

    # Select the appropriate model based on the model_type
    logging.info('Estimating pose...\n')
    pose_model_name = pose_model
    pose_model, ModelClass, mode = setup_model_class_mode(pose_model, mode, config_dict)

    # Estimate pose
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        if not overwrite_pose:
            logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
        else:
            logging.info('Overwriting previous pose estimation. Set overwrite_pose to false in Config.toml if you want to keep the previous results.')
            raise
            
    except:
        # Set up model and mode
        logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
        logging.info(f'Mode: {mode}.')
        
        # Set up detection frequency
        low_likelihood_message = f'Detections are dropped if their average keypoint likelihood is below {average_likelihood_threshold_pose}.'
        if det_frequency>1:
            logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points. {low_likelihood_message}')
        elif det_frequency==1:
            logging.info(f'Inference run on every single frame. {low_likelihood_message}')
        else:
            raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")
        
        # Select device and backend
        backend, device = setup_backend_device(backend=backend, device=device)
        
        # Set up parallelization
        if not isinstance(max_workers_input, int) and max_workers_input != 'auto' and max_workers_input != False:
            raise ValueError(f"Invalid parallel_workers_pose value: {max_workers_input}. Must be an integer greater or equal to 1, 'auto', or false.")
        if max_workers_input == 1 or max_workers_input == False:
            logging.info(f'Pose estimation will not be processed in parallel as parallel_workers_pose is set to {max_workers_input}.')
        else:
            if display_detection:
                logging.warning(f'Cannot run pose estimation in parallel with display_detection=true. Set it to false for faster pose estimation.')
                parallel_pose = 1
            elif device not in {'cpu', 'cuda', 'mps', 'rocm'}:
                logging.warning(f'Parallel pose estimation is not supported for device "{device.upper()}": falling back to sequential.')
                parallel_pose = 1
            else:
                max_workers_calc = get_max_workers(device)
                requested = max_workers_calc if max_workers_input == 'auto' else int(max_workers_input)
                parallel_pose = min(requested, len(video_files))
                if requested > max_workers_calc:
                    logging.warning(f'Going with the requested {requested} workers, but the recommended limit with this hardware is {max_workers_calc}.')
                else:
                    logging.info(f'Using {parallel_pose} parallel workers.')
                    
        # Tracking
        if tracking_mode not in ['deepsort', 'sports2d']:
            logging.warning(f"Tracking mode {tracking_mode} not recognized. Using sports2d method.")
            tracking_mode = 'sports2d'
        logging.info(f'Tracking is performed with {tracking_mode}{"" if not tracking_mode=="deepsort" else f" with parameters: {deepsort_params}"}.\n')

        if not len(video_files) == 0:
            # Process video files
            logging.info(f'Found {len(video_files)} video files in {video_dir}.')

            # In parallel
            if parallel_pose != 1 and len(video_files) > 1:
                import threading
                init_lock = threading.Lock()
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=parallel_pose) as executor:
                    futures = {
                        executor.submit(
                            process_video_worker, video_path,
                            ModelClass, det_frequency, mode, backend, device,
                            pose_model, frame_range, average_likelihood_threshold_pose, 
                            output_format, save_video, save_images, display_detection, tracking_mode,
                            max_distance_px, deepsort_params, init_lock
                        ): video_path for video_path in video_files
                    }
                    for future in as_completed(futures):
                        video_path = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logging.error(f'Failed processing {os.path.basename(video_path)}: {e}')
                            raise RuntimeError(f'Failed processing {os.path.basename(video_path)}: {e}')
    
            # Sequentially
            else:
                try:
                    pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
                except:
                    logging.error('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
                    raise ValueError('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
                for video_path in video_files:
                    pose_tracker.reset()
                    if tracking_mode == 'deepsort':
                        deepsort_tracker.tracker.delete_all_tracks()
                    process_video(video_path, pose_tracker, pose_model, frame_range, average_likelihood_threshold_pose, output_format, save_video, save_images, display_detection, tracking_mode, max_distance_px, deepsort_tracker)

        else:
            # Process image folders
            image_folders = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
            # Keep only folders that contain at least one image file
            image_folders = [folder for folder in image_folders 
                            if any(is_image_file(os.path.join(folder, f)) for f in os.listdir(folder))]
            if len(image_folders) == 0:
                raise NameError(f'No video files and no image folders with recognized image extensions found in {video_dir}.')
            else:
                logging.info(f'No video files found. Found {len(image_folders)} image folders in {video_dir}.')