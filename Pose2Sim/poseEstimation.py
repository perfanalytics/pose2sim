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
from functools import partial
from tqdm import tqdm
from anytree.importer import DictImporter
import cv2

from rtmlib import PoseTracker, BodyWithFeet, Wholebody, Body, Hand, Custom, draw_skeleton
from deep_sort_realtime.deepsort_tracker import DeepSort
from Pose2Sim.common import natural_sort_key, sort_people_sports2d, sort_people_deepsort,\
                        colors, thickness, draw_bounding_box, draw_keypts, draw_skel
from Pose2Sim.skeletons import *


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
                logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
            elif torch.cuda.is_available() == True and 'ROCMExecutionProvider' in ort.get_available_providers():
                device = 'rocm'
                backend = 'onnxruntime'
                logging.info(f"\nValid ROCM installation found: using ONNXRuntime backend with GPU.")
            else:
                raise 
        except:
            try:
                import onnxruntime as ort
                if 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
                    device = 'mps'
                    backend = 'onnxruntime'
                    logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
                else:
                    raise
            except:
                device = 'cpu'
                backend = 'openvino'
                logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")
        
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
    if not os.path.isdir(json_output_dir): os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def process_video(video_path, pose_tracker, pose_model, output_format, save_video, save_images, display_detection, frame_range, multi_person, tracking_mode, deepsort_tracker):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - multi_person: bool. Whether to detect multiple people in the video
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    try:
        cap = cv2.VideoCapture(video_path)
        cap.read()
        if cap.read()[0] == False:
            raise
    except:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    if save_video: # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = round(cap.get(cv2.CAP_PROP_FPS)) # Get the frame rate from the raw video
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the width and height from the raw video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file
        
    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[total_frames] if frame_range in ('all', 'auto', []) else frame_range][0]
    with tqdm(iterable=range(*f_range), desc=f'Processing {os.path.basename(video_path)}') as pbar:
        frame_count = 0
        while cap.isOpened():
            # print('\nFrame ', frame_idx)
            success, frame = cap.read()
            frame_count += 1
            if not success:
                break
            
            if frame_idx in range(*f_range):
                # Detect poses
                keypoints, scores = pose_tracker(frame)

                # Track poses across frames
                if multi_person:
                    if tracking_mode == 'deepsort':
                        keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_count)
                    if tracking_mode == 'sports2d': 
                        if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                        prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores)
                    
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

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def process_images(image_folder_path, vid_img_extension, pose_tracker, pose_model, output_format, fps, save_video, save_images, display_detection, frame_range, multi_person, tracking_mode, deepsort_tracker):
    '''
    Estimate pose estimation from a folder of images
    
    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - vid_img_extension: str. Extension of the image files
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - pose_model: str. The pose model to use for pose estimation (HALPE_26, COCO_133, COCO_17)
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process
    - multi_person: bool. Whether to detect multiple people in the video
    - tracking_mode: str. The tracking mode to use for person tracking (deepsort, sports2d)
    - deepsort_tracker: DeepSort tracker object or None

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''    

    pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = glob.glob(os.path.join(image_folder_path, '*'+vid_img_extension))
    sorted(image_files, key=natural_sort_key)

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        W, H = cv2.imread(image_files[0]).shape[:2][::-1] # Get the width and height from the first image (assuming all images have the same size)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file

    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
    
    f_range = [[len(image_files)] if frame_range in ('all', 'auto', []) else frame_range][0]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):
            try:
                frame = cv2.imread(image_file)
                frame_idx += 1
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            # Detect poses
            keypoints, scores = pose_tracker(frame)

            # Track poses across frames
            if multi_person:
                if tracking_mode == 'deepsort':
                    keypoints, scores = sort_people_deepsort(keypoints, scores, deepsort_tracker, frame, frame_idx)
                if tracking_mode == 'sports2d': 
                    if 'prev_keypoints' not in locals(): prev_keypoints = keypoints
                    prev_keypoints, keypoints, scores = sort_people_sports2d(prev_keypoints, keypoints, scores=scores)
                    
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
    project_dir = config_dict['project']['project_dir']
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    frame_range = config_dict.get('project').get('frame_range')
    multi_person = config_dict.get('project').get('multi_person')
    video_dir = os.path.join(project_dir, 'videos')
    pose_dir = os.path.join(project_dir, 'pose')

    pose_model = config_dict['pose']['pose_model']
    mode = config_dict['pose']['mode'] # lightweight, balanced, performance
    vid_img_extension = config_dict['pose']['vid_img_extension']
    
    output_format = config_dict['pose']['output_format']
    save_video = True if 'to_video' in config_dict['pose']['save_video'] else False
    save_images = True if 'to_images' in config_dict['pose']['save_video'] else False
    display_detection = config_dict['pose']['display_detection']
    overwrite_pose = config_dict['pose']['overwrite_pose']
    det_frequency = config_dict['pose']['det_frequency']
    tracking_mode = config_dict.get('pose').get('tracking_mode')
    if tracking_mode == 'deepsort' and multi_person:
        deepsort_params = config_dict.get('pose').get('deepsort_params')
        try:
            deepsort_params = ast.literal_eval(deepsort_params)
        except: # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
            deepsort_params = deepsort_params.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/',':/').replace('":"\\',':\\')
            deepsort_params = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', deepsort_params) # changes "[640", "640]" to [640,640]
            deepsort_params = json.loads(deepsort_params)
        deepsort_tracker = DeepSort(**deepsort_params)
    else:
        deepsort_tracker = None
    backend = config_dict['pose']['backend']
    device = config_dict['pose']['device']

    # Determine frame rate
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            if not cap.isOpened():
                raise FileNotFoundError(f'Error: Could not open {video_files[0]}. Check that the file exists.')
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
            if frame_rate == 0:
                frame_rate = 30
                logging.warning(f'Error: Could not retrieve frame rate from {video_files[0]}. Defaulting to 30fps.')
        except:
            frame_rate = 30

    # Set detection frequency
    if det_frequency>1:
        logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points.')
    elif det_frequency==1:
        logging.info(f'Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")

    # Select the appropriate model based on the model_type
    logging.info('\nEstimating pose...')
    if pose_model.upper() in ('HALPE_26', 'BODY_WITH_FEET'):
        model_name = 'HALPE_26'
        ModelClass = BodyWithFeet # 26 keypoints(halpe26)
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() in ('COCO_133', 'WHOLE_BODY', 'WHOLE_BODY_WRIST'):
        model_name = 'COCO_133'
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() in ('COCO_17', 'BODY'):
        model_name = 'COCO_17'
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    elif pose_model.upper() =='HAND':
        model_name = 'HAND_21'
        ModelClass = Hand
        logging.info(f"Using HAND_21 model for pose estimation.")
    elif pose_model.upper() =='FACE':
        model_name = 'FACE_106'
        logging.info(f"Using FACE_106 model for pose estimation.")
    elif pose_model.upper() =='ANIMAL':
        model_name = 'ANIMAL2D_17'
        logging.info(f"Using ANIMAL2D_17 model for pose estimation.")
    else:
        model_name = pose_model.upper()
        logging.info(f"Using model {model_name} for pose estimation.")
    pose_model_name = pose_model
    try:
        pose_model = eval(model_name)
    except:
        try: # from Config.toml
            pose_model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if pose_model.id == 'None':
                pose_model.id = None
        except:
            raise NameError(f'{pose_model} not found in skeletons.py nor in Config.toml')

    # Select device and backend
    backend, device = setup_backend_device(backend=backend, device=device)

    # Manually select the models if mode is a dictionary rather than 'lightweight', 'balanced', or 'performance'
    if not mode in ['lightweight', 'balanced', 'performance'] or 'ModelClass' not in locals():
        try:
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
                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size,
                        backend=backend, device=device)
            
        except (json.JSONDecodeError, TypeError):
            logging.warning("\nInvalid mode. Must be 'lightweight', 'balanced', 'performance', or '''{dictionary}''' of parameters within triple quotes. Make sure input_sizes are within square brackets.")
            logging.warning('Using the default "balanced" mode.')
            mode = 'balanced'


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
        # Set up pose tracker
        try:
            pose_tracker = setup_pose_tracker(ModelClass, det_frequency, mode, False, backend, device)
        except:
            logging.error('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')
            raise ValueError('Error: Pose estimation failed. Check in Config.toml that pose_model and mode are valid.')

        if tracking_mode not in ['deepsort', 'sports2d']:
            logging.warning(f"Tracking mode {tracking_mode} not recognized. Using sports2d method.")
            tracking_mode = 'sports2d'
        logging.info(f'\nPose tracking set up for "{pose_model_name}" model.')
        logging.info(f'Mode: {mode}.')
        logging.info(f'Tracking is performed with {tracking_mode}{"" if not tracking_mode=="deepsort" else f" with parameters: {deepsort_params}"}.\n')

        video_files = sorted(glob.glob(os.path.join(video_dir, '*'+vid_img_extension)))
        if not len(video_files) == 0: 
            # Process video files
            logging.info(f'Found video files with {vid_img_extension} extension.')
            for video_path in video_files:
                pose_tracker.reset()
                if tracking_mode == 'deepsort': deepsort_tracker.tracker.delete_all_tracks()
                process_video(video_path, pose_tracker, pose_model, output_format, save_video, save_images, display_detection, frame_range, multi_person, tracking_mode, deepsort_tracker)

        else:
            # Process image folders
            logging.info(f'Found image folders with {vid_img_extension} extension.')
            image_folders = sorted([f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])
            for image_folder in image_folders:
                pose_tracker.reset()
                image_folder_path = os.path.join(video_dir, image_folder)
                if tracking_mode == 'deepsort': deepsort_tracker.tracker.delete_all_tracks()                
                process_images(image_folder_path, vid_img_extension, pose_tracker, pose_model, output_format, frame_rate, save_video, save_images, display_detection, frame_range, multi_person, tracking_mode, deepsort_tracker)
