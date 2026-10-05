#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ########################################################
    ## Run Mediapipe BlazePose and save coordinates       ##
    ########################################################

    Runs BlazePose (Mediapipe Tasks API) on a video or a webcam stream.
    Saves 2D pixel coordinates to json files, (OpenPose format), csv/h5 (DeepLabCut format),
    or 3D metric estimated coordinates to trc (Pose2Sim/OpenSim format, with keypoints defined in the BLAZEPOSE tree)
    Optionally displays and saves images with keypoints overlayed

    N.B.: First install mediapipe: `uv pip install mediapipe`
    If you need to save as .h5, also install tables: `uv pip install tables`
    The .task model file is downloaded on the first run.

    Usage:
    Mediapipe_to_pose_or_trc -i input_file --display --save_images --save_video --to_csv --to_h5 --to_json --to_trc --model_complexity 2 -o output_folder
    OR Mediapipe_to_pose_or_trc -i 0 -dTv (on webcam, display, save video, save trc)
    OR from Pose2Sim.Utilities.Mediapipe_to_pose_or_trc import mediapipe_to_pose_or_trc_func; mediapipe_to_pose_or_trc_func(input='input.mp4', display=True, save_images=True, save_video=True, to_csv=True, to_h5=True, to_json=True, to_trc=True, model_complexity=2, output_folder='output')`
'''


## INIT
import os
import av
import cv2
import json
import argparse
import urllib.request
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from anytree import Node, PreOrderIter
from importlib.metadata import version

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## SKELETON AND MODEL DEFINITION
BLAZEPOSE = Node("Hip", id=None, children=[
    Node("RHip", id=24, children=[
        Node("RKnee", id=26, children=[
            Node("RAnkle", id=28, children=[
                Node("RHeel", id=30),
                Node("RBigToe", id=32),
            ]),
        ]),
    ]),
    Node("LHip", id=23, children=[
        Node("LKnee", id=25, children=[
            Node("LAnkle", id=27, children=[
                Node("LHeel", id=29),
                Node("LBigToe", id=31),
            ]),
        ]),
    ]),
    Node("Nose", id=0, children=[
        Node("REye", id=5),
        Node("LEye", id=2),
    ]),
    Node("RShoulder", id=12, children=[
        Node("RElbow", id=14, children=[
            Node("RWrist", id=16, children=[
                Node("RPinky", id=18),
                Node("RIndex", id=20),
                Node("RThumb", id=22),
            ]),
        ]),
    ]),
    Node("LShoulder", id=11, children=[
        Node("LElbow", id=13, children=[
            Node("LWrist", id=15, children=[
                Node("LPinky", id=17),
                Node("LIndex", id=19),
                Node("LThumb", id=21),
            ]),
        ]),
    ]),
])
tree = BLAZEPOSE

MODEL_URLS = {
    0: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    1: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
    2: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
}


## FUNCTIONS
def setup_webcam(webcam_id):
    '''
    Set up webcam capture with OpenCV.

    INPUTS:
    - webcam_id: int. The ID of the webcam to capture from
    
    OUTPUTS:
    - cap: cv2.VideoCapture. The webcam capture object
    - cam_width: int. The actual width of the webcam frame
    - cam_height: int. The actual height of the webcam frame
    - fps: int. The frame rate of the webcam
    '''

    cap = cv2.VideoCapture(webcam_id)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open webcam #{webcam_id}. Make sure that your webcam is available and has the right id.")

    # set width and height to closest available for the webcam
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30

    return cap, cam_width, cam_height, fps


def resample_video(vid_output_path, desired_framerate=30, fps=240):
    '''
    Resample video to the desired fps using av (PyAV).
    '''

    vid_output_path = Path(vid_output_path)
    new_vid_path = vid_output_path.parent / Path(vid_output_path.stem + '_2' + vid_output_path.suffix)
    pts_factor = fps / desired_framerate

    with av.open(str(vid_output_path)) as in_container:
        in_stream = in_container.streams.video[0]
        codec_name = in_stream.codec_context.name
        pix_fmt = in_stream.pix_fmt or 'yuv420p'

        with av.open(str(new_vid_path), mode='w') as out_container:
            out_stream = out_container.add_stream(codec_name, rate=desired_framerate)
            out_stream.width = in_stream.width
            out_stream.height = in_stream.height
            out_stream.pix_fmt = pix_fmt

            for frame in in_container.decode(video=0):
                if frame.pts is not None:
                    frame.pts = int(frame.pts * pts_factor)
                frame.dts = None
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)
            for packet in out_stream.encode():
                out_container.mux(packet)

    vid_output_path.unlink()
    new_vid_path.rename(vid_output_path)
    

def download_model(model_path, complexity=2):
    '''
    Download the pose landmarker .task model if not present locally.
    '''

    model_path = Path(model_path)
    if model_path.exists():
        return str(model_path)
    url = MODEL_URLS.get(complexity, MODEL_URLS[2])
    print(f'Downloading model (complexity={complexity}) to {model_path}...')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, model_path)
    print('Model downloaded.')
    return str(model_path)


def draw_keypoints_on_frame(frame, keypoints, w, h):
    '''
    Draw the tree-defined skeleton on a BGR frame.

    INPUTS:
    - frame: BGR image (numpy array)
    - keypoints: full list of MediaPipe NormalizedLandmark objects (33)
    - w, h: frame width and height
    '''

    for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        i, j = connection.start, connection.end
        lm_i, lm_j = keypoints[i], keypoints[j]
        if lm_i.visibility > 0.5 and lm_j.visibility > 0.5:
            pt1 = (int(lm_i.x * w), int(lm_i.y * h))
            pt2 = (int(lm_j.x * w), int(lm_j.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    for lm in keypoints:
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


def save_to_csv_or_h5(kpt_list, output_folder, video_name, to_csv, to_h5):
    '''
    Saves 2D Mediapipe BlazePose keypoint coordinates to csv or h5 file,
    in the DeepLabCut format.

    INPUTS:
    - kpt_list: List of lists of keypoints X and Y coordinates and likelihood, for each frame
    - output_folder: Folder where to save the csv or h5 file
    - video_name: Name of the video
    - to_csv: Boolean, whether to save to csv
    - to_h5: Boolean, whether to save to h5

    OUTPUTS:
    - Creation of csv or h5 file in output_folder
    '''

    # Prepare dataframe file
    all_landmarks_number = len(vision.PoseLandmark)
    scorer = ['DavidPagnon'] * all_landmarks_number * 3
    individuals = ['person'] * all_landmarks_number * 3
    bodyparts = [[landmark.name] * 3 for landmark in vision.PoseLandmark]
    bodyparts = [item for sublist in bodyparts for item in sublist]
    coords = ['x', 'y', 'likelihood'] * all_landmarks_number
    tuples = list(zip(scorer, individuals, bodyparts, coords))
    index_csv = pd.MultiIndex.from_tuples(tuples, names=['scorer', 'individuals', 'bodyparts', 'coords'])
    Q = pd.DataFrame(np.array(kpt_list).T, index=index_csv).T

    if to_csv:
        csv_file = Path(output_folder) / (video_name+'.csv')
        Q.to_csv(csv_file, sep=',', index=True, lineterminator='\n')

    if to_h5:
        h5_file = Path(output_folder) / (video_name+'.h5')
        Q.to_hdf(h5_file, index=True, key='mediapipe_detection')


def save_to_json(kpt_list, output_folder, video_name):
    '''
    Saves blazepose keypoint coordinates to json files, in the OpenPose format.

    INPUTS:
    - kpt_list: List of lists of keypoints X and Y coordinates and likelihood, for each frame
    - output_folder: Folder where to save the json files
    - video_name: Name of the video

    OUTPUTS:
    - Creation of json files in output_folder
    '''

    json_folder = Path(output_folder) / ('mediapipe_'+video_name+'_json')
    if not Path(json_folder).exists():
        os.mkdir(json_folder)
    print(json_folder)

    # json preparation
    json_dict = {'version':1.3, 'people':[]}
    json_dict['people'] = [{'person_id':[-1], 
                    'pose_keypoints_2d': [], 
                    'face_keypoints_2d': [], 
                    'hand_left_keypoints_2d':[], 
                    'hand_right_keypoints_2d':[], 
                    'pose_keypoints_3d':[], 
                    'face_keypoints_3d':[], 
                    'hand_left_keypoints_3d':[], 
                    'hand_right_keypoints_3d':[]}]
    
    # write each h5 line in json file
    for frame, kpt in enumerate(kpt_list):
        json_dict['people'][0]['pose_keypoints_2d'] = kpt
        json_file = Path(json_folder) / ( 'mediapipe_'+video_name+'.'+str(frame).zfill(5)+'.json')
        with open(json_file, 'w') as js_f:
            js_f.write(json.dumps(json_dict))


def write_trc(trc_path, trc_data, frames_col, time_col, header):
    '''
    Write a .trc file (OpenSim marker trajectory file).

    INPUTS:
    - trc_path: path to the output .trc file
    - trc_data: pandas DataFrame of filtered data columns
    - frames_col: pandas Series of frame numbers
    - time_col: pandas Series of time values
    - header: list of header lines
    '''

    try:
        with open(trc_path, 'w') as trc_o:
            for line in header:
                trc_o.write(line)

            all_trc_data = pd.concat(
                [pd.Series(frames_col, name='Frame#').reset_index(drop=True),
                 pd.Series(time_col, name='Time').reset_index(drop=True),
                 trc_data.reset_index(drop=True)
                ], axis=1)
            all_trc_data.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    except Exception as e:
        raise ValueError(f"Error writing TRC file at {trc_path}: {e}")


def save_to_trc(kpt_3d_list, output_folder, video_name, fps, tree=BLAZEPOSE):
    '''
    Saves Mediapipe BlazePose 3D metric keypoint coordinates to a .trc file (OpenSim format).
    Only the keypoints defined in the tree are written.
    '''

    nodes = [n for n in PreOrderIter(tree) if n.id is not None]
    keypoint_ids = sorted(n.id for n in nodes)
    name_by_id = {n.id: n.name for n in nodes}
    keypoint_names = [name_by_id[i] for i in keypoint_ids]

    nb_all = len(vision.PoseLandmark)
    kpt_array = np.array(kpt_3d_list).reshape(-1, nb_all, 3)
    subset = kpt_array[:, keypoint_ids, :]
    data = {}
    for i, name in enumerate(keypoint_names):
        # X->-Z, Y->-Y, Z->-X
        data[f'{name}_X'] = - subset[:, i, 2]
        data[f'{name}_Y'] = - subset[:, i, 1]
        data[f'{name}_Z'] = - subset[:, i, 0]
    Q = pd.DataFrame(data)

    num_frames = len(Q)
    frames_col = pd.Series(range(num_frames))
    time_col = pd.Series(np.arange(num_frames) / fps)

    nb_keypoints = len(keypoint_names)

    header = []
    header.append(f'PathFileType\t4\t(X/Y/Z)\t{video_name}.trc\n')
    header.append('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    header.append(f'{fps}\t{fps}\t{num_frames}\t{nb_keypoints}\tm\t{fps}\t1\t{num_frames}\n')
    header.append('Frame#\tTime\t' + '\t'.join(f'{name}\t\t' for name in keypoint_names) + '\n')
    header.append('\t\t' + '\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(nb_keypoints)]) + '\n')
    header.append('\n')

    trc_path = Path(output_folder) / (video_name + '.trc')
    write_trc(trc_path, Q, frames_col, time_col, header)


def mediapipe_to_pose_or_trc_func(**args):
    '''
    Runs BlazePose (Mediapipe Tasks API) on a video
    Saves 2D pixel coordinates to json files, (OpenPose format), csv/h5 (DeepLabCut format),
    or 3D metric estimated coordinates to trc (Pose2Sim/OpenSim format, with keypoints defined in the BLAZEPOSE tree)
    Optionally displays and saves images with keypoints overlayed

    N.B.: First install mediapipe: `uv pip install mediapipe`
    If you need to save as .h5, also install tables: `uv pip install tables`
    The .task model file is downloaded on the first run.

    Usage:
    Mediapipe_to_pose_or_trc -i input_file --display --save_images --save_video --to_csv --to_h5 --to_json --to_trc --model_complexity 2 -o output_folder
    OR Mediapipe_to_pose_or_trc -i 0 -dTv (on webcam, display, save video, save trc)
    OR from Pose2Sim.Utilities.Mediapipe_to_pose_or_trc import mediapipe_to_pose_or_trc_func; mediapipe_to_pose_or_trc_func(input='input.mp4', display=True, save_images=True, save_video=True, to_csv=True, to_h5=True, to_json=True, to_trc=True, model_complexity=2, output_folder='output')`
    '''

    # Retrieve arguments
    raw_input = args.get('input')
    is_webcam = isinstance(raw_input, int) or str(raw_input).isdigit()
    if is_webcam:
        video_source = int(raw_input)
        video_dir = Path.cwd()
        video_name = f'webcam{video_source}_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        video_input = Path(raw_input).resolve()
        video_source = str(video_input)
        video_dir = Path(video_input).parent
        video_name = Path(video_input).stem
    output_folder = args.get('output_folder')

    display = args.get('display')
    if is_webcam and not display:
        print('Webcam input: --display must be enabled.')
        display = True
    save_images = args.get('save_images')
    save_video = args.get('save_video')

    to_csv = args.get('to_csv')
    to_h5 = args.get('to_h5')
    to_json = args.get('to_json')
    to_trc = args.get('to_trc')

    model_complexity = int(args.get('model_complexity', 2))
    model_path = args.get('model_path')

    if to_h5:
        try:
            import tables
        except ImportError:
            raise ImportError("Saving to .h5 requires the 'tables' package. Please install it using 'uv pip install tables'.")

    if to_csv or to_h5 or to_json or to_trc or save_images or save_video:
        if output_folder is None:
            output_folder = video_dir
        if not Path(output_folder).exists():
            os.mkdir(Path(output_folder).resolve())
    
    # Download model if not present
    model_name = 'pose_landmarker_' + ['lite', 'full', 'heavy'][model_complexity] + '.task'
    print(f'\nUsing model: {model_name}')
    if model_path is None:
        model_root = Path(os.getenv('TORCH_HOME', Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser() / 'mediapipe'))
        model_path = model_root / model_name
    model_path = download_model(model_path, complexity=model_complexity)

    # Configure PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1
    )

    # Run Mediapipe BlazePose
    print(f'Running Mediapipe BlazePose on: {video_source}')
    if is_webcam:
        cap, W, H, fps = setup_webcam(video_source)
        nb_frames = 0
        print('Webcam input: the framerate may vary, it will be resampled to its average.')
    else:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f'Could not open input {video_source}')
        W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    frame_processing_times = []
    count = 0
    kpt_list = []
    kpt_3d_list = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        with tqdm(total=nb_frames) as pbar:
            while cap.isOpened():
                start_time = datetime.now()
                ret, frame = cap.read()
                if ret == True:  
                    # Convert BGR to sRGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    # Mediapipe BlazePose detection
                    frame_timestamp_ms = int(count * 1000 / fps) if fps > 0 else count * 33
                    results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

                    try:
                        if results.pose_landmarks and len(results.pose_landmarks) > 0:
                            keypoints = results.pose_landmarks[0]
                            kpt = [[lm.x * W, lm.y * H, lm.visibility] for lm in keypoints]
                            kpt = [item for sublist in kpt for item in sublist]
                            draw_keypoints_on_frame(frame, keypoints, W, H)

                            keypoints3d = results.pose_world_landmarks[0]
                            kpt_3d = [[wlm.x, wlm.y, wlm.z] for wlm in keypoints3d]
                            kpt_3d = [item for sublist in kpt_3d for item in sublist]
                        else:
                            raise ValueError("No pose keypoints")
                    except Exception:
                        print(f'No person detected by Mediapipe BlazePose on frame {count}')
                        kpt = [np.nan] * 3 * len(vision.PoseLandmark)
                        kpt_3d = [np.nan] * 3 * len(vision.PoseLandmark)

                    # Display images
                    if display: 
                        img = frame.copy()
                        cv2.putText(img, "Press 'q' to stop", (int(W)-int(600*0.3), int(H)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.3+0.2, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow(video_name, img)
                        key = cv2.waitKey(1 if is_webcam else 30) & 0xFF
                        window_closed = cv2.getWindowProperty(video_name, cv2.WND_PROP_VISIBLE) < 1
                        if key == ord('q') or key == 27 or window_closed:
                            break

                    # Save images
                    if save_images: 
                        images_folder = Path(output_folder) / ('mediapipe_'+video_name + '_img')
                        if not Path(images_folder).exists():
                            os.mkdir(images_folder)
                        img_path = Path(images_folder) / ('mediapipe_'+video_name+'.'+str(count).zfill(5)+'.png')
                        cv2.imwrite(str(img_path), frame)

                    # Save video
                    if save_video:
                        if count == 0:
                            video_path = Path(output_folder) / (video_name+'_mediapipe.mp4')
                            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (int(W), int(H)))
                        writer.write(frame)

                    # Store coordinates
                    if to_csv or to_h5 or to_json or to_trc:
                        kpt_list.append(kpt)
                    if to_trc:
                        kpt_3d_list.append(kpt_3d)

                    count += 1
                    if is_webcam:
                        frame_processing_times.append((datetime.now() - start_time).total_seconds())

                else:
                    break

                pbar.update(1)

            cap.release()
            if save_video:
                print(f'Saving video to {video_path}')
                writer.release()
            cv2.destroyAllWindows()

    if is_webcam and len(frame_processing_times) > 0:
        actual_framerate = round(len(frame_processing_times) / sum(frame_processing_times))
        if save_video:
            print(f"Rewriting webcam video based on the average framerate {actual_framerate}.")
            resample_video(video_path, desired_framerate=actual_framerate, fps=fps)
        fps = actual_framerate

    # Save coordinates
    print(f'Input successfully processed.\n')
    if to_csv or to_h5:
        print(f'Saving coordinates to csv/h5 in {output_folder}')
        save_to_csv_or_h5(kpt_list, output_folder, video_name, to_csv, to_h5)
    if to_json:
        print(f'Saving coordinates to json in {output_folder}')
        save_to_json(kpt_list, output_folder, video_name)
    if to_trc:
        print(f'Saving 3D metric coordinates to trc in {output_folder}')
        save_to_trc(kpt_3d_list, output_folder, video_name, fps, tree=tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input video file, or webcam id (e.g. 0)')
    parser.add_argument('-d', '--display', action='store_true', help='display video with keypoints overlayed')
    parser.add_argument('-s', '--save_images', action='store_true', help='save images with keypoints overlayed')
    parser.add_argument('-v', '--save_video', action='store_true', help='save video with keypoints overlayed')
    parser.add_argument('-C', '--to_csv', action='store_true', help='save coordinates to csv file')
    parser.add_argument('-H', '--to_h5', action='store_true', help='save coordinates to h5 file')
    parser.add_argument('-J', '--to_json', action='store_true', help='save coordinates to json files')
    parser.add_argument('-T', '--to_trc', action='store_true', help='save 3D coordinates to trc file')
    parser.add_argument('--model_complexity', type=int, default=2, help='model complexity (0, 1 or 2)')
    parser.add_argument('--model_path', type=str, default=None, help='path to pose_landmarker .task model')
    parser.add_argument('-o', '--output_folder', required=False, help='output folder')
    args = vars(parser.parse_args())

    mediapipe_to_pose_or_trc_func(**args)


if __name__ == '__main__':
    main()
