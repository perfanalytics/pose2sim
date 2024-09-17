#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SYNCHRONIZE CAMERAS                 ##
#########################################

    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walkswith random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
'''


## INIT
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import json
import os
import glob
import fnmatch
import re
import shutil
from anytree import RenderTree
from anytree.importer import DictImporter
import logging

from Pose2Sim.common import sort_stringlist_by_last_number
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def convert_json2pandas(json_files, likelihood_threshold=0.6):
    '''
    Convert a list of JSON files to a pandas DataFrame.

    INPUTS:
    - json_files: list of str. Paths of the the JSON files.
    - likelihood_threshold: float. Drop values if confidence is below likelihood_threshold.
    - frame_range: select files within frame_range.

    OUTPUTS:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    nb_coord = 25 # int(len(json_data)/3)
    json_coords = []
    for j_p in json_files:
        with open(j_p) as j_f:
            try:
                json_data = json.load(j_f)['people']
                max_confidence_person = max(json_data, key=lambda p: np.mean(p['pose_keypoints_2d'][2::3]))
                json_data = max_confidence_person['pose_keypoints_2d']
                # remove points with low confidence
                json_data = np.array([[json_data[3*i],json_data[3*i+1],json_data[3*i+2]] if json_data[3*i+2]>likelihood_threshold else [0.,0.,0.] for i in range(nb_coord)]).ravel().tolist()
            except:
                # print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                json_data = [np.nan] * 25*3
        json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)

    return df_json_coords


def drop_col(df, col_nb):
    '''
    Drops every nth column from a DataFrame.

    INPUTS:
    - df: dataframe. The DataFrame from which columns will be dropped.
    - col_nb: int. The column number to drop.

    OUTPUTS:
    - dataframe: DataFrame with dropped columns.
    '''

    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped


def vert_speed(df, axis='y'):
    '''
    Calculate the vertical speed of a DataFrame along a specified axis.

    INPUTS:
    - df: dataframe. DataFrame of 2D coordinates.
    - axis: str. The axis along which to calculate speed. 'x', 'y', or 'z', default is 'y'.

    OUTPUTS:
    - df_vert_speed: DataFrame of vertical speed values.
    '''

    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1] / 2))]).T # modified ( df_diff.shape[1]*2 to df_diff.shape[1] / 2 )
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed


def interpolate_zeros_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUTS:
    - col_interp: interpolated pandas column
    '''
    
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    try: 
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
        col_interp = np.where(mask, col, f_interp(col.index))
        return col_interp 
    except:
        # print('No good values to interpolate')
        return col


def time_lagged_cross_corr(camx, camy, lag_range, show=True, ref_cam_id=0, cam_id=1):
    '''
    Compute the time-lagged cross-correlation between two pandas series.

    INPUTS:
    - camx: pandas series. Coordinates of reference camera.
    - camy: pandas series. Coordinates of camera to compare.
    - lag_range: int or list. Range of frames for which to compute cross-correlation.
    - show: bool. If True, display the cross-correlation plot.
    - ref_cam_id: int. The reference camera id.
    - cam_id: int. The camera id to compare.

    OUTPUTS:
    - offset: int. The time offset for which the correlation is highest.
    - max_corr: float. The maximum correlation value.
    '''

    if isinstance(lag_range, int):
        lag_range = [-lag_range, lag_range]

    pearson_r = [camx.corr(camy.shift(lag)) for lag in range(lag_range[0], lag_range[1])]
    offset = int(np.floor(len(pearson_r)/2)-np.argmax(pearson_r))
    if not np.isnan(pearson_r).all():
        max_corr = np.nanmax(pearson_r)

        if show:
            f, ax = plt.subplots(2,1)
            # speed
            camx.plot(ax=ax[0], label = f'Reference: camera #{ref_cam_id}')
            camy.plot(ax=ax[0], label = f'Compared: camera #{cam_id}')
            ax[0].set(xlabel='Frame', ylabel='Speed (px/frame)')
            ax[0].legend()
            # time lagged cross-correlation
            ax[1].plot(list(range(lag_range[0], lag_range[1])), pearson_r)
            ax[1].axvline(np.ceil(len(pearson_r)/2) + lag_range[0],color='k',linestyle='--')
            ax[1].axvline(np.argmax(pearson_r) + lag_range[0],color='r',linestyle='--',label='Peak synchrony')
            plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
            ax[1].set(title=f'Offset = {offset} frames', xlabel='Offset (frames)',ylabel='Pearson r')
            
            plt.legend()
            f.tight_layout()
            plt.show()
    else:
        max_corr = 0
        offset = 0
        if show:
            # print('No good values to interpolate')
            pass

    return offset, max_corr


def synchronize_cams_all(config_dict):
    '''
    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walkswith random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
    '''
    
    # Get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    pose_dir = os.path.realpath(os.path.join(project_dir, 'pose'))
    pose_model = config_dict.get('pose').get('pose_model')
    multi_person = config_dict.get('project').get('multi_person')
    fps =  config_dict.get('project').get('frame_rate')
    frame_range = config_dict.get('project').get('frame_range')
    display_sync_plots = config_dict.get('synchronization').get('display_sync_plots')
    keypoints_to_consider = config_dict.get('synchronization').get('keypoints_to_consider')
    approx_time_maxspeed = config_dict.get('synchronization').get('approx_time_maxspeed') 
    time_range_around_maxspeed = config_dict.get('synchronization').get('time_range_around_maxspeed')

    likelihood_threshold = config_dict.get('synchronization').get('likelihood_threshold')
    filter_cutoff = int(config_dict.get('synchronization').get('filter_cutoff'))
    filter_order = int(config_dict.get('synchronization').get('filter_order'))

    # Determine frame rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    if fps == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        except:
            fps = 60  
    lag_range = time_range_around_maxspeed*fps # frames


    # Warning if multi_person
    if multi_person:
        logging.warning('\nYou set your project as a multi-person one: make sure you set `approx_time_maxspeed` and `time_range_around_maxspeed` at times where one single person is in the scene, or you may get inaccurate results.')
        do_synchro = input('Do you want to continue? (y/n)')
        if do_synchro.lower() not in ["y","yes"]:
            logging.warning('Synchronization cancelled.')
            return
        else:
            logging.warning('Synchronization will be attempted.\n')

    # Retrieve keypoints from model
    try: # from skeletons.py
        model = eval(pose_model)
    except:
        try: # from Config.toml
            model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('Model not found in skeletons.py nor in Config.toml')
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]

    # List json files
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names] # list of json directories in pose_dir
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
    json_files_names = [sort_stringlist_by_last_number(j) for j in json_files_names]
    nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
    cam_nb = len(json_dirs)
    cam_list = list(range(cam_nb))
    
    # frame range selection
    f_range = [[0, min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    # json_files_names = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*f_range)] for json_files_cam in json_files_names]

    # Determine frames to consider for synchronization
    if isinstance(approx_time_maxspeed, list): # search around max speed
        approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]
        nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
        search_around_frames = [[int(a-lag_range) if a-lag_range>0 else 0, int(a+lag_range) if a+lag_range<nb_frames_per_cam[i] else nb_frames_per_cam[i]+f_range[0]] for i,a in enumerate(approx_frame_maxspeed)]
        logging.info(f'Synchronization is calculated around the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
    elif approx_time_maxspeed == 'auto': # search on the whole sequence (slower if long sequence)
        search_around_frames = [[f_range[0], f_range[0]+nb_frames_per_cam[i]] for i in range(cam_nb)]
        logging.info('Synchronization is calculated on the whole sequence. This may take a while.')
    else:
        raise ValueError('approx_time_maxspeed should be a list of floats or "auto"')
    
    if keypoints_to_consider == 'right':
        logging.info(f'Keypoints used to compute the best synchronization offset: right side.')
    elif keypoints_to_consider == 'left':
        logging.info(f'Keypoints used to compute the best synchronization offset: left side.')
    elif isinstance(keypoints_to_consider, list):
        logging.info(f'Keypoints used to compute the best synchronization offset: {keypoints_to_consider}.')
    elif keypoints_to_consider == 'all':
        logging.info(f'All keypoints are used to compute the best synchronization offset.')
    logging.info(f'These keypoints are filtered with a Butterworth filter (cut-off frequency: {filter_cutoff} Hz, order: {filter_order}).')
    logging.info(f'They are removed when their likelihood is below {likelihood_threshold}.\n')

    # Extract, interpolate, and filter keypoint coordinates
    logging.info('Synchronizing...')
    df_coords = []
    b, a = signal.butter(filter_order/2, filter_cutoff/(fps/2), 'low', analog = False) 
    json_files_names_range = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*frames_cam)] for (json_files_cam, frames_cam) in zip(json_files_names,search_around_frames)]
    json_files_range = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    if np.array([j==[] for j in json_files_names_range]).any():
        raise ValueError(f'No json files found within the specified frame range ({frame_range}) at the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
    
    for i in range(cam_nb):
        df_coords.append(convert_json2pandas(json_files_range[i], likelihood_threshold=likelihood_threshold))
        df_coords[i] = drop_col(df_coords[i],3) # drop likelihood
        if keypoints_to_consider == 'right':
            kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k.startswith('R') or k.startswith('right')]
            kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
            df_coords[i] = df_coords[i][kpt_indices]
        elif keypoints_to_consider == 'left':
            kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k.startswith('L') or k.startswith('left')]
            kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
            df_coords[i] = df_coords[i][kpt_indices]
        elif isinstance(keypoints_to_consider, list):
            kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k in keypoints_to_consider]
            kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
            df_coords[i] = df_coords[i][kpt_indices]
        elif keypoints_to_consider == 'all':
            pass
        else:
            raise ValueError('keypoints_to_consider should be "all", "right", "left", or a list of keypoint names.\n\
                            If you specified keypoints, make sure that they exist in your pose_model.')
        
        df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args = ['linear'])
        df_coords[i] = df_coords[i].bfill().ffill()
        df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))


    # Compute sum of speeds
    df_speed = []
    sum_speeds = []
    for i in range(cam_nb):
        df_speed.append(vert_speed(df_coords[i]))
        sum_speeds.append(abs(df_speed[i]).sum(axis=1))
        # nb_coord = df_speed[i].shape[1]
        # sum_speeds[i][ sum_speeds[i]>vmax*nb_coord ] = 0
        
        # # Replace 0 by random values, otherwise 0 padding may lead to unreliable correlations
        # sum_speeds[i].loc[sum_speeds[i] < 1] = sum_speeds[i].loc[sum_speeds[i] < 1].apply(lambda x: np.random.normal(0,1))
        
        sum_speeds[i] = pd.DataFrame(signal.filtfilt(b, a, sum_speeds[i], axis=0)).squeeze()


    # Compute offset for best synchronization:
    # Highest correlation of sum of absolute speeds for each cam compared to reference cam
    ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam)) # ref cam: least amount of frames
    ref_frame_nb = len(df_coords[ref_cam_id])
    lag_range = int(ref_frame_nb/2)
    cam_list.pop(ref_cam_id)
    offset = []
    for cam_id in cam_list:
        offset_cam_section, max_corr_cam = time_lagged_cross_corr(sum_speeds[ref_cam_id], sum_speeds[cam_id], lag_range, show=display_sync_plots, ref_cam_id=ref_cam_id, cam_id=cam_id)
        offset_cam = offset_cam_section - (search_around_frames[ref_cam_id][0] - search_around_frames[cam_id][0])
        if isinstance(approx_time_maxspeed, list):
            logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset ({offset_cam_section} on the selected section), correlation {round(max_corr_cam, 2)}.')
        else:
            logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset, correlation {round(max_corr_cam, 2)}.')
        offset.append(offset_cam)
    offset.insert(ref_cam_id, 0)

    # rename json files according to the offset and copy them to pose-sync
    sync_dir = os.path.abspath(os.path.join(pose_dir, '..', 'pose-sync'))
    os.makedirs(sync_dir, exist_ok=True)
    for d, j_dir in enumerate(json_dirs):
        os.makedirs(os.path.join(sync_dir, os.path.basename(j_dir)), exist_ok=True)
        for j_file in json_files_names[d]:
            j_split = re.split(r'(\d+)',j_file)
            j_split[-2] = f'{int(j_split[-2])-offset[d]:06d}'
            if int(j_split[-2]) > 0:
                json_offset_name = ''.join(j_split)
                shutil.copy(os.path.join(pose_dir, os.path.basename(j_dir), j_file), os.path.join(sync_dir, os.path.basename(j_dir), json_offset_name))

    logging.info(f'Synchronized json files saved in {sync_dir}.')
