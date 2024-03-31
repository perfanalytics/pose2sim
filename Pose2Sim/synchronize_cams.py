#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    #########################################
    ## SYNCHRONIZE CAMERAS                 ##
    #########################################

    Steps undergone in this script
    0. Converting json files to pandas dataframe
    1. Computing speeds (vertical)
    2. Plotting paired correlations of speeds from one camera viewpoint to another (work on one single keypoint, or on all keypoints, or on a weighted selection of keypoints)
    3. 
    Ideally, this should be done automatically for all views, checking pairs 2 by 2 with the highest correlation coefficient, 
    and ask for confirmation before deleting the frames in question (actually renamed .json.del - reset_sync option in Config.toml).
'''


## INIT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import json
import os
import fnmatch
import pickle as pk
import re

from Pose2Sim.filtering import loess_filter_1d


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.7'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def convert_json2pandas(json_dir):
    '''
    Convert JSON files in a directory to a pandas DataFrame.

    INPUTS:
    - json_dir: str. The directory path containing the JSON files.

    OUTPUT:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    json_files_names = fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json') # modified ( 'json' to '*.json' )
    json_files_names.sort(key=lambda name: int(re.search(r'(\d+)\.json', name).group(1)))
    json_files_path = [os.path.join(json_dir, j_f) for j_f in json_files_names]
    json_coords = []
    for i, j_p in enumerate(json_files_path):
        with open(j_p) as j_f:
            try:
                json_data = json.load(j_f)['people'][0]['pose_keypoints_2d']
            except:
                print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                json_data = [0]*75
        json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)
    return df_json_coords


def drop_col(df, col_nb):
    '''
    Drops every nth column from a DataFrame.

    INPUTS:
    - df: dataframe. The DataFrame from which columns will be dropped.
    - col_nb: int. The column number to drop.

    OUTPUT:
    - dataframe: DataFrame with dropped columns.
    '''

    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped


def speed_vert(df, axis='y'):
    '''
    Calculate the vertical speed of a DataFrame along a specified axis.

    Parameters:
    - df: dataframe. DataFrame of 2D coordinates.
    - axis (str): The axis along which to calculate the speed. Default is 'y'.

    OUTPUT:
    - DataFrame: The DataFrame containing the vertical speed values.
    '''

    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1] / 2))]).T # modified ( df_diff.shape[1]*2 to df_diff.shape[1] / 2 )
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed


def speed_2D(df):
    '''
    Calculate the 2D speed of a DataFrame.

    INPUTS:
    - df: dataframe. DataFrame of 2D coordinates.

    OUTPUT:
    - DataFrame: The DataFrame containing the 2D speed values.
    '''
    
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_2Dspeed = pd.DataFrame([np.sqrt(df_diff.loc[:,2*k]*2 + df_diff.loc[:,2*k+1]*2) for k in range(int(df_diff.shape[1]*2))]).T
    return df_2Dspeed


def interpolate_zeros_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUT
    - col_interp: interpolated pandas column
    '''
    
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    try: 
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
        col_interp = np.where(mask, col, f_interp(col.index))
        return col_interp 
    except:
        print('No good values to interpolate')
        return col


def find_highest_wrist_position(df_coords, wrist_index):
    '''
    Find the frame with the highest wrist position in a list of coordinate DataFrames.
    Highest wrist position frame use for finding the fastest frame.
    
    INPUT:
    - df_coords (list): List of coordinate DataFrames.
    - wrist_index (int): The index of the wrist in the keypoint list.
    
    OUTPUT:
    - list: The index of the frame with the highest wrist position.
    '''

    start_frames = []
    min_y_coords = []
    for df in df_coords:
        # Wrist y-coordinate column index (2n where n is the keypoint index)
        # Assuming wrist_index is a list and we want to use the first element
        y_col_index = wrist_index[0] * 2 + 1
        
        # Replace 0 with NaN to avoid considering them and find the index of the lowest y-coordinate value
        min_y_coord = df.iloc[:, y_col_index].replace(0, np.nan).min()
        min_y_index = df.iloc[:, y_col_index].replace(0, np.nan).idxmin()
        if min_y_coord <= 100: # if the wrist is too high, it is likely to be an outlier
            print("The wrist is too high. Please check the data for outliers.")

        start_frames.append(min_y_index)
        min_y_coords.append(min_y_coord)

    return start_frames, min_y_coords


def find_motion_end(df_coords, wrist_index, start_frame, lowest_y, fps):
    '''
    Find the frame where hands down movement ends.
    Hands down movement is defined as the time when the wrist moves down from the highest position.

    INPUT:
    - df_coord (DataFrame): The coordinate DataFrame of the reference camera.
    - wrist_index (int): The index of the wrist in the keypoint list.
    - start_frame (int): The frame where the hands down movement starts.
    - fps (int): The frame rate of the cameras in Hz.

    OUTPUT:
    - int: The index of the frame where hands down movement ends.
    '''

    y_col_index = wrist_index * 2 + 1
    wrist_y_values = df_coords.iloc[:, y_col_index].values # wrist y-coordinates
    highest_y_value = lowest_y
    highest_y_index = start_frame

    # Find the highest y-coordinate value and its index
    for i in range(highest_y_index + 1, len(wrist_y_values)):
        if wrist_y_values[i] - highest_y_value >= 50:
            start_increase_index = i
            break
    else:
        raise ValueError("The wrist does not move down.")
    
    start = start_increase_index - start_frame
    time = (start + fps) / fps

    return time


def find_fastest_frame(df_speed_list):
    '''
    Find the frame with the highest speed in a list of speed DataFrames.
    Fastest frame should locate in after highest wrist position frame.
    
    INPUT:
    - df_speed_list (list): List of speed DataFrames.
    - df_speed (DataFrame): The speed DataFrame of the reference camera.
    - fps (int): The frame rate of the cameras in Hz.
    - lag_time (float): The time lag in seconds.

    OUTPUT:
    - int: The index of the frame with the highest speed.
    '''

    for speed_series in df_speed_list:
        max_speed = speed_series.abs().max()
        max_speed_index = speed_series.abs().idxmax()
    
    if max_speed < 10:
        print(" !!Warning!! : The maximum speed is likely to be not representative of the actual movement. Consider increasing the time parameter in Config.toml.")
    return max_speed_index, max_speed


def plot_time_lagged_cross_corr(camx, camy, ax, fps, lag_time):
    '''
    Calculate and plot the max correlation between two cameras with a time lag.
    How it works:
     1. Reference camera is camx and the other is camy. (Reference camera should record last. If not, the offset will be positive.)
     2. The initial shift alppied to camy to match camx is calculated.
     3. Additionally shift camy by max_lag frames to find the max correlation.
    
    INPUT:
    - camx: pd.Series. Speed series of the reference camera.
    - camy: pd.Series). Speed series of the other camera.
    - ax: plt.axis. Plot correlation on second axis.
    - fps: int. Framerate of the cameras in Hz.
    - lag_time: float. Time lag in seconds.

    OUTPUT:
    - offset: int. Offset value to apply to synchronize the cameras.
    - max_corr: float. Maximum correlation value.
    '''

    max_lag = int(fps * lag_time)
    pearson_r = []
    lags = range(-max_lag, 1)
    
    for lag in lags:
        if lag < 0:
            shifted_camy = camy.shift(lag).dropna() # shift the camy segment by lag
            corr = camx.corr(shifted_camy) # calculate the correlation between the camx segment and the shifted camy segment
        elif lag == 0:
            corr = camx.corr(camy)
        else:
            continue 
        pearson_r.append(corr)


   # Handle NaN values in pearson_r and find the max correlation ignoring NaNs
    pearson_r = np.array(pearson_r)
    max_corr = np.nanmax(pearson_r)  # Use nanmax to ignore NaNs
    offset = np.nanargmax(pearson_r) - max_lag  # Use nanargmax to find the index of the max correlation ignoring NaNs
    # real_offset = offset + initial_shift
    
    # visualize
    ax.plot(lags, pearson_r)
    ax.axvline(offset, color='r', linestyle='--', label='Peak synchrony')
    plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
    # ax.set(title=f'Offset = {offset}{initial_shift} = {real_offset} frames', xlabel='Offset (frames)', ylabel='Pearson r')
    ax.set(title=f'Offset = {offset} frames', xlabel='Offset (frames)', ylabel='Pearson r')
    plt.legend()
    
    return offset, max_corr


def apply_offset(offset, json_dirs, reset_sync, cam1_nb, cam2_nb):
    '''
    Apply the offset to synchronize the cameras.
    Offset is always applied to the second camera.
    Offset would be always negative if the first camera is the last to start recording.
    Delete the camy json files from initial frame to offset frame.

    INPUT:
    - offset (int): The offset value to apply to synchronize the cameras.
    - json_dirs (list): List of directories containing the JSON files for each camera.
    - reset_sync (bool): Whether to reset the synchronization by deleting the .del files.
    - cam1_nb (int): The number of the reference camera.
    - cam2_nb (int): The number of the other camera.
    ''' 

    if offset == 0:
        print(f"Cams {cam1_nb} and {cam2_nb} are already synchronized. No offset applied.")
        json_dir_to_offset = json_dirs[cam2_nb]
    elif offset > 0 and not reset_sync:
        print(f"Consider adjusting the lag time.")
        raise ValueError(f"Are you sure the reference camera is the last to start recording?")
    else:
        offset = abs(offset)
        json_dir_to_offset = json_dirs[cam2_nb]

    json_files = sorted(fnmatch.filter(os.listdir(json_dir_to_offset), '*.json'), key=lambda x: int(re.findall('\d+', x)[0]))

    if reset_sync:
        del_files = fnmatch.filter(os.listdir(json_dir_to_offset), '*.del')
        for del_file in del_files:
            os.rename(os.path.join(json_dir_to_offset, del_file), os.path.join(json_dir_to_offset, del_file[:-4]))
    else:
            for i in range(offset):
                os.rename(os.path.join(json_dir_to_offset, json_files[i]), os.path.join(json_dir_to_offset, json_files[i] + '.del'))


def synchronize_cams_all(config_dict):
    '''
    
    '''
    
    # get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    pose_dir = os.path.realpath(os.path.join(project_dir, 'pose'))
    fps =  config_dict.get('project').get('frame_rate') # frame rate of the cameras (Hz)
    reset_sync = config_dict.get('synchronization').get('reset_sync')  # Start synchronization over each time it is run
    id_kpt = 4
    weights_kpt = 1
    filter_order = 4
    filter_cutoff = 6
    vmax = 20 # px/s

    # List json files
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    pose_listdirs_names.sort(key=lambda name: int(re.search(r'(\d+)', name).group(1)))
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names] # list of json directories in pose_dir
    cam_nb = len(json_dirs)

    # Extract, interpolate, and filter keypoint coordinates
    df_coords = []
    b, a = signal.butter(filter_order/2, filter_cutoff/(fps/2), 'low', analog = False) 
    for i, json_dir in enumerate(json_dirs):
        df_coords.append(convert_json2pandas(json_dir))
        df_coords[i] = drop_col(df_coords[i],3) # drop likelihood
        df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args = ['cubic'])
        df_coords[i] = df_coords[i].apply(loess_filter_1d, axis=0, args = [30])
        df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))




    # Save keypoint coordinates to pickle
    with open(os.path.join(pose_dir, 'coords'), 'wb') as fp:
        pk.dump(df_coords, fp)
    # with open(os.path.join(pose_dir, 'coords'), 'rb') as fp:
    #     df_coords = pk.load(fp)

    # Compute vertical speed
    df_speed = []
    for i in range(cam_nb):
        df_speed.append(speed_vert(df_coords[i]))
        # df_speed[i] = df_speed[i].where(abs(df_speed[i])<vmax, other=np.nan) # replaces by nan if jumps in speed
        # df_speed[i] = df_speed[i].apply(interpolate_nans, axis=0, args = ['cubic'])


    # Frame with maximum of the sum of absolute speeds
    max_speed_frame = []
    for i in range(cam_nb):
        max_speed_frame += [np.argmax(abs(df_speed[i].sum(axis=1)))]

    #############################################
    # 2. PLOTTING PAIRED CORRELATIONS OF SPEEDS #
    #############################################

    # Do this on all cam pairs
    # Choose pair with highest correlation

    # on a particular point (typically the wrist on a vertical movement)
    # or on a selection of weighted points

    # find the lowest position of the wrist
    lowest_frames, lowest_y_coords = find_highest_wrist_position(df_coords, id_kpt)

    # set reference camera
    nb_frames_per_cam = [len(d) for d in df_speed]
    ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam))
    
    
    max_speeds = []

    
    cam_list = list(range(cam_nb))
    cam_list.pop(ref_cam_id)
    for cam_id in cam_list:
        # find the highest wrist position for each camera
        camx_start_frame = lowest_frames[ref_cam_id]
        camy_start_frame = lowest_frames[cam_id]

        camx_lowest_y = lowest_y_coords[ref_cam_id]
        camy_lowest_y = lowest_y_coords[cam_id]

        camx_time = find_motion_end(df_coords[ref_cam_id], id_kpt[0], camx_start_frame, camx_lowest_y, fps)
        camy_time = find_motion_end(df_coords[cam_id], id_kpt[0], camy_start_frame, camy_lowest_y, fps)

        camx_end_frame = camx_start_frame + int(camx_time * fps)
        camy_end_frame = camy_start_frame + int(camy_time * fps)

        camx_segment = df_speed[ref_cam_id].iloc[camx_start_frame:camx_end_frame+1, id_kpt[0]]
        camy_segment = df_speed[cam_id].iloc[camy_start_frame:camy_end_frame+1, id_kpt[0]]

        # Find the fastest speed and the frame
        camx_max_speed_index, camx_max_speed = find_fastest_frame([camx_segment])
        camy_max_speed_index, camy_max_speed = find_fastest_frame([camy_segment])
        max_speeds.append(camx_max_speed)
        max_speeds.append(camy_max_speed)
        vmax = max(max_speeds)

        # Find automatically the best lag time
        lag_time = round((camy_max_speed_index - camx_max_speed_index) / fps + 1)

        # FInd the fatest frame
        camx_start_frame = camx_max_speed_index - (fps) * (lag_time)
        if camx_start_frame < 0:
            camx_start_frame = 0
        else:
            camx_start_frame = int(camx_start_frame)
        camy_start_frame = camy_max_speed_index - (fps) * (lag_time)
        camx_end_frame = camx_max_speed_index + (fps) * (lag_time)
        camy_end_frame = camy_max_speed_index + (fps) * (lag_time)

        if len(id_kpt) == 1 and id_kpt[0] != 'all':
            camx = df_speed[ref_cam_id].iloc[camx_start_frame:camx_end_frame+1, id_kpt[0]]
            camy = df_speed[cam_id].iloc[camy_start_frame:camy_end_frame+1, id_kpt[0]]
        elif id_kpt == ['all']:
            camx = df_speed[ref_cam_id].iloc[camx_start_frame:camx_end_frame+1].sum(axis=1)
            camy = df_speed[cam_id].iloc[camy_start_frame:camy_end_frame+1].sum(axis=1)
        elif len(id_kpt) == 1 and len(id_kpt) == len(weights_kpt):
            dict_id_weights = {i:w for i, w in zip(id_kpt, weights_kpt)}
            camx = df_speed[ref_cam_id] @ pd.Series(dict_id_weights).reindex(df_speed[ref_cam_id].columns, fill_value=0)
            camy = df_speed[cam_id] @ pd.Series(dict_id_weights).reindex(df_speed[cam_id].columns, fill_value=0)
            camx = camx.iloc[camx_start_frame:camx_end_frame+1]
            camy = camy.iloc[camy_start_frame:camy_end_frame+1]
        else:
            raise ValueError('wrong values for id_kpt or weights_kpt')        
        
        # filter the speeds
        camx = camx.where(lambda x: (x <= vmax) & (x >= -vmax), other=np.nan)
        camy = camy.where(lambda x: (x <= vmax) & (x >= -vmax), other=np.nan)

        f, ax = plt.subplots(2,1)

        # speed
        camx.plot(ax=ax[0], label = f'cam {ref_cam_id+1}')
        camy.plot(ax=ax[0], label = f'cam {cam_id+1}')
        ax[0].set(xlabel='Frame',ylabel='Speed (pxframe)')
        ax[0].legend()
  
        # time lagged cross-correlation
        offset, max_corr = plot_time_lagged_cross_corr(camx, camy, ax[1], fps, lag_time, camx_max_speed_index, camy_max_speed_index)
        f.tight_layout()
        plt.show()
        print(f'Using number{id_kpt} keypoint, synchronized camera {ref_cam_id+1} and camera {cam_id+1}, with an offset of {offset} and a max correlation of {max_corr}.')

        # apply offset
        apply_offset(offset, json_dirs, reset_sync, ref_cam_id, cam_id)


