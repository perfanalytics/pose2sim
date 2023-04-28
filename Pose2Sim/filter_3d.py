#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## FILTER 3D COORDINATES                                                 ##
    ###########################################################################
    
    Filter trc 3D coordinates.

    Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median
    Set your parameters in Config.toml
        
    INPUTS: 
    - a trc file
    - filtering parameters in Config.toml
    
    OUTPUT: 
    - a filtered trc file
    
'''


## INIT
import os
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from scipy import signal
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess

from Pose2Sim.common import plotWindow

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def butterworth_filter_1d(config, col):
    '''
    1D Zero-phase Butterworth filter (dual pass)
    Deals with nans

    INPUT:
    - col: numpy array
    - order: int
    - cutoff: int
    - framerate: int

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    type = config.get('3d-filtering').get('butterworth').get('type')
    order = int(config.get('3d-filtering').get('butterworth').get('order'))
    cutoff = int(config.get('3d-filtering').get('butterworth').get('cut_off_frequency'))    
    framerate = config.get('project').get('frame_rate')

    b, a = signal.butter(order/2, cutoff/(framerate/2), 'low', analog = False) 
    padlen = 3 * max(len(a), len(b))
    
    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)  | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = signal.filtfilt(b, a, col_filtered[seq_f])
    
    return col_filtered
    

def butterworth_on_speed_filter_1d(config, col):
    '''
    1D zero-phase Butterworth filter (dual pass) on derivative

    INPUT:
    - col: Pandas dataframe column
    - frame rate, order, cut-off frequency, type (from Config.toml)

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    type = config.get('3d-filtering').get('butterworth_on_speed').get('type')
    order = int(config.get('3d-filtering').get('butterworth_on_speed').get('order'))
    cutoff = int(config.get('3d-filtering').get('butterworth_on_speed').get('cut_off_frequency'))
    framerate = config.get('project').get('frame_rate')

    b, a = signal.butter(order/2, cutoff/(framerate/2), type, analog = False)
    padlen = 3 * max(len(a), len(b))
    
    # derivative
    col_filtered = col.copy()
    col_filtered_diff = col_filtered.diff()   # derivative
    col_filtered_diff = col_filtered_diff.fillna(col_filtered_diff.iloc[1]/2) # set first value correctly instead of nan
    
    # split into sequences of not nans
    mask = np.isnan(col_filtered_diff)  | col_filtered_diff.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered_diff[seq_f] = signal.filtfilt(b, a, col_filtered_diff[seq_f])
    col_filtered = col_filtered_diff.cumsum() + col.iloc[0] # integrate filtered derivative
    
    return col_filtered


def gaussian_filter_1d(config, col):
    '''
    1D Gaussian filter

    INPUT:
    - col: Pandas dataframe column
    - gaussian_filter_sigma_kernel: kernel size from Config.toml

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    gaussian_filter_sigma_kernel = int(config.get('3d-filtering').get('gaussian').get('sigma_kernel'))

    col_filtered = gaussian_filter1d(col, gaussian_filter_sigma_kernel)

    return col_filtered
    

def loess_filter_1d(config, col):
    '''
    1D LOWESS filter (Locally Weighted Scatterplot Smoothing)

    INPUT:
    - col: Pandas dataframe column
    - loess_filter_nb_values: window used for smoothing from Config.toml
    frac = loess_filter_nb_values * frames_number

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    kernel = config.get('3d-filtering').get('LOESS').get('nb_values_used')

    col_filtered = col.copy()
    mask = np.isnan(col_filtered) 
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > kernel]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = lowess(col_filtered[seq_f], seq_f, is_sorted=True, frac=kernel/len(seq_f), it=0)[:,1]

    return col_filtered
    

def median_filter_1d(config, col):
    '''
    1D median filter

    INPUT:
    - col: Pandas dataframe column
    - median_filter_kernel_size: kernel size from Config.toml
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''
    
    median_filter_kernel_size = config.get('3d-filtering').get('median').get('kernel_size')
    
    col_filtered = signal.medfilt(col, kernel_size=median_filter_kernel_size)

    return col_filtered


def display_figures_fun(Q_unfilt, Q_filt, time_col, keypoints_names):
    '''
    Displays filtered and unfiltered data for comparison

    INPUTS:
    - Q_unfilt: pandas dataframe of unfiltered 3D coordinates
    - Q_filt: pandas dataframe of filtered 3D coordinates
    - time_col: pandas column
    - keypoints_names: list of strings

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''

    pw = plotWindow()
    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        
        axX = plt.subplot(311)
        plt.plot(time_col.to_numpy(), Q_unfilt.iloc[:,id*3].to_numpy(), label='unfiltered')
        plt.plot(time_col.to_numpy(), Q_filt.iloc[:,id*3].to_numpy(), label='filtered')
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(312)
        plt.plot(time_col.to_numpy(), Q_unfilt.iloc[:,id*3+1].to_numpy(), label='unfiltered')
        plt.plot(time_col.to_numpy(), Q_filt.iloc[:,id*3+1].to_numpy(), label='filtered')
        plt.setp(axY.get_xticklabels(), visible=False)
        axY.set_ylabel(keypoint+' Y')
        plt.legend()

        axZ = plt.subplot(313)
        plt.plot(time_col.to_numpy(), Q_unfilt.iloc[:,id*3+2].to_numpy(), label='unfiltered')
        plt.plot(time_col.to_numpy(), Q_filt.iloc[:,id*3+2].to_numpy(), label='filtered')
        axZ.set_ylabel(keypoint+' Z')
        axZ.set_xlabel('Time')
        plt.legend()

        pw.addPlot(keypoint, f)
    
    pw.show()


def filter1d(col, config, filter_type):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - filter_type: filter type from Config.toml
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    # Choose filter
    filter_mapping = {
        'butterworth': butterworth_filter_1d, 
        'butterworth_on_speed': butterworth_on_speed_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'LOESS': loess_filter_1d, 
        'median': median_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(config, col)

    return col_filtered


def recap_filter3d(config, trc_path):
    '''
    Print a log message giving filtering parameters. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''

    # Read Config
    filter_type = config.get('3d-filtering').get('type')
    butterworth_filter_type = config.get('3d-filtering').get('butterworth').get('type')
    butterworth_filter_order = int(config.get('3d-filtering').get('butterworth').get('order'))
    butterworth_filter_cutoff = int(config.get('3d-filtering').get('butterworth').get('cut_off_frequency'))
    butter_speed_filter_type = config.get('3d-filtering').get('butterworth_on_speed').get('type')
    butter_speed_filter_order = int(config.get('3d-filtering').get('butterworth_on_speed').get('order'))
    butter_speed_filter_cutoff = int(config.get('3d-filtering').get('butterworth_on_speed').get('cut_off_frequency'))
    gaussian_filter_sigma_kernel = int(config.get('3d-filtering').get('gaussian').get('sigma_kernel'))
    loess_filter_nb_values = config.get('3d-filtering').get('LOESS').get('nb_values_used')
    median_filter_kernel_size = config.get('3d-filtering').get('median').get('kernel_size')

    # Recap
    filter_mapping_recap = {
        'butterworth': f'--> Filter type: Butterworth {butterworth_filter_type}-pass. Order {butterworth_filter_order}, Cut-off frequency {butterworth_filter_cutoff} Hz.', 
        'butterworth_on_speed': f'--> Filter type: Butterworth on speed {butter_speed_filter_type}-pass. Order {butter_speed_filter_order}, Cut-off frequency {butter_speed_filter_cutoff} Hz.', 
        'gaussian': f'--> Filter type: Gaussian. Standard deviation kernel: {gaussian_filter_sigma_kernel}', 
        'LOESS': f'--> Filter type: LOESS. Number of values used: {loess_filter_nb_values}', 
        'median': f'--> Filter type: Median. Kernel size: {median_filter_kernel_size}'
    }
    logging.info(filter_mapping_recap[filter_type])
    logging.info(f'Filtered 3D coordinates are stored at {trc_path}.')


def filter_all(config):
    '''
    Filter the 3D coordinates of the trc file.
    Displays filtered coordinates for checking.

    INPUTS:
    - a trc file
    - filtration parameters from Config.toml

    OUTPUT:
    - a filtered trc file
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    pose_tracked_folder_name = config.get('project').get('poseTracked_folder_name')
    pose_folder_name = config.get('project').get('pose_folder_name')
    try:
        pose_tracked_dir = os.path.join(project_dir, pose_folder_name)
        os.path.isdir(pose_tracked_dir)
        pose_dir = pose_tracked_dir
    except:
        pose_dir = os.path.join(project_dir, pose_folder_name)
    json_folder_extension =  config.get('project').get('pose_json_folder_extension')
    frame_range = config.get('project').get('frame_range')
    seq_name = os.path.basename(project_dir)
    pose3d_folder_name = config.get('project').get('pose3d_folder_name')
    pose3d_dir = os.path.join(project_dir, pose3d_folder_name)
    display_figures = config.get('3d-filtering').get('display_figures')
    filter_type = config.get('3d-filtering').get('type')

    # Frames range
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k]
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
    f_range = [[0,min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
    
    # Trc paths
    trc_f_in = f'{seq_name}_{f_range[0]}-{f_range[1]}.trc'
    trc_f_out = f'{seq_name}_filt_{f_range[0]}-{f_range[1]}.trc'
    trc_path_in = os.path.join(pose3d_dir, trc_f_in)
    trc_path_out = os.path.join(pose3d_dir, trc_f_out)
    
    # Read trc header
    with open(trc_path_in, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]

    # Read trc coordinates values
    trc_df = pd.read_csv(trc_path_in, sep="\t", skiprows=4)
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # Filter coordinates
    Q_filt = Q_coord.apply(filter1d, axis=0, args = [config, filter_type])

    # Display figures
    if display_figures:
        # Retrieve keypoints
        keypoints_names = pd.read_csv(trc_path_in, sep="\t", skiprows=3, nrows=0).columns[2::3].to_numpy()
        display_figures_fun(Q_coord, Q_filt, time_col, keypoints_names)

    # Reconstruct trc file with filtered coordinates
    with open(trc_path_out, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_filt.insert(0, 'Frame#', frames_col)
        Q_filt.insert(1, 'Time', time_col)
        Q_filt.to_csv(trc_o, sep='\t', index=False, header=None, line_terminator='\n')

    # Recap
    recap_filter3d(config, trc_path_out)
