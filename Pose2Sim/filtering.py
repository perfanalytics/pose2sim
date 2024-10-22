#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## FILTER 3D COORDINATES                                                 ##
###########################################################################

Filter trc 3D coordinates.

Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median, Kalman
Set your parameters in Config.toml
    
INPUTS: 
- a trc file
- filtering parameters in Config.toml

OUTPUT: 
- a filtered trc file
'''


## INIT
import os
import glob
import fnmatch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import logging

from scipy import signal
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from filterpy.kalman import KalmanFilter, rts_smoother
from filterpy.common import Q_discrete_white_noise

from Pose2Sim.common import plotWindow
from Pose2Sim.common import convert_to_c3d

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def kalman_filter(coords, frame_rate, measurement_noise, process_noise, nb_dimensions=3, nb_derivatives=3, smooth=True):
    '''
    Filters coordinates with a Kalman filter or a Kalman smoother
    
    INPUTS:
    - coords: array of shape (nframes, ndims)
    - frame_rate: integer
    - measurement_noise: integer
    - process_noise: integer
    - nb_dimensions: integer, number of dimensions (3 if 3D coordinates)
    - nb_derivatives: integer, number of derivatives (3 if constant acceleration model)
    - smooth: boolean. True if souble pass (recommended), False if single pass (if real-time)
    
    OUTPUTS:
    - kpt_coords_filt: filtered coords
    '''

    # Variables
    dim_x = nb_dimensions * nb_derivatives # 9 state variables 
    dt = 1/frame_rate
    
    # Filter definition
    f = KalmanFilter(dim_x=dim_x, dim_z=nb_dimensions)

    # States: initial position, velocity, accel, in 3D
    def derivate_array(arr, dt=1):
        return np.diff(arr, axis=0)/dt
    def repeat(func, arg_func, nb_reps):
        for i in range(nb_reps):
            arg_func = func(arg_func)
        return arg_func
    x_init = []
    for n_der in range(nb_derivatives):
        x_init += [repeat(derivate_array, coords, n_der)[0]] # pose*3D, vel*3D, accel*3D
    f.x = np.array(x_init).reshape(nb_dimensions,nb_derivatives).T.flatten() # pose, vel, accel *3D
    
    # State transition matrix
    F_per_coord = np.zeros((int(dim_x/nb_dimensions), int(dim_x/nb_dimensions)))
    for i in range(nb_derivatives):
        for j in range(min(i+1, nb_derivatives)):
            F_per_coord[j,i] = dt**(i-j) / np.math.factorial(i - j)
    f.F = np.kron(np.eye(nb_dimensions),F_per_coord) 
    # F_per_coord= [[1, dt, dt**2/2], 
                 # [ 0, 1,  dt     ],
                 # [ 0, 0,  1      ]])

    # No control input
    f.B = None 

    # Measurement matrix (only positions)
    H = np.zeros((nb_dimensions, dim_x)) 
    for i in range(min(nb_dimensions,dim_x)):
        H[i, int(i*(dim_x/nb_dimensions))] = 1
    f.H = H
    # H = [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 1., 0., 0., 0., 0., 0.],
        # [0., 0., 0., 0., 0., 0., 1., 0., 0.]]

    # Covariance matrix
    f.P *= measurement_noise 

    # Measurement noise
    f.R = np.diag([measurement_noise**2]*nb_dimensions) 

    # Process noise
    f.Q = Q_discrete_white_noise(nb_derivatives, dt=dt, var=process_noise**2, block_size=nb_dimensions) 

    # Run filter: predict and update for each frame
    mu, cov, _, _ = f.batch_filter(coords) # equivalent to below
    # mu = []
    # for kpt_coord_frame in coords:
        # f.predict()
        # f.update(kpt_coord_frame)
        # mu.append(f.x.copy())
    ind_of_position = [int(d*(dim_x/nb_dimensions)) for d in range(nb_dimensions)]
    coords_filt = np.array(mu)[:,ind_of_position]

    # RTS smoother
    if smooth == True:
        mu2, P, C, _ = f.rts_smoother(mu, cov)
        coords_filt = np.array(mu2)[:,ind_of_position]

    return coords_filt


def kalman_filter_1d(config_dict, frame_rate, col):
    '''
    1D Kalman filter
    Deals with nans
    
    INPUT:
    - col: Pandas dataframe column
    - trustratio: int, ratio process_noise/measurement_noise
    - frame_rate: int
    - smooth: boolean, True if double pass (recommended), False if single pass (if real-time)

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    trustratio = int(config_dict.get('filtering').get('kalman').get('trust_ratio'))
    smooth = int(config_dict.get('filtering').get('kalman').get('smooth'))
    measurement_noise = 20
    process_noise = measurement_noise * trustratio

    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)  | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    if idx_sequences[0].size > 0:
        idx_sequences_to_filter = [seq for seq in idx_sequences]
    
        # Filter each of the selected sequences
        for seq_f in idx_sequences_to_filter:
            col_filtered[seq_f] = kalman_filter(col_filtered[seq_f], frame_rate, measurement_noise, process_noise, nb_dimensions=1, nb_derivatives=3, smooth=smooth).flatten()

    return col_filtered


def butterworth_filter_1d(config_dict, frame_rate, col):
    '''
    1D Zero-phase Butterworth filter (dual pass)
    Deals with nans

    INPUT:
    - col: numpy array
    - order: int
    - cutoff: int
    - frame_rate: int

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    type = 'low' #config_dict.get('filtering').get('butterworth').get('type')
    order = int(config_dict.get('filtering').get('butterworth').get('order'))
    cutoff = int(config_dict.get('filtering').get('butterworth').get('cut_off_frequency'))    

    b, a = signal.butter(order/2, cutoff/(frame_rate/2), type, analog = False) 
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
    

def butterworth_on_speed_filter_1d(config_dict, frame_rate, col):
    '''
    1D zero-phase Butterworth filter (dual pass) on derivative

    INPUT:
    - col: Pandas dataframe column
    - frame rate, order, cut-off frequency, type (from Config.toml)

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    type = 'low' # config_dict.get('filtering').get('butterworth_on_speed').get('type')
    order = int(config_dict.get('filtering').get('butterworth_on_speed').get('order'))
    cutoff = int(config_dict.get('filtering').get('butterworth_on_speed').get('cut_off_frequency'))

    b, a = signal.butter(order/2, cutoff/(frame_rate/2), type, analog = False)
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


def gaussian_filter_1d(config_dict, frame_rate, col):
    '''
    1D Gaussian filter

    INPUT:
    - col: Pandas dataframe column
    - gaussian_filter_sigma_kernel: kernel size from Config.toml

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    gaussian_filter_sigma_kernel = int(config_dict.get('filtering').get('gaussian').get('sigma_kernel'))

    col_filtered = gaussian_filter1d(col, gaussian_filter_sigma_kernel)

    return col_filtered
    

def loess_filter_1d(config_dict, frame_rate, col):
    '''
    1D LOWESS filter (Locally Weighted Scatterplot Smoothing)

    INPUT:
    - col: Pandas dataframe column
    - loess_filter_nb_values: window used for smoothing from Config.toml
    frac = loess_filter_nb_values * frames_number

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    kernel = config_dict.get('filtering').get('LOESS').get('nb_values_used')

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
    

def median_filter_1d(config_dict, frame_rate, col):
    '''
    1D median filter

    INPUT:
    - col: Pandas dataframe column
    - median_filter_kernel_size: kernel size from Config.toml
    
    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''
    
    median_filter_kernel_size = config_dict.get('filtering').get('median').get('kernel_size')
    
    col_filtered = signal.medfilt(col, kernel_size=median_filter_kernel_size)

    return col_filtered


def display_figures_fun(Q_unfilt, Q_filt, time_col, keypoints_names, person_id=0):
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
    pw.MainWindow.setWindowTitle('Person '+ str(person_id) + ' coordinates')
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


def filter1d(col, config_dict, filter_type, frame_rate):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - filter_type: filter type from Config.toml
    - frame_rate: int
    
    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    # Choose filter
    filter_mapping = {
        'kalman': kalman_filter_1d,
        'butterworth': butterworth_filter_1d, 
        'butterworth_on_speed': butterworth_on_speed_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'LOESS': loess_filter_1d, 
        'median': median_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(config_dict, frame_rate, col)

    return col_filtered


def recap_filter3d(config_dict, trc_path):
    '''
    Print a log message giving filtering parameters. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''

    # Read Config
    filter_type = config_dict.get('filtering').get('type')
    kalman_filter_trustratio = int(config_dict.get('filtering').get('kalman').get('trust_ratio'))
    kalman_filter_smooth = int(config_dict.get('filtering').get('kalman').get('smooth'))
    kalman_filter_smooth_str = 'smoother' if kalman_filter_smooth else 'filter'
    butterworth_filter_type = 'low' # config_dict.get('filtering').get('butterworth').get('type')
    butterworth_filter_order = int(config_dict.get('filtering').get('butterworth').get('order'))
    butterworth_filter_cutoff = int(config_dict.get('filtering').get('butterworth').get('cut_off_frequency'))
    butter_speed_filter_type = 'low' # config_dict.get('filtering').get('butterworth_on_speed').get('type')
    butter_speed_filter_order = int(config_dict.get('filtering').get('butterworth_on_speed').get('order'))
    butter_speed_filter_cutoff = int(config_dict.get('filtering').get('butterworth_on_speed').get('cut_off_frequency'))
    gaussian_filter_sigma_kernel = int(config_dict.get('filtering').get('gaussian').get('sigma_kernel'))
    loess_filter_nb_values = config_dict.get('filtering').get('LOESS').get('nb_values_used')
    median_filter_kernel_size = config_dict.get('filtering').get('median').get('kernel_size')
    make_c3d = config_dict.get('filtering').get('make_c3d')
    
    # Recap
    filter_mapping_recap = {
        'kalman': f'--> Filter type: Kalman {kalman_filter_smooth_str}. Measurements trusted {kalman_filter_trustratio} times as much as previous data, assuming a constant acceleration process.', 
        'butterworth': f'--> Filter type: Butterworth {butterworth_filter_type}-pass. Order {butterworth_filter_order}, Cut-off frequency {butterworth_filter_cutoff} Hz.', 
        'butterworth_on_speed': f'--> Filter type: Butterworth on speed {butter_speed_filter_type}-pass. Order {butter_speed_filter_order}, Cut-off frequency {butter_speed_filter_cutoff} Hz.', 
        'gaussian': f'--> Filter type: Gaussian. Standard deviation kernel: {gaussian_filter_sigma_kernel}', 
        'LOESS': f'--> Filter type: LOESS. Number of values used: {loess_filter_nb_values}', 
        'median': f'--> Filter type: Median. Kernel size: {median_filter_kernel_size}'
    }
    logging.info(filter_mapping_recap[filter_type])
    logging.info(f'Filtered 3D coordinates are stored at {trc_path}.\n')
    if make_c3d:
        logging.info('All filtered trc files have been converted to c3d.')


def filter_all(config_dict):
    '''
    Filter the 3D coordinates of the trc file.
    Displays filtered coordinates for checking.

    INPUTS:
    - a trc file
    - filtration parameters from Config.toml

    OUTPUT:
    - a filtered trc file
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    pose3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    display_figures = config_dict.get('filtering').get('display_figures')
    filter_type = config_dict.get('filtering').get('type')
    make_c3d = config_dict.get('filtering').get('make_c3d')

    # Get frame_rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        except:
            frame_rate = 60
    
    # Trc paths
    trc_path_in = [file for file in glob.glob(os.path.join(pose3d_dir, '*.trc')) if 'filt' not in file]
    trc_f_out = [f'{os.path.basename(t).split(".")[0]}_filt_{filter_type}.trc' for t in trc_path_in]
    trc_path_out = [os.path.join(pose3d_dir, t) for t in trc_f_out]
    
    for person_id, (t_in, t_out) in enumerate(zip(trc_path_in, trc_path_out)):
        # Read trc header
        with open(t_in, 'r') as trc_file:
            header = [next(trc_file) for line in range(5)]

        # Read trc coordinates values
        trc_df = pd.read_csv(t_in, sep="\t", skiprows=4)
        frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
        Q_coord = trc_df.drop(trc_df.columns[[0, 1, -1]], axis=1)

        # Filter coordinates
        Q_filt = Q_coord.apply(filter1d, axis=0, args = [config_dict, filter_type, frame_rate])

        # Display figures
        if display_figures:
            # Retrieve keypoints
            keypoints_names = pd.read_csv(t_in, sep="\t", skiprows=3, nrows=0).columns[2::3][:-1].to_numpy()
            display_figures_fun(Q_coord, Q_filt, time_col, keypoints_names, person_id)

        # Reconstruct trc file with filtered coordinates
        with open(t_out, 'w') as trc_o:
            [trc_o.write(line) for line in header]
            Q_filt.insert(0, 'Frame#', frames_col)
            Q_filt.insert(1, 'Time', time_col)
            # Q_filt = Q_filt.fillna(' ')
            Q_filt.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

        # Save c3d
        if make_c3d:
            convert_to_c3d(t_out)

        # Recap
        recap_filter3d(config_dict, t_out)


