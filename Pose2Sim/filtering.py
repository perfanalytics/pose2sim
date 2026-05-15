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
import math
import numpy as np
np.set_printoptions(legacy='1.21') # otherwise prints np.float64(3.0) rather than 3.0
import pandas as pd
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import platform

from scipy import signal
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate._bsplines import _coeff_of_divided_diff, _compute_optimal_gcv_parameter
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve

from statsmodels.nonparametric.smoothers_lowess import lowess
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from Pose2Sim.common import plotWindow
from Pose2Sim.common import convert_to_c3d, read_trc, write_trc, read_mot, write_mot, is_video_file

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def hampel_filter(col, window_size=7, n_sigma=2):
    '''
    Hampel filter for outlier rejection before other filtering methods.
    Takes a sliding window of size 7, calculates its median and standard deviation, 
    replaces value by median if difference is more than 2 times the standard deviation (95% confidence interval), 
    else keeps the value.
    '''

    col_filtered = col.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(col) - half_window):
        window = col[i-half_window:i+half_window+1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))  # Median Absolute Deviation
        
        if mad != 0:
            modified_z_score = 0.6745 * (col[i] - median) / mad #75% percentile from median
            if np.abs(modified_z_score) > n_sigma:
                col_filtered[i] = median
    
    return col_filtered


def one_euro_filter_1d(config_dict, frame_rate, col):
    '''
    Zero-phase OneEuro filter for 1D signal with NaN handling
    
    INPUT:
    - config_dict: Dictionary containing filtering parameters
    - frame_rate: Sampling frequency in Hz
    - col: Pandas Series (single column of scalar values)
    
    OUTPUT:
    - col_filtered: Filtered pandas Series
    '''
    
    # Get parameters from config
    min_cutoff = config_dict.get('filtering', {}).get('one_euro', {}).get('cut_off_frequency', 2.5)
    beta = config_dict.get('filtering', {}).get('one_euro', {}).get('beta', 0.9)
    d_cutoff = config_dict.get('filtering', {}).get('one_euro', {}).get('d_cut_off_frequency', 1.0)
    dt = 1.0 / frame_rate
    
    def smoothing_factor(dt, cutoff):
        '''
        Equivalent to a 1st order butterworth filter
        '''
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def apply_filter(data):
        '''
        Apply OneEuro filter to 1D data
        '''

        if len(data) < 2:
            return data

        filtered = [data[0]]
        x_prev = data[0]
        dx_prev = 0.0
        for i in range(1, len(data)):
            x = data[i]
            
            # Filter derivative
            alpha_d = smoothing_factor(dt, d_cutoff)
            dx = (x-x_prev) / dt
            dx_hat = alpha_d*dx + (1-alpha_d)*dx_prev
            
            # Adaptive cutoff and filter signal
            cutoff = min_cutoff + beta*abs(dx_hat) # cutoff = min_cutoff + beta*velocity
            alpha = smoothing_factor(dt, cutoff) # equivalent to a 1st order butterworth filter
            x_hat = alpha*x + (1-alpha)*x_prev # same as 1st order Butterworth filter: more or less sensitive to previous value
            
            filtered.append(x_hat)
            x_prev = x_hat
            dx_prev = dx_hat
        
        return np.array(filtered)
    
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 2]

    # Filter each valid sequence
    for seq in idx_sequences_to_filter:
        data = col_filtered[seq].values
        
        # Forward and backward passes (for zero-phase filtering)
        filtered_forward = apply_filter(data)
        filtered_backward = apply_filter(filtered_forward[::-1])[::-1]
        
        col_filtered[seq] = filtered_backward
    
    return col_filtered


def _compute_optimal_gcv_parameter_numstable(x, y):
    '''
    Makes x values spaced 1 apart, to make sure the optimal lambda value 
    is correctly computed.
    See https://stackoverflow.com/a/79740481/12196632
    '''
    
    x_spacing = np.diff(x)
    assert (x_spacing >= 0).all(), "x must be sorted"
    x_spacing_avg = x_spacing.mean()
    assert x_spacing_avg != 0, "div by zero"

    # x values spaced 1 apart
    new_x = x / x_spacing_avg
    X, wE, y, w = get_smoothing_spline_intermediate_arrays(new_x, y)
    
    # Rescale the value of lambda we found back to the original problem
    lam = _compute_optimal_gcv_parameter(X, wE, y, w) * x_spacing_avg ** 3
    
    return lam


def get_smoothing_spline_intermediate_arrays(x, y, w=None):
    '''
    Used by _compute_optimal_gcv_parameter_numstable to compute the optimal lambda value for a smoothing spline.
    See https://stackoverflow.com/a/79740481/12196632
    '''

    axis = 0
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)

    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')

    if x.ndim != 1 or x.shape[0] != y.shape[axis]:
        raise ValueError(f'``x`` should be 1D and {x.shape = } == {y.shape = }')

    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')

    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]

    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')

    y = np.moveaxis(y, axis, 0)
    y_shape1 = y.shape[1:]
    if y_shape1 != ():
        y = y.reshape((n, -1))

    # create design matrix in the B-spline basis
    X_bspl = BSpline.design_matrix(x, t, 3)
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2: -2] = X_bspl[i: i - 4, 3: -3][np.diag_indices(n - 4)]

    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0],
                X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2],
                 (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]

    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3])\
                   / w[j-2: j+3]
    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6
    return X, wE, y, w


def gcv_spline_filter_1d(config_dict, frame_rate, col):
    '''
    1D GCV Spline filter.
    
    If cutoff is a number, it is used as the cut-off frequency in Hz and behaves like a butterworth filter.
    If cutoff is 'auto', GCV finds the best trade-off between smoothness and fidelity to data (optimal lambda),
    and falls back to a biomechanically sensible frequency if GCV returns an unreliable result.
    Smoothing_factor biases results towards smoothing if > 1, and towards fidelity to input data if <1. Ignored if cutoff is not 'auto'.

    NOTE: In specific cases of a triangular wave + drift (eg pelvis Y coordinates of a subject walking uphill), 
    the filter may occasionally overfilter the trajectory.
    Known issue posted at: https://github.com/scipy/scipy/issues/23472

    INPUT:
    - config_dict: configuration dictionary:
        - filtering.gcv_spline.cut_off_frequency: 'auto' (default) or float (Hz)
        - filtering.gcv_spline.smoothing_factor: float >= 0, default 1.0. >1 to prioritize smoothing, <1 for fidelity to input data
    - frame_rate: float, frames per second
    - col: Pandas Series

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    cutoff = config_dict.get('filtering', {}).get('gcv_spline', {}).get('cut_off_frequency', 'auto')
    smoothing_factor = float(config_dict.get('filtering', {}).get('gcv_spline', {}).get('smoothing_factor', 1.0))

    # If the automatically computed lambda corresponds to a frequency above max_cutoff (eg short sequence),
    # it falls back to a lambda corresponding to max_cutoff. Likewise if frequency is too low (eg noisy short sequence)
    max_cutoff = 30.0
    min_cutoff = 1.0
    lam_min = (frame_rate / (2.0 * np.pi * max_cutoff)) ** 4 # lam from cutoff
    lam_max = (frame_rate / (2.0 * np.pi * min_cutoff)) ** 4

    # Split into sequences of not nans
    # print('\n', col.name)
    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 5] # Minimum length for 4th order smoothing spline
    
    # Filter each of the selected sequences
    for seq_f in idx_sequences_to_filter:
        x = np.arange(len(seq_f), dtype=float)
        y_raw = col_filtered.iloc[seq_f].to_numpy(dtype=float)

        # Normalize y_raw around 1, because zero mean leads to unstabilities
        median_y = np.median(y_raw) # median of time series
        mad_y = np.median(np.abs(y_raw - median_y)) # median absolute deviation from median
        scale = 1.4826 * mad_y if mad_y > 0 else 1.0 # 1.4826*mad_y equivalent to dividing by std (for y_norm to have a std of 1)
        y_norm = 1.0 + (y_raw - median_y) / scale

        if cutoff == 'auto':
            # Compute optimal lam 
            # See https://stackoverflow.com/a/79740481/12196632
            lam = _compute_optimal_gcv_parameter_numstable(x, y_norm)

            # More smoothing if smoothing_factor > 1, more fidelity to data if < 1
            lam *= smoothing_factor

            # Bounds if lam is not coherent (short, near-constant sequences, triangular wave+drift...)
            if lam < lam_min or lam > lam_max:
                old_lam = lam
                lam = np.clip(lam, lam_min, lam_max)
                logging.warning(f'{col.name}: Automatically computed lambda is equivalent to a cut-off frequency of {frame_rate / (2.0 * np.pi * old_lam**0.25):.2f} Hz, which is outside of the expected range [{min_cutoff}, {max_cutoff}] Hz. Falling back to a lambda value corresponding to a cut-off frequency of {frame_rate / (2.0 * np.pi * lam**0.25):.2f} Hz. Your sequence might be too short, noisy, or near-constant for cutoff=\'auto\' to work effectively.')

        else:
            # Estimate lam from cutoff frequency
            lam = (frame_rate / (2.0 * np.pi * float(cutoff))) ** 4

            # More smoothing if smoothing_factor > 1, more fidelity to data if < 1
            lam *= smoothing_factor

        spline = make_smoothing_spline(x, y_norm, w=None, lam=lam)
        y_filtered_norm = spline(x)

        # Denormalize
        col_filtered.iloc[seq_f] = (y_filtered_norm - 1.0) * scale + median_y

    return col_filtered


def acc_minimizing_filter_1d(config_dict, frame_rate, col):
    '''
    1D Whittaker-Henderson filter (acceleration-minimizing).
    Inspired by AddBiomechanics: https://github.com/keenon/AddBiomechanics/blob/main/server/engine/src/dynamics_pass/acceleration_minimizing_pass.py

    INPUT:
    - config_dict: configuration dictionary:
        - filtering.acc_minimizing.cut_off_frequency: float (Hz)
    - frame_rate: float, frames per second
    - col: Pandas Series

    OUTPUT:
    - col_filtered: filtered Pandas Series (same index as input)
    '''

    cutoff = float(config_dict.get('filtering', {}).get('acc_minimizing', {}).get('cut_off_frequency', 6.0))
    pad = 10

    # conversion from cutoff to lambda
    lam = (frame_rate / (2.0 * np.pi * cutoff)) ** 4

    # Split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 3] # Minimum length for accelerations (2nd order differences)

    # Filter each of the selected sequences
    for seq_f in idx_sequences_to_filter:
        y = col_filtered.iloc[seq_f].to_numpy(dtype=float)

        # Pad at start and end of the sequence to minimize edge effects
        effective_pad = min(pad, len(y))
        y_padded = np.concatenate([
            np.repeat(y[0],  effective_pad),
            y,
            np.repeat(y[-1], effective_pad),
        ])

        # Sparse second-difference operator: D[i] = y[i] - 2*y[i+1] + y[i+2]
        d = np.ones(len(y_padded) - 2)
        D = sparse.diags(
            [d, -2.0 * d, d],
            offsets=[0, 1, 2],
            shape=(len(y_padded) - 2, len(y_padded)),
            format='csc'
        )

        # Solve (I + lambda * D^T D) y_smooth = y_padded
        # Equivalent to minimizing ||y_smooth - y_padded||^2 + lambda * ||D y_smooth||^2
        #                            fidelity term            acceleration smoothness term
        A = sparse.eye(len(y_padded), format='csc') + lam * (D.T @ D)
        y_smooth_padded = spsolve(A, y_padded)

        # Remove padding
        col_filtered.iloc[seq_f] = y_smooth_padded[effective_pad:effective_pad + len(y)]

    return col_filtered


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
    coords = np.array(coords)
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
            F_per_coord[j,i] = dt**(i-j) / math.factorial(i - j)
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

    trustratio = int(config_dict.get('filtering', {}).get('kalman', {}).get('trust_ratio', 500))
    smooth = int(config_dict.get('filtering', {}).get('kalman', {}).get('smooth', True))
    measurement_noise = 20
    process_noise = measurement_noise * trustratio

    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)  | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 2]

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

    type = 'low' #config_dict.get('filtering', {}).get('butterworth', {}).get('type')
    order = int(config_dict.get('filtering', {}).get('butterworth', {}).get('order', 4))
    cutoff = int(config_dict.get('filtering', {}).get('butterworth', {}).get('cut_off_frequency', 6))    

    b, a = signal.butter(order/2, cutoff/(frame_rate/2), type, analog = False) 
    padlen = 3 * max(len(a), len(b))
    
    # split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)  | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
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

    type = 'low' # config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('type')
    order = int(config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('order', 4))
    cutoff = int(config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('cut_off_frequency', 10))

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

    kernel = config_dict.get('filtering', {}).get('loess', config_dict.get('filtering', {}).get('LOESS', {})).get('nb_values_used', 5)

    col_filtered = col.copy()
    mask = np.isnan(col_filtered) 
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1 
    idx_sequences = np.split(falsemask_indices, gaps)
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
    
    median_filter_kernel_size = config_dict.get('filtering', {}).get('median', {}).get('kernel_size', 3)
    
    col_filtered = signal.medfilt(col, kernel_size=median_filter_kernel_size)

    return col_filtered


def display_figures_trc(Q_unfilt, Q_filt, time_col, keypoints_names, person_id=0, show=True):
    '''
    Displays filtered and unfiltered data for comparison

    INPUTS:
    - Q_unfilt: pandas dataframe of unfiltered 3D coordinates
    - Q_filt: pandas dataframe of filtered 3D coordinates
    - time_col: pandas column
    - keypoints_names: list of strings
    - person_id: int, person number
    - show: boolean, if True shows the figures    

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''

    os_name = platform.system()
    if os_name == 'Windows':
        mpl.use('qtagg') # windows
    mpl.rc('figure', max_open_warning=0)

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
    
    if show:
        pw.show()

    return pw


def display_figures_mot(Q_unfilt, Q_filt, time_col, col_names, person_id=0, show=True):
    '''
    Displays filtered and unfiltered mot data for comparison.
    Each column gets its own tab.

    INPUTS:
    - Q_unfilt: pandas dataframe of unfiltered data
    - Q_filt: pandas dataframe of filtered data
    - time_col: pandas Series of time values
    - col_names: array of column names
    - person_id: int
    - show: boolean

    OUTPUT:
    - plotWindow with tabbed figures
    '''

    os_name = platform.system()
    if os_name == 'Windows':
        mpl.use('qtagg')
    mpl.rc('figure', max_open_warning=0)

    pw = plotWindow()
    pw.MainWindow.setWindowTitle('Person ' + str(person_id) + ' mot data')
    
    for id, col_name in enumerate(col_names):
        f = plt.figure()
        ax = plt.subplot(111)
        plt.plot(time_col.to_numpy(), Q_unfilt.iloc[:, id].to_numpy(), label='unfiltered')
        plt.plot(time_col.to_numpy(), Q_filt.iloc[:, id].to_numpy(), label='filtered')
        ax.set_ylabel(col_name)
        ax.set_xlabel('Time (s)')
        plt.legend()
        plt.title(col_name)
        pw.addPlot(col_name, f)
    
    if show:
        pw.show()

    return pw


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
        'butterworth': butterworth_filter_1d, 
        'acc_minimizing': acc_minimizing_filter_1d,
        'one_euro': one_euro_filter_1d,
        'gcv_spline': gcv_spline_filter_1d,
        'kalman': kalman_filter_1d,
        'butterworth_on_speed': butterworth_on_speed_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'loess': loess_filter_1d, 
        'median': median_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(config_dict, frame_rate, col)

    return col_filtered


def recap_filter3d(config_dict, output_path):
    '''
    Print a log message giving filtering parameters. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''

    # Read Config
    project_dir = config_dict.get('project', {}).get('project_dir', '.')
    pose3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    kinematics_dir = os.path.realpath(os.path.join(project_dir, 'kinematics'))
    save_plots = config_dict.get('filtering', {}).get('save_filt_plots', True)
    do_filter = config_dict.get('filtering', {}).get('filter', True)
    reject_outliers = config_dict.get('filtering', {}).get('reject_outliers', False)
    filter_type = config_dict.get('filtering', {}).get('type', 'butterworth')
    filter_ik = config_dict.get('filtering', {}).get('filter_ik', False)
    plots_output_dir = os.path.join(kinematics_dir, 'filtering_plots') if filter_ik else os.path.join(pose3d_dir, 'filtering_plots_ik')
    kalman_filter_trustratio = int(config_dict.get('filtering', {}).get('kalman', {}).get('trust_ratio', 500))
    kalman_filter_smooth = int(config_dict.get('filtering', {}).get('kalman', {}).get('smooth', True))
    kalman_filter_smooth_str = 'smoother' if kalman_filter_smooth else 'filter'
    accminimizing_filter_cutoff = float(config_dict.get("filtering", {}).get("acc_minimizing", {}).get("cut_off_frequency", 6.0))
    butterworth_filter_type = 'low' # config_dict.get('filtering', {}).get('butterworth', {}).get('type')
    butterworth_filter_order = int(config_dict.get('filtering', {}).get('butterworth', {}).get('order', 4))
    butterworth_filter_cutoff = int(config_dict.get('filtering', {}).get('butterworth', {}).get('cut_off_frequency', 6))
    gcv_filter_cutoff = config_dict.get('filtering', {}).get('gcv_spline', {}).get('cut_off_frequency', 'auto')
    gcv_filter_smoothing_factor = float(config_dict.get('filtering', {}).get('gcv_spline', {}).get('smoothing_factor', 1.0))
    one_euro_filter_1d_min_cutoff = config_dict.get('filtering', {}).get('one_euro', {}).get('cut_off_frequency', 2.5)
    one_euro_filter_1d_beta = config_dict.get('filtering', {}).get('one_euro', {}).get('beta', 0.9)
    one_euro_filter_1d_d_cutoff = config_dict.get('filtering', {}).get('one_euro', {}).get('d_cut_off_frequency', 1.0)
    butter_speed_filter_type = 'low' # config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('type')
    butter_speed_filter_order = int(config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('order', 4))
    butter_speed_filter_cutoff = int(config_dict.get('filtering', {}).get('butterworth_on_speed', {}).get('cut_off_frequency', 10))
    gaussian_filter_sigma_kernel = int(config_dict.get('filtering', {}).get('gaussian', {}).get('sigma_kernel', 1))
    loess_filter_nb_values = config_dict.get('filtering', {}).get('loess', config_dict.get('filtering', {}).get('LOESS', {})).get('nb_values_used', 5)
    median_filter_kernel_size = config_dict.get('filtering', {}).get('median', {}).get('kernel_size', 3)
    make_c3d = config_dict.get('filtering', {}).get('make_c3d', True)
    
    # Recap
    if reject_outliers:
        logging.info('--> Outliers rejected with a Hampel filter.')
    else:
        logging.info('--> No outlier rejection applied. Set reject_outliers to true in Config.toml to reject outliers.')
    if do_filter:
        filter_mapping_recap = {
            'butterworth': f'--> Filter type: Butterworth {butterworth_filter_type}-pass. Order {butterworth_filter_order}, Cut-off frequency {butterworth_filter_cutoff} Hz.', 
            'acc_minimizing': f'--> Filter type: Acceleration-minimizing. Cut-off frequency: {accminimizing_filter_cutoff} Hz.',
            'one_euro': f'--> Filter type: OneEuro (zero-phase). Min cutoff frequency: {one_euro_filter_1d_min_cutoff} Hz, Beta: {one_euro_filter_1d_beta}, Derivative cutoff frequency: {one_euro_filter_1d_d_cutoff} Hz.',
            'gcv_spline': f'--> Filter type: Generalized Cross-Validation Spline. {f"Optimal parameters automatically estimated with smoothing factor {gcv_filter_smoothing_factor}" if gcv_filter_cutoff == "auto" else "Cut-off frequency {gcv_filter_cutoff} Hz"}.',
            'kalman': f'--> Filter type: Kalman {kalman_filter_smooth_str}. Measurements trusted {kalman_filter_trustratio} times as much as previous data, assuming a constant acceleration process.', 
            'butterworth_on_speed': f'--> Filter type: Butterworth on speed {butter_speed_filter_type}-pass. Order {butter_speed_filter_order}, Cut-off frequency {butter_speed_filter_cutoff} Hz.', 
            'gaussian': f'--> Filter type: Gaussian. Standard deviation kernel: {gaussian_filter_sigma_kernel}', 
            'loess': f'--> Filter type: LOESS. Number of values used: {loess_filter_nb_values}', 
            'median': f'--> Filter type: Median. Kernel size: {median_filter_kernel_size}'
        }
        logging.info(filter_mapping_recap[filter_type])
    else:
        logging.info('--> No filtering applied. Set filtering to true in Config.toml to filter coordinates.')
    logging.info(f'Filtered {"IK .mot" if filter_ik else "3D .trc"} coordinates are stored at {output_path}.')
    if make_c3d and not filter_ik:
        logging.info('All filtered trc files have been converted to c3d.')
    if save_plots:
        logging.info(f'Filtering plots are saved in {plots_output_dir}.')


def filter_all(config_dict):
    '''
    Filter the 3D coordinates of trc files, or IK results from mot files.
    Set filter_ik to true in Config.toml to filter mot files instead of trc files.
    Displays filtered coordinates for checking.

    INPUTS:
    - trc or mot files
    - filtration parameters from Config.toml

    OUTPUT:
    - filtered trc or mot files
    '''

    # Read config_dict
    project_dir = config_dict.get('project', {}).get('project_dir', '.')
    pose3d_dir = os.path.realpath(os.path.join(project_dir, 'pose-3d'))
    frame_range = config_dict.get('project', {}).get('frame_range', 'auto')
    display_figures = config_dict.get('filtering', {}).get('display_figures', True)
    save_plots = config_dict.get('filtering', {}).get('save_filt_plots', True)
    do_filter = config_dict.get('filtering', {}).get('filter', True)
    reject_outliers = config_dict.get('filtering', {}).get('reject_outliers', False)
    filter_type = config_dict.get('filtering', {}).get('type', 'butterworth')
    make_c3d = config_dict.get('filtering', {}).get('make_c3d', True)
    
    filter_ik = config_dict.get('temp_filter_ik', False) # only applied when run from Pose2Sim.kinematics()
    kinematics_dir = os.path.realpath(os.path.join(project_dir, 'kinematics'))
    plots_output_dir = os.path.join(kinematics_dir, 'filtering_plots') if filter_ik else os.path.join(pose3d_dir, 'filtering_plots_ik')
    if save_plots and not os.path.exists(plots_output_dir):
        os.makedirs(plots_output_dir)
    
    # Find input files
    if filter_ik:
        kinematics_dir = os.path.realpath(os.path.join(project_dir, 'kinematics'))
        files_in = [f for f in glob.glob(os.path.join(kinematics_dir, '*.mot')) if f.count('filt') <= 1]
        if not files_in:
            logging.error(f'No .mot files found in {kinematics_dir}. '
                        f'Make sure inverse kinematics has been run before filtering mot files, '
                        f'or set filter_ik to false in Config.toml to filter trc files instead.')
            raise FileNotFoundError(f'No .mot files found in {kinematics_dir} or {pose3d_dir}. '
                                    f'Make sure inverse kinematics has been run before filtering mot files, '
                                    f'or set filter_ik to false in Config.toml to filter trc files instead.')
    else:
        files_in = [f for f in glob.glob(os.path.join(pose3d_dir, '*.trc')) if 'filt' not in f]

    for person_id, file_path_in in enumerate(files_in):
        logging.info(f'\nFiltering {"IK mot" if filter_ik else "3D trc"} data for person {person_id}...')

        # Read file
        if filter_ik:
            Q_coords, time_col, header = read_mot(file_path_in)
            file_path_out = file_path_in.replace('.mot', f'_filt_{filter_type}.mot')
        else:
            Q_coords, frames_col, time_col, markers, header = read_trc(file_path_in)
            f_range = [[frames_col.iloc[0], frames_col.iloc[-1]]
                       if (frame_range in ('all', 'auto', []) or frames_col.iloc[0]>frame_range[0] or frames_col.iloc[1]<frame_range[1]) 
                       else frame_range][0]
            f_index = [frames_col[frames_col==f_range[0]].index[0], frames_col[frames_col==f_range[1]-1].index[0]+1]
            Q_coords = Q_coords.iloc[f_index[0]:f_index[1]].reset_index(drop=True)
            frames_col = frames_col.iloc[f_index[0]:f_index[1]].reset_index(drop=True)
            time_col = time_col.iloc[f_index[0]:f_index[1]].reset_index(drop=True)
            file_path_out = file_path_in.replace(file_path_in.split('_')[-1], f'{f_range[0]}-{f_range[1]}_filt_{filter_type}.trc')
            file_out = os.path.basename(file_path_out)
            header[0] = header[0].replace(os.path.basename(file_path_in), file_out)
        frame_rate = (1/time_col.diff().mean()).round()

        # Filter coordinates
        if reject_outliers:
            Q_coords = Q_coords.apply(hampel_filter, axis=0)  # Hampel filter for outlier rejection
        
        if do_filter:
            Q_filt = Q_coords.apply(filter1d, axis=0, args = [config_dict, filter_type, frame_rate])

        if not do_filter and not reject_outliers:
            logging.warning(f'Reject_outliers and filter have been set to false. No further processing done on {file_path_in}.\n')
        
        else:
            # Display figures
            if display_figures or save_plots:
                col_names = Q_coords.columns.to_numpy() if filter_ik else Q_coords.columns.to_numpy()[::3]
                fig_args = Q_coords, Q_filt, time_col, col_names, person_id, display_figures
                pw = display_figures_mot(*fig_args) if filter_ik else display_figures_trc(*fig_args)
                if save_plots:
                    for n, f in enumerate(pw.figure_handles):
                        dpi = pw.canvases[0].figure.dpi
                        f.set_size_inches(1280/dpi, 720/dpi)
                        title = pw.tabs.tabText(n)
                        plot_path = os.path.join(plots_output_dir, f'person{person_id:02d}_{title.replace(" ","_").replace("/","_")}.png')
                        f.savefig(plot_path, dpi=dpi, bbox_inches='tight')

            # Write output file
            if filter_ik:
                write_mot(file_path_out, Q_filt, time_col, header)
            else:
                write_trc(file_path_out, Q_filt, frames_col, time_col, header)
                if make_c3d:
                    convert_to_c3d(file_path_out)

            # Recap
            recap_filter3d(config_dict, file_path_out)
