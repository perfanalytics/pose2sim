#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Filter TRC and MOT files                     ##
    ##################################################
    
    Filters trc or mot files (automatically detected from extension).
    Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median,
                       Kalman, GCV Spline, One Euro.
    Optional Hampel outlier rejection before filtering.

    Usage examples: 
    Butterworth filter, low-pass, 4th order, cut off frequency 6 Hz:
        from Pose2Sim.Utilities import trc_mot_filter; trc_mot_filter.trc_filter_func(input_file = 'input.trc', output_file = 'output.trc', 
            display=True, type='butterworth', pass_type='low', order=4, cut_off_frequency=6)
        OR trc_mot_filter -i input.trc -o output.trc -d True -t butterworth -p low -n 4 -f 6
        OR trc_mot_filter -i input.trc -t butterworth -p low -n 4 -f 6
        OR trc_mot_filter -i input.mot -t butterworth -p low -n 4 -f 6
    Butterworth filter on speed, low-pass, 4th order, cut off frequency 6 Hz:
        trc_mot_filter -i input.trc -t butterworth_on_speed -p low -n 4 -f 6
    Gaussian filter, kernel 5:
        trc_mot_filter -i input.trc -t gaussian -k 5
    LOESS filter, kernel 5: NB: frac = kernel * frames_number
        trc_mot_filter -i input.trc -t loess -k 5
    Median filter, kernel 5:
        trc_mot_filter -i input.trc -t median -k 5
    Kalman filter, trust ratio 500:
        trc_mot_filter -i input.trc -t kalman --trust_ratio 500
    GCV Spline filter, automatic cut-off:
        trc_mot_filter -i input.trc -t gcv_spline -f auto
    One Euro filter:
        trc_mot_filter -i input.trc -t one_euro -f 2.5 --beta 0.9
    With Hampel outlier rejection:
        trc_mot_filter -i input.trc -t butterworth -p low -n 4 -f 6 --reject_outliers
    
    Also works on .mot files:
        trc_mot_filter -i input.mot -t butterworth -p low -n 4 -f 6
'''


## INIT
import os
import math
import numpy as np
np.set_printoptions(legacy='1.21')
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
os_name = platform.system()
if os_name == 'Windows':
    mpl.use('qtagg')
mpl.rc('figure', max_open_warning=0)
from scipy import signal
from scipy import sparse
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import make_smoothing_spline
from scipy.interpolate._bsplines import _coeff_of_divided_diff, _compute_optimal_gcv_parameter
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import argparse


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


## CLASSES
def main():
    parser = argparse.ArgumentParser(description='Filter TRC or MOT files. File type is auto-detected from extension.')
    parser.add_argument('-i', '--input_file', required=True, help='trc or mot input file')
    parser.add_argument('-t', '--type', required=True, help='type of filter: "butterworth", '
        '"butterworth_on_speed", "loess"/"lowess", "gaussian", "median", "kalman", "gcv_spline", "one_euro", "acc_minimizing"')
    parser.add_argument('-d', '--display', required=False, default=True, help='display plots (True/False)')
    parser.add_argument('-o', '--output_file', required=False, help='filtered output file')
    
    # Butterworth, Butterworth on speed, GCV Spline, One Euro, Acceleration minimizing
    parser.add_argument('-f', '--cut_off_frequency', required=False, type=float, default=6, help='cut-off frequency in Hz (default: 6)')

    # Butterworth, Butterworth on speed
    parser.add_argument('-p', '--pass_type', required=False, default='low', help='"low" or "high" pass filter (default: low)')
    parser.add_argument('-n', '--order', required=False, type=int, default=4, help='filter order (default: 4)')

    # Gaussian, Median, LOESS
    parser.add_argument('-k', '--kernel', required=False, type=int, default=5, help='kernel of the median, gaussian, or loess filter (default: 5)')

    # Kalman
    parser.add_argument('--trust_ratio', required=False, type=int, default=500, help='Kalman filter trust ratio (default: 500)')
    parser.add_argument('--smooth', required=False, type=bool, default=True, help='Kalman smoother double pass (default: True)')

    # GCV Spline
    parser.add_argument('--smoothing_factor', required=False, type=float, default=1.0, help='GCV spline smoothing factor (default: 1.0)')

    # One Euro
    parser.add_argument('--beta', required=False, type=float, default=0.9, help='One Euro beta (default: 0.9)')

    # Hampel outlier rejection
    parser.add_argument('--reject_outliers', action='store_true', help='Apply Hampel filter for outlier rejection before filtering')
    parser.add_argument('--hampel_window', required=False, type=int, default=7, help='Hampel filter window size (default: 7)')
    parser.add_argument('--hampel_sigma', required=False, type=float, default=2, help='Hampel filter n_sigma threshold (default: 2)')

    # Frame rate override
    parser.add_argument('--frame_rate', required=False, type=float, default=None, help='Override frame rate (auto-detected from file if not set)')

    args = vars(parser.parse_args())
    
    trc_filter_func(**args)


class plotWindow():
    '''
    Display several figures in tabs
    Taken from https://github.com/superjax/plotWindow/blob/master/plotWindow.py

    USAGE:
    pw = plotWindow()
    f = plt.figure()
    plt.plot(x1, y1)
    pw.addPlot("1", f)
    f = plt.figure()
    plt.plot(x2, y2)
    pw.addPlot("2", f)
    '''

    def __init__(self, parent=None):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
        from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout

        self.FigureCanvas = FigureCanvas
        self.NavigationToolbar = NavigationToolbar
        self.QWidget = QWidget
        self.QVBoxLayout = QVBoxLayout

        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.setWindowTitle("Multitabs figure")
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(1280, 720)
        self.MainWindow.show()

    def addPlot(self, title, figure):
        new_tab = self.QWidget()
        layout = self.QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = self.FigureCanvas(figure)
        new_toolbar = self.NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec()


## FUNCTIONS
def hampel_filter(col, window_size=7, n_sigma=2):
    '''
    Hampel filter for outlier rejection before other filtering methods.
    Replaces value by median if difference is more than n_sigma times 
    the median absolute deviation within the sliding window.

    INPUT:
    - col: pandas Series or numpy array
    - window_size: int, sliding window size (default: 7)
    - n_sigma: float, number of sigma for threshold (default: 2)

    OUTPUT:
    - col_filtered: filtered array with outliers replaced by median
    '''

    col_filtered = col.copy()
    half_window = window_size // 2
    
    for i in range(half_window, len(col) - half_window):
        if np.isnan(col.iloc[i] if hasattr(col, 'iloc') else col[i]):
            continue
        window = col.iloc[i-half_window:i+half_window+1] if hasattr(col, 'iloc') else col[i-half_window:i+half_window+1]
        window_valid = window[~np.isnan(window)]
        if len(window_valid) == 0:
            continue
        median = np.median(window_valid)
        mad = np.median(np.abs(window_valid - median))
        
        if mad != 0:
            val = col.iloc[i] if hasattr(col, 'iloc') else col[i]
            modified_z_score = 0.6745 * (val - median) / mad
            if np.abs(modified_z_score) > n_sigma:
                if hasattr(col_filtered, 'iloc'):
                    col_filtered.iloc[i] = median
                else:
                    col_filtered[i] = median
    
    return col_filtered


def butterworth_filter_1d(col, **args):
    '''
    1D Zero-phase Butterworth filter (dual pass)
    Handles NaN values by splitting into contiguous sequences.

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary of pass_type, order, cut_off_frequency, frame_rate

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    butterworth_filter_type = args.get('pass_type', 'low')
    butterworth_filter_order = int(args.get('order', 4))
    butterworth_filter_cutoff = float(args.get('cut_off_frequency', 6))
    frame_rate = float(args.get('frame_rate', 30))

    b, a = signal.butter(butterworth_filter_order/2, butterworth_filter_cutoff/(frame_rate/2), butterworth_filter_type, analog=False)
    padlen = 3 * max(len(a), len(b))

    # Split into sequences of not nans
    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    
    if len(falsemask_indices) == 0:
        return col_filtered
    
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]

    for seq_f in idx_sequences_to_filter:
        col_filtered.iloc[seq_f] = signal.filtfilt(b, a, col_filtered.iloc[seq_f])
    
    return col_filtered


def butterworth_on_speed_filter_1d(col, **args):
    '''
    1D zero-phase Butterworth filter (dual pass) on derivative.
    Handles NaN values by splitting into contiguous sequences.

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary of pass_type, order, cut_off_frequency, frame_rate

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    butter_speed_filter_type = args.get('pass_type', 'low')
    butter_speed_filter_order = int(args.get('order', 4))
    butter_speed_filter_cutoff = float(args.get('cut_off_frequency', 10))
    frame_rate = float(args.get('frame_rate', 30))

    b, a = signal.butter(butter_speed_filter_order/2, butter_speed_filter_cutoff/(frame_rate/2), butter_speed_filter_type, analog=False)
    padlen = 3 * max(len(a), len(b))
    
    # Derivative
    col_filtered = col.copy()
    col_filtered_diff = col_filtered.diff()
    col_filtered_diff = col_filtered_diff.fillna(col_filtered_diff.iloc[1]/2 if len(col_filtered_diff) > 1 else 0)
    
    # Split into sequences of not nans
    mask = np.isnan(col_filtered_diff) | col_filtered_diff.eq(0)
    falsemask_indices = np.where(~mask)[0]
    
    if len(falsemask_indices) == 0:
        return col_filtered
    
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > padlen]

    for seq_f in idx_sequences_to_filter:
        col_filtered_diff.iloc[seq_f] = signal.filtfilt(b, a, col_filtered_diff.iloc[seq_f])
    col_filtered = col_filtered_diff.cumsum() + col.iloc[0]
    
    return col_filtered


def gaussian_filter_1d(col, **args):
    '''
    1D Gaussian filter

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with kernel value

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    gaussian_filter_sigma_kernel = int(args.get('kernel', 5))
    col_filtered = gaussian_filter1d(col, gaussian_filter_sigma_kernel)

    return col_filtered
    

def loess_filter_1d(col, **args):
    '''
    1D LOWESS filter (Locally Weighted Scatterplot Smoothing)
    Handles NaN values by splitting into contiguous sequences.

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with kernel: window used for smoothing 
    NB: frac = kernel / frames_number

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    loess_filter_nb_values = int(args.get('kernel', 5))

    col_filtered = col.copy()
    mask = np.isnan(col_filtered)
    falsemask_indices = np.where(~mask)[0]
    
    if len(falsemask_indices) == 0:
        return col_filtered
    
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) > loess_filter_nb_values]

    for seq_f in idx_sequences_to_filter:
        col_filtered.iloc[seq_f] = lowess(col_filtered.iloc[seq_f], seq_f, is_sorted=True, frac=loess_filter_nb_values/len(seq_f), it=0)[:,1]

    return col_filtered
    

def median_filter_1d(col, **args):
    '''
    1D median filter

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with "kernel" size
    
    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''
    
    median_filter_kernel_size = int(args.get('kernel', 5))
    col_filtered = signal.medfilt(col, kernel_size=median_filter_kernel_size)

    return col_filtered


def kalman_filter(coords, frame_rate, measurement_noise, process_noise, nb_dimensions=3, nb_derivatives=3, smooth=True):
    '''
    Filters coordinates with a Kalman filter or a Kalman smoother
    
    INPUTS:
    - coords: array of shape (nframes, ndims)
    - frame_rate: integer
    - measurement_noise: integer
    - process_noise: integer
    - nb_dimensions: integer, number of dimensions (3 if 3D coordinates, 1 if single column)
    - nb_derivatives: integer, number of derivatives (3 if constant acceleration model)
    - smooth: boolean. True if double pass (recommended), False if single pass (real-time)
    
    OUTPUTS:
    - coords_filt: filtered coords
    '''

    coords = np.array(coords)
    dim_x = nb_dimensions * nb_derivatives
    dt = 1/frame_rate
    
    f = KalmanFilter(dim_x=dim_x, dim_z=nb_dimensions)

    def derivate_array(arr, dt=1):
        return np.diff(arr, axis=0)/dt
    def repeat(func, arg_func, nb_reps):
        for i in range(nb_reps):
            arg_func = func(arg_func)
        return arg_func
    
    x_init = []
    for n_der in range(nb_derivatives):
        x_init += [repeat(derivate_array, coords, n_der)[0]]
    f.x = np.array(x_init).reshape(nb_dimensions, nb_derivatives).T.flatten()
    
    F_per_coord = np.zeros((int(dim_x/nb_dimensions), int(dim_x/nb_dimensions)))
    for i in range(nb_derivatives):
        for j in range(min(i+1, nb_derivatives)):
            F_per_coord[j,i] = dt**(i-j) / math.factorial(i - j)
    f.F = np.kron(np.eye(nb_dimensions), F_per_coord)

    f.B = None

    H = np.zeros((nb_dimensions, dim_x))
    for i in range(min(nb_dimensions, dim_x)):
        H[i, int(i*(dim_x/nb_dimensions))] = 1
    f.H = H

    f.P *= measurement_noise
    f.R = np.diag([measurement_noise**2]*nb_dimensions)
    f.Q = Q_discrete_white_noise(nb_derivatives, dt=dt, var=process_noise**2, block_size=nb_dimensions)

    mu, cov, _, _ = f.batch_filter(coords)
    ind_of_position = [int(d*(dim_x/nb_dimensions)) for d in range(nb_dimensions)]
    coords_filt = np.array(mu)[:, ind_of_position]

    if smooth:
        mu2, P, C, _ = f.rts_smoother(mu, cov)
        coords_filt = np.array(mu2)[:, ind_of_position]

    return coords_filt


def kalman_filter_1d(col, **args):
    '''
    1D Kalman filter. Handles NaN values by splitting into contiguous sequences.
    
    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with trust_ratio, smooth, frame_rate

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    trustratio = int(args.get('trust_ratio', 500))
    smooth = args.get('smooth', True)
    frame_rate = float(args.get('frame_rate', 30))
    measurement_noise = 20
    process_noise = measurement_noise * trustratio

    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    
    if len(falsemask_indices) == 0:
        return col_filtered
    
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 2]

    for seq_f in idx_sequences_to_filter:
        col_filtered.iloc[seq_f] = kalman_filter(col_filtered.iloc[seq_f], frame_rate, measurement_noise, process_noise, nb_dimensions=1, nb_derivatives=3, smooth=smooth).flatten()

    return col_filtered


def _compute_optimal_gcv_parameter_numstable(x, y):
    '''
    Makes x values spaced 1 apart for numerically stable GCV lambda computation.
    See https://stackoverflow.com/a/79740481/12196632
    '''
    
    x_spacing = np.diff(x)
    assert (x_spacing >= 0).all(), "x must be sorted"
    x_spacing_avg = x_spacing.mean()
    assert x_spacing_avg != 0, "div by zero"

    new_x = x / x_spacing_avg
    X, wE, y, w = _get_smoothing_spline_intermediate_arrays(new_x, y)
    
    lam = _compute_optimal_gcv_parameter(X, wE, y, w) * x_spacing_avg ** 3
    
    return lam


def _get_smoothing_spline_intermediate_arrays(x, y, w=None):
    '''
    Used by _compute_optimal_gcv_parameter_numstable to compute the optimal lambda value.
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
        wE[:, j] = (x[j+2] - x[j-2]) * _coeff_of_divided_diff(x[j-2:j+3]) / w[j-2: j+3]
    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6
    return X, wE, y, w


def gcv_spline_filter_1d(col, **args):
    '''
    If cutoff is a number, it is used as the cut-off frequency in Hz and behaves like a butterworth filter.
    If cutoff is 'auto', GCV finds the best trade-off between smoothness and fidelity to data (optimal lambda),
    and falls back to a biomechanically sensible frequency if GCV returns an unreliable result.
    Smoothing_factor biases results towards smoothing if > 1, and towards fidelity to input data if <1. Ignored if cutoff is not 'auto'.

    NOTE: In specific cases of a triangular wave + drift (eg pelvis Y coordinates of a subject walking uphill), 
    the filter may occasionally overfilter the trajectory.
    Known issue posted at: https://github.com/scipy/scipy/issues/23472


    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with cut_off_frequency, smoothing_factor, frame_rate

    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    cutoff = args.get('cut_off_frequency', 'auto')
    smoothing_factor = float(args.get('smoothing_factor', 1.0))
    frame_rate = float(args.get('frame_rate', 30))

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
                print(f'{col.name}: Automatically computed lambda is equivalent to a cut-off frequency of {frame_rate / (2.0 * np.pi * old_lam**0.25):.2f} Hz, which is outside of the expected range [{min_cutoff}, {max_cutoff}] Hz. Falling back to a lambda value corresponding to a cut-off frequency of {frame_rate / (2.0 * np.pi * lam**0.25):.2f} Hz. Your sequence might be too short, noisy, or near-constant for cutoff=\'auto\' to work effectively.')

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


def acc_minimizing_filter_1d(col, **args):
    '''
    1D Whittaker-Henderson filter (acceleration-minimizing).
    Inspired by AddBiomechanics: https://github.com/keenon/AddBiomechanics/blob/main/server/engine/src/dynamics_pass/acceleration_minimizing_pass.py

    INPUT:
    - args: dictionary with 'cut_off_frequency' (Hz)
    - frame_rate: float, frames per second
    - col: Pandas Series

    OUTPUT:
    - col_filtered: filtered Pandas Series (same index as input)
    '''
    
    cutoff = args.get('cut_off_frequency', 6)
    frame_rate = float(args.get('frame_rate', 30))
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


def one_euro_filter_1d(col, **args):
    '''
    Zero-phase One Euro filter for 1D signal with NaN handling.
    
    INPUT:
    - col: Pandas Series
    - args: dictionary with cut_off_frequency, beta, d_cutoff, frame_rate

    OUTPUT:
    - col_filtered: Filtered pandas Series
    '''
    
    min_cutoff = float(args.get('cut_off_frequency', 2.5))
    beta = float(args.get('beta', 0.9))
    d_cutoff = float(args.get('d_cutoff', 1.0))
    frame_rate = float(args.get('frame_rate', 30))
    dt = 1.0 / frame_rate
    
    def smoothing_factor(dt, cutoff):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def apply_filter(data):
        if len(data) < 2:
            return data

        filtered = [data[0]]
        x_prev = data[0]
        dx_prev = 0.0
        for i in range(1, len(data)):
            x = data[i]
            
            alpha_d = smoothing_factor(dt, d_cutoff)
            dx = (x - x_prev) / dt
            dx_hat = alpha_d * dx + (1 - alpha_d) * dx_prev
            
            cutoff = min_cutoff + beta * abs(dx_hat)
            alpha = smoothing_factor(dt, cutoff)
            x_hat = alpha * x + (1 - alpha) * x_prev
            
            filtered.append(x_hat)
            x_prev = x_hat
            dx_prev = dx_hat
        
        return np.array(filtered)
    
    col_filtered = col.copy()
    mask = np.isnan(col_filtered)
    falsemask_indices = np.where(~mask)[0]
    
    if len(falsemask_indices) == 0:
        return col_filtered
    
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)
    idx_sequences_to_filter = [seq for seq in idx_sequences if len(seq) >= 2]

    for seq in idx_sequences_to_filter:
        data = col_filtered.iloc[seq].values
        
        # Forward and backward passes for zero-phase filtering
        filtered_forward = apply_filter(data)
        filtered_backward = apply_filter(filtered_forward[::-1])[::-1]
        
        col_filtered.iloc[seq] = filtered_backward
    
    return col_filtered


def display_figures_trc(Q_unfilt, Q_filt, time_col, keypoints_names):
    '''
    Displays filtered and unfiltered TRC data for comparison

    INPUTS:
    - Q_unfilt: pandas dataframe of unfiltered 3D coordinates
    - Q_filt: pandas dataframe of filtered 3D coordinates
    - time_col: pandas column
    - keypoints_names: list of strings

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    pw = plotWindow()
    pw.MainWindow.setWindowTitle('TRC Filtering Results')
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
        axZ.set_xlabel('Time (s)')
        plt.legend()

        pw.addPlot(keypoint, f)
    
    pw.show()


def display_figures_mot(Q_unfilt, Q_filt, time_col, col_names):
    '''
    Displays filtered and unfiltered MOT data for comparison.
    Each column gets its own tab.

    INPUTS:
    - Q_unfilt: pandas dataframe of unfiltered data
    - Q_filt: pandas dataframe of filtered data
    - time_col: pandas Series of time values
    - col_names: array of column names

    OUTPUT:
    - matplotlib window with tabbed figures
    '''

    pw = plotWindow()
    pw.MainWindow.setWindowTitle('MOT Filtering Results')
    
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
    
    pw.show()


def filter1d(col, **args):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with type and filter-specific parameters
    
    OUTPUT:
    - col_filtered: Filtered pandas dataframe column
    '''

    filter_type = args.get('type')
    
    filter_mapping = {
        'butterworth': butterworth_filter_1d, 
        'butterworth_on_speed': butterworth_on_speed_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'loess': loess_filter_1d, 
        'lowess': loess_filter_1d,
        'median': median_filter_1d,
        'kalman': kalman_filter_1d,
        'gcv_spline': gcv_spline_filter_1d,
        'acc_minimizing': acc_minimizing_filter_1d,
        'one_euro': one_euro_filter_1d,
        }
    
    if filter_type not in filter_mapping:
        raise ValueError(f"Unknown filter type '{filter_type}'. Available: {list(filter_mapping.keys())}")
    
    filter_fun = filter_mapping[filter_type]
    col_filtered = filter_fun(col, **args)

    return col_filtered


def read_mot(mot_path):
    '''
    Read an OpenSim .mot file
    
    INPUT:
    - mot_path: path to the .mot file
    
    OUTPUT:
    - data: pandas DataFrame with all data columns, excluding time
    - time_col: pandas Series of time values
    - header: list of header lines
    '''
    
    header_lines = []
    header_end_line = 0
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if line.strip().lower() == 'endheader':
            header_end_line = i
            break
        
    header_lines = lines[:header_end_line + 1]
    
    data = pd.read_csv(mot_path, sep='\t', skiprows=header_end_line + 1)
    
    time_col = data.iloc[:, 0]
    data_cols = data.iloc[:, 1:]
    
    return data_cols, time_col, header_lines


def write_mot(mot_path, data, time_col, header_lines):
    '''
    Write a .mot file (OpenSim motion file).
    
    INPUT:
    - mot_path: path to the output .mot file
    - data: pandas DataFrame of filtered data columns
    - time_col: pandas Series of time values
    - header_lines: list of header lines
    '''
    
    with open(mot_path, 'w') as f:
        for line in header_lines:
            f.write(line)
        output = data.copy()
        output.insert(0, time_col.name if time_col.name else 'time', time_col)
        output.to_csv(f, sep='\t', index=False, lineterminator='\n')


def trc_filter_func(**args):
    '''
    Filters trc or mot files (auto-detected from file extension).
    Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median,
                       Kalman, GCV Spline, One Euro.
    Optional Hampel outlier rejection before filtering.

    Usage examples: 
    Butterworth filter, low-pass, 4th order, cut off frequency 6 Hz:
        trc_filter_func(input_file='input.trc', type='butterworth', pass_type='low', order=4, cut_off_frequency=6)
        trc_filter_func(input_file='input.mot', type='butterworth', pass_type='low', order=4, cut_off_frequency=6)
    Kalman filter:
        trc_filter_func(input_file='input.trc', type='kalman', trust_ratio=500)
    GCV Spline filter:
        trc_filter_func(input_file='input.trc', type='gcv_spline', gcv_cutoff='auto', smoothing_factor=1.0)
    One Euro filter:
        trc_filter_func(input_file='input.trc', type='one_euro', min_cutoff=2.5, beta=0.9)
    With outlier rejection:
        trc_filter_func(input_file='input.trc', type='butterworth', reject_outliers=True, pass_type='low', order=4, cut_off_frequency=6)
    '''

    input_path = args.get('input_file')
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    is_mot = (ext == '.mot')
    
    # --- Read file ---
    if is_mot:
        # MOT file
        Q_coords, time_col, header = read_mot(input_path)
        
        # Determine frame rate
        frame_rate_override = args.get('frame_rate')
        if frame_rate_override is not None:
            args['frame_rate'] = float(frame_rate_override)
        else:
            # Estimate from time column
            dt = time_col.diff().median()
            if dt > 0:
                args['frame_rate'] = round(1.0 / dt)
            else:
                args['frame_rate'] = 30
                print("Warning: Could not determine frame rate from .mot file. Defaulting to 30 fps.")
    else:
        # TRC file
        with open(input_path, 'r') as trc_file:
            header = [next(trc_file) for line in range(5)]
        
        frame_rate_override = args.get('frame_rate')
        if frame_rate_override is not None:
            args['frame_rate'] = float(frame_rate_override)
        else:
            args['frame_rate'] = float(header[2].split('\t')[0])

        trc_df = pd.read_csv(input_path, sep="\t", skiprows=4)
        frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # --- Outlier rejection ---
    reject_outliers = args.get('reject_outliers', False)
    if reject_outliers:
        hampel_window = int(args.get('hampel_window', 7))
        hampel_sigma = float(args.get('hampel_sigma', 2))
        Q_coords = Q_coords.apply(hampel_filter, axis=0, window_size=hampel_window, n_sigma=hampel_sigma)
        print(f"Outliers rejected with Hampel filter (window={hampel_window}, sigma={hampel_sigma}).")

    # --- Filter coordinates ---
    Q_filt = Q_coords.apply(filter1d, axis=0, **args)

    # --- Display figures ---
    display = args.get('display', True)
    if display == True or display == 'True' or display == 'true':
        if is_mot:
            col_names = Q_coords.columns.to_numpy()
            display_figures_mot(Q_coords, Q_filt, time_col, col_names)
        else:
            keypoints_names = pd.read_csv(input_path, sep="\t", skiprows=3, nrows=0).columns[2::3][:-1].to_numpy()
            display_figures_trc(Q_coords, Q_filt, time_col, keypoints_names)

    # --- Write output file ---
    filter_type = args.get('type', 'butterworth')
    output_path = args.get('output_file')
    
    if is_mot:
        if output_path is None:
            output_path = input_path.replace('.mot', f'_filt_{filter_type}.mot')
        write_mot(output_path, Q_filt, time_col, header)
    else:
        if output_path is None:
            output_path = input_path.replace('.trc', f'_filt_{filter_type}.trc')
        with open(output_path, 'w') as trc_o:
            [trc_o.write(line) for line in header]
            Q_filt.insert(0, 'Frame#', frames_col)
            Q_filt.insert(1, 'Time', time_col)
            Q_filt.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')

    # --- Summary ---
    filter_recap = {
        'butterworth': f"Butterworth {args.get('pass_type', 'low')}-pass, order {args.get('order', 4)}, cutoff {args.get('cut_off_frequency', 6)} Hz",
        'butterworth_on_speed': f"Butterworth on speed {args.get('pass_type', 'low')}-pass, order {args.get('order', 4)}, cutoff {args.get('cut_off_frequency', 10)} Hz",
        'gaussian': f"Gaussian, sigma kernel {args.get('kernel', 5)}",
        'loess': f"LOESS, kernel {args.get('kernel', 5)}",
        'lowess': f"LOESS, kernel {args.get('kernel', 5)}",
        'median': f"Median, kernel {args.get('kernel', 5)}",
        'kalman': f"Kalman {'smoother' if args.get('smooth', True) else 'filter'}, trust ratio {args.get('trust_ratio', 500)}",
        'gcv_spline': f"GCV Spline, cutoff {'auto' if args.get('cut_off_frequency', 'auto') == 'auto' else str(args.get('cut_off_frequency')) + ' Hz'}, smoothing factor {args.get('smoothing_factor', 1.0)}",
        'one_euro': f"One Euro, min cutoff {args.get('cut_off_frequency', 2.5)} Hz, beta {args.get('beta', 0.9)}, d_cutoff {args.get('d_cutoff', 1.0)} Hz",
        'acc_minimizing': f"Acceleration-minimizing (Whittaker-Henderson), cutoff {args.get('cut_off_frequency', 6)} Hz",
    }
    
    file_type = "mot" if is_mot else "trc"
    recap_str = filter_recap.get(filter_type, filter_type)
    outlier_str = " (with Hampel outlier rejection)" if reject_outliers else ""
    print(f"{file_type} file filtered with {recap_str}{outlier_str}: {output_path}")
    

if __name__ == '__main__':
    main()