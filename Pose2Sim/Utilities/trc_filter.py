#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Filter TRC files                             ##
    ##################################################
    
    Filters trc files.
    Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median.

    Usage examples: 
    Butterworth filter, low-pass, 4th order, cut off frequency 6 Hz:
        from Pose2Sim.Utilities import trc_filter; trc_filter.trc_filter_func(input_file = input_trc_file, output_file = output_trc_file, 
            display=True, type='butterworth', pass_type = 'low', order=4, cut_off_frequency=6)
        OR python -m trc_filter -i input_trc_file -o output_trc_file -d True -t butterworth -p low -n 4 -f 6
        OR python -m trc_filter -i input_trc_file -t butterworth -p low -n 4 -f 6
    Butterworth filter on speed, low-pass, 4th order, cut off frequency 6 Hz:
        python -m trc_filter -i input_trc_file -t butterworth_on_speed -p low -n 4 -f 6
    Gaussian filter, kernel 5:
        python -m trc_filter -i input_trc_file -t gaussian, -k 5
    LOESS filter, kernel 5: NB: frac = kernel * frames_number
        python -m trc_filter -i input_trc_file -t loess, -k 5
    Median filter, kernel 5:
        python -m trc_filter -i input_trc_file -t gaussian, -k 5
'''


## INIT
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('qt5agg')
mpl.rc('figure', max_open_warning=0)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import argparse


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## CLASSES
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
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()
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
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        figure.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.91, wspace=0.2, hspace=0.2)
        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        self.app.exec_() 


## FUNCTIONS
def butterworth_filter_1d(col, **args):
    '''
    1D Zero-phase Butterworth filter (dual pass)

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary of pass_type, order, cut_off_frequency, frame_rate

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    butterworth_filter_type = args.get('pass_type')
    butterworth_filter_order = int(args.get('order'))
    butterworth_filter_cutoff = int(args.get('cut_off_frequency'))    
    frame_rate = int(args.get('frame_rate'))

    b, a = signal.butter(butterworth_filter_order/2, butterworth_filter_cutoff/(frame_rate/2), butterworth_filter_type, analog = False) 
    col_filtered = signal.filtfilt(b, a, col)
    
    return col_filtered


def butterworth_on_speed_filter_1d(col, **args):
    '''
    1D zero-phase Butterworth filter (dual pass) on derivative

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary of pass_type, order, cut_off_frequency, frame_rate

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    butter_speed_filter_type = args.get('pass_type')
    butter_speed_filter_order = int(args.get('order'))
    butter_speed_filter_cutoff = int(args.get('cut_off_frequency'))
    frame_rate = int(args.get('frame_rate'))

    b, a = signal.butter(butter_speed_filter_order/2, butter_speed_filter_cutoff/(frame_rate/2), butter_speed_filter_type, analog = False)
    
    col_diff = col.diff()   # derivative
    col_diff = col_diff.fillna(col_diff.iloc[1]/2) # set first value correctly instead of nan
    col_diff_filt = signal.filtfilt(b, a, col_diff) # filter derivative
    col_filtered = col_diff_filt.cumsum() + col.iloc[0] # integrate filtered derivative
    
    return col_filtered


def gaussian_filter_1d(col, **args):
    '''
    1D Gaussian filter

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with kernel value

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    gaussian_filter_sigma_kernel = int(args.get('kernel'))

    col_filtered = gaussian_filter1d(col, gaussian_filter_sigma_kernel)

    return col_filtered
    

def loess_filter_1d(col, **args):
    '''
    1D LOWESS filter (Locally Weighted Scatterplot Smoothing)

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with nb_values_used: window used for smoothing 
    NB: frac = nb_values_used * frames_number

    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    loess_filter_nb_values = int(args.get('kernel'))

    col_filtered = lowess(col, col.index, is_sorted=True, frac=loess_filter_nb_values/len(col), it=0)[:,1]

    return col_filtered
    

def median_filter_1d(col, **args):
    '''
    1D median filter

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with "kernel" size
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''
    
    median_filter_kernel_size = args.get('kernel')
    
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
        plt.plot(time_col, Q_unfilt.iloc[:,id*3], label='unfiltered')
        plt.plot(time_col, Q_filt.iloc[:,id*3], label='filtered')
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')
        plt.legend()

        axY = plt.subplot(312)
        plt.plot(time_col, Q_unfilt.iloc[:,id*3+1], label='unfiltered')
        plt.plot(time_col, Q_filt.iloc[:,id*3+1], label='filtered')
        plt.setp(axY.get_xticklabels(), visible=False)
        axY.set_ylabel(keypoint+' Y')
        plt.legend()

        axZ = plt.subplot(313)
        plt.plot(time_col, Q_unfilt.iloc[:,id*3+2], label='unfiltered')
        plt.plot(time_col, Q_filt.iloc[:,id*3+2], label='filtered')
        axZ.set_ylabel(keypoint+' Z')
        axZ.set_xlabel('Time')
        plt.legend()

        pw.addPlot(keypoint, f)
    
    pw.show()


def filter1d(col, **args):
    '''
    Choose filter type and filter column

    INPUT:
    - col: Pandas dataframe column
    - args: dictionary with filter_type: "butterworth", "butterworth_on_speed", 
    "loess"/"lowess", "gaussian", or "median"
    
    OUTPUT
    - col_filtered: Filtered pandas dataframe column
    '''

    # Choose filter
    filter_type = args.get('type')
    
    filter_mapping = {
        'butterworth': butterworth_filter_1d, 
        'butterworth_on_speed': butterworth_on_speed_filter_1d, 
        'gaussian': gaussian_filter_1d, 
        'loess': loess_filter_1d, 
        'median': median_filter_1d
        }
    filter_fun = filter_mapping[filter_type]
    
    # Filter column
    col_filtered = filter_fun(col, **args)

    return col_filtered


def trc_filter_func(**args):
    '''
    Filters trc files.
    Available filters: Butterworth, Butterworth on speed, Gaussian, LOESS, Median.

    Usage examples: 
    Butterworth filter, low-pass, 4th order, cut off frequency 6 Hz:
        import trc_filter; trc_filter.trc_filter_func(input_file = input_trc_file, output_file = output_trc_file, 
            display=True, type='butterworth', pass_type = 'low', order=4, cut_off_frequency=6)
        OR python -m trc_filter -i input_trc_file -o output_trc_file -d True -t butterworth -p low -n 4 -f 6
        OR python -m trc_filter -i input_trc_file -t butterworth, -p low -n 4 -f 6
    Butterworth filter on speed, low-pass, 4th order, cut off frequency 6 Hz:
        python -m trc_filter -i input_trc_file -t butterworth_on_speed, -p low -n 4 -f 6
    Gaussian filter, kernel 5:
        python -m trc_filter -i input_trc_file -t gaussian, -k 5
    LOESS filter, kernel 5: NB: frac = kernel * frames_number
        python -m trc_filter -i input_trc_file -t loess, -k 5
    Median filter, kernel 5:
        python -m trc_filter -i input_trc_file -t gaussian, -k 5
    '''

    # Read trc header
    trc_path_in = args.get('input_file')
    with open(trc_path_in, 'r') as trc_file:
        header = [next(trc_file) for line in range(5)]
    args['frame_rate'] = int(header[2].split('\t')[0])

    # Read trc coordinates values
    trc_df = pd.read_csv(trc_path_in, sep="\t", skiprows=4)
    frames_col, time_col = trc_df.iloc[:,0], trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # Filter coordinates
    Q_filt = Q_coord.apply(filter1d, axis=0, **args)

    # Display figures
    display = args.get('display')
    if display == True or display == 'True':
        keypoints_names = pd.read_csv(trc_path_in, sep="\t", skiprows=3, nrows=0).columns[2::3].tolist()
        display_figures_fun(Q_coord, Q_filt, time_col, keypoints_names)

    # Reconstruct trc file with filtered coordinates
    trc_path_out = args.get('output_file')
    if trc_path_out == None: 
        trc_path_out = trc_path_in.replace('.trc', '_filt.trc')
    with open(trc_path_out, 'w') as trc_o:
        [trc_o.write(line) for line in header]
        Q_filt.insert(0, 'Frame#', frames_col)
        Q_filt.insert(1, 'Time', time_col)
        Q_filt.to_csv(trc_o, sep='\t', index=False, header=None, lineterminator='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='trc input file')
    parser.add_argument('-t', '--type', required=True, help='type of filter. "butterworth", \
        "butterworth_on_speed", "loess"/"lowess", "gaussian", or "median"')
    parser.add_argument('-d', '--display', required = False, default = True, help='display plots')
    parser.add_argument('-o', '--output_file', required=False, help='filtered trc output file')
    
    parser.add_argument('-p', '--pass_type', required=False, help='"low" or "high" pass filter')
    parser.add_argument('-n', '--order', required=False, help='filter order')
    parser.add_argument('-f', '--cut_off_frequency', required=False, help='cut-off frequency')
    parser.add_argument('-k', '--kernel', required=False, help='kernel of the median, gaussian, or loess filter')

    args = vars(parser.parse_args())
    
    try:
        trc_filter_func(**args)
    except:
        print('ERROR: You probably passed bad arguments. See examples in trc_filter.py module header or with "help(trc_filter_func)"')
