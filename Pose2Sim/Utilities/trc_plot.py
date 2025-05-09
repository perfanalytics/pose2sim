#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## Plot TRC files                               ##
    ##################################################
    
    Display each point of a TRC file in a different matplotlib tab.
    
    Usage: 
        from Pose2Sim.Utilities import trc_plot; trc_plot.trc_plot_func(r'<input_trc_file>')
        OR trc_plot -i input_trc_file
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
from importlib.metadata import version
__version__ = version('pose2sim')
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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='trc input file')
    args = vars(parser.parse_args())
    
    trc_plot_func(args)
    

def display_figures_fun(Q, time_col, keypoints_names):
    '''
    Displays filtered and unfiltered data for comparison

    INPUTS:
    - Q: pandas dataframe of 3D coordinates
    - time_col: pandas column
    - keypoints_names: list of strings

    OUTPUT:
    - matplotlib window with tabbed figures for each keypoint
    '''
    
    pw = plotWindow()
    for id, keypoint in enumerate(keypoints_names):
        f = plt.figure()
        
        axX = plt.subplot(311)
        plt.plot(time_col, Q.iloc[:,id*3])
        plt.setp(axX.get_xticklabels(), visible=False)
        axX.set_ylabel(keypoint+' X')

        axY = plt.subplot(312)
        plt.plot(time_col, Q.iloc[:,id*3+1])
        plt.setp(axY.get_xticklabels(), visible=False)
        axY.set_ylabel(keypoint+' Y')

        axZ = plt.subplot(313)
        plt.plot(time_col, Q.iloc[:,id*3+2])
        axZ.set_ylabel(keypoint+' Z')
        axZ.set_xlabel('Time')

        pw.addPlot(keypoint, f)
    
    pw.show()


def trc_plot_func(*args):
    '''
    Plot trc files.
    
    Usage: 
        import trc_plot; trc_plot.trc_plot_func(r'<input_trc_file>')
        OR trc_plot -i input_trc_file
    '''
    
    try:
        trc_path = args[0].get('input_file') # invoked with argparse
    except:
        trc_path = args[0] # invoked as a function

    # Read trc coordinates values
    trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4)
    time_col =trc_df.iloc[:,1]
    Q_coord = trc_df.drop(trc_df.columns[[0, 1]], axis=1)

    # Display figures
    keypoints_names = pd.read_csv(trc_path, sep="\t", skiprows=3, nrows=0).columns[2::3].tolist()
    display_figures_fun(Q_coord, time_col, keypoints_names)


if __name__ == '__main__':
    main()