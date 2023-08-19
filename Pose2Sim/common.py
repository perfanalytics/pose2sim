#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## OTHER SHARED UTILITIES                                                ##
    ###########################################################################
    
    Functions shared between modules, and other utilities
    
'''

## INIT
import toml
import numpy as np
import re
import cv2

import matplotlib as mpl
mpl.use('qt5agg')
mpl.rc('figure', max_open_warning=0)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import sys


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Maya-Mocap"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def computeP(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    K, R, T, Kh, H = [], [], [], [], []
    P = []
    
    calib = toml.load(calib_file)
    for cam in list(calib.keys()):
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            Kh = np.block([K, np.zeros(3).reshape(3,1)])
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
            
            P.append(Kh.dot(H))
   
    return P


def weighted_triangulation(P_all,x_all,y_all,likelihood_all):
    '''
    Triangulation with direct linear transform,
    weighted with likelihood of joint pose estimation.
    
    INPUTS:
    - P_all: list of arrays. Projection matrices of all cameras
    - x_all,y_all: x, y 2D coordinates to triangulate
    - likelihood_all: likelihood of joint pose estimation
    
    OUTPUT:
    - Q: array of triangulated point (x,y,z,1.)
    '''
    
    A = np.empty((0,4))
    for c in range(len(x_all)):
        P_cam = P_all[c]
        A = np.vstack((A, (P_cam[0] - x_all[c]*P_cam[2]) * likelihood_all[c] ))
        A = np.vstack((A, (P_cam[1] - y_all[c]*P_cam[2]) * likelihood_all[c] ))
        
    if np.shape(A)[0] >= 4:
        S, U, Vt = cv2.SVDecomp(A)
        V = Vt.T
        Q = np.array([V[0][3]/V[3][3], V[1][3]/V[3][3], V[2][3]/V[3][3], 1])
    else: 
        Q = np.array([0.,0.,0.,1])
        
    return Q


def reprojection(P_all, Q):
    '''
    Reprojects 3D point on all cameras.
    
    INPUTS:
    - P_all: list of arrays. Projection matrix for all cameras
    - Q: array of triangulated point (x,y,z,1.)

    OUTPUTS:
    - x_calc, y_calc: list of coordinates of point reprojected on all cameras
    '''
    
    x_calc, y_calc = [], []
    for c in range(len(P_all)):  
        P_cam = P_all[c]
        x_calc.append(P_cam[0].dot(Q) / P_cam[2].dot(Q))
        y_calc.append(P_cam[1].dot(Q) / P_cam[2].dot(Q))
        
    return x_calc, y_calc
        

def euclidean_distance(q1, q2):
    '''
    Euclidean distance between 2 points (N-dim).
    
    INPUTS:
    - q1: list of N_dimensional coordinates of point
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    
    '''
    
    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    
    euc_dist = np.sqrt(np.sum( [d**2 for d in dist]))
    
    return euc_dist


def RT_qca2cv(r, t):
    '''
    Converts rotation R and translation T 
    from Qualisys object centered perspective
    to OpenCV camera centered perspective
    and inversely.

    Qc = RQ+T --> Q = R-1.Qc - R-1.T
    '''

    r = r.T
    t = - r.dot(t) 

    return r, t


def rotate_cam(r, t, ang_x=0, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    Angle in radians
    '''

    r,t = np.array(r), np.array(t)
    if r.shape == (3,3):
        rt_h = np.block([[r,t.reshape(3,1)], [np.zeros(3), 1 ]]) 
    elif r.shape == (3,):
        rt_h = np.block([[cv2.Rodrigues(r)[0],t.reshape(3,1)], [np.zeros(3), 1 ]])
    
    r_ax_x = np.array([1,0,0, 0,np.cos(ang_x),-np.sin(ang_x), 0,np.sin(ang_x),np.cos(ang_x)]).reshape(3,3) 
    r_ax_y = np.array([np.cos(ang_y),0,np.sin(ang_y), 0,1,0, -np.sin(ang_y),0,np.cos(ang_y)]).reshape(3,3)
    r_ax_z = np.array([np.cos(ang_z),-np.sin(ang_z),0, np.sin(ang_z),np.cos(ang_z),0, 0,0,1]).reshape(3,3) 
    r_ax = r_ax_z.dot(r_ax_y).dot(r_ax_x)

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h.dot(rt_h)
    
    r = r_ax_h__rt_h[:3,:3]
    t = r_ax_h__rt_h[:3,3]

    return r, t


def quat2rod(quat, scalar_idx=0):
    '''
    Converts quaternion to Rodrigues vector

    INPUT:
    - quat: quaternion. np.array of size 4
    - scalar_idx: index of scalar part of quaternion. Default: 0, sometimes 3

    OUTPUT:
    - rod: Rodrigues vector. np.array of size 3
    '''

    if scalar_idx == 0:
        w, qx, qy, qz = np.array(quat)
    if scalar_idx == 3:
        qx, qy, qz, w = np.array(quat)
    else:
        print('Error: scalar_idx should be 0 or 3')

    rodx = qx * np.tan(w/2)
    rody = qy * np.tan(w/2)
    rodz = qz * np.tan(w/2)
    rod = np.array([rodx, rody, rodz])

    return rod


def quat2mat(quat, scalar_idx=0):
    '''
    Converts quaternion to rotation matrix

    INPUT:
    - quat: quaternion. np.array of size 4
    - scalar_idx: index of scalar part of quaternion. Default: 0, sometimes 3

    OUTPUT:
    - mat: 3x3 rotation matrix
    '''

    if scalar_idx == 0:
        w, qx, qy, qz = np.array(quat)
    elif scalar_idx == 3:
        qx, qy, qz, w = np.array(quat)
    else:
        print('Error: scalar_idx should be 0 or 3')

    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx*qy - qz*w)
    r13 = 2 * (qx*qz + qy*w)
    r21 = 2 * (qx*qy + qz*w)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy*qz - qx*w)
    r31 = 2 * (qx*qz - qy*w)
    r32 = 2 * (qy*qz + qx*w)
    r33 = 1 - 2 * (qx**2 + qy**2)
    mat = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T

    return mat


def natural_sort(list): 
    '''
    Sorts list of strings with numbers in natural order
    Example: ['item_1', 'item_2', 'item_10']
    Taken from: https://stackoverflow.com/a/11150413/12196632
    '''

    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    
    return sorted(list, key=alphanum_key)


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