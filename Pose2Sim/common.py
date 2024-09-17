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
import json
import numpy as np
import re
import cv2
import c3d
import sys

import matplotlib as mpl
mpl.use('qt5agg')
mpl.rc('figure', max_open_warning=0)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="c3d")


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Maya-Mocap"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def common_items_in_list(list1, list2):
    '''
    Do two lists have any items in common at the same index?
    Returns True or False
    '''
    
    for i, j in enumerate(list1):
        if j == list2[i]:
            return True
    return False


def bounding_boxes(js_file, margin_percent=0.1, around='extremities'):
    '''
    Compute the bounding boxes of the people in the json file.
    Either around the extremities (with a margin)
    or around the center of the person (with a margin).

    INPUTS:
    - js_file: json file
    - margin_percent: margin around the person
    - around: 'extremities' or 'center'

    OUTPUT:
    - bounding_boxes: list of bounding boxes [x_min, y_min, x_max, y_max]
    '''

    bounding_boxes = []
    with open(js_file, 'r') as json_f:
        js = json.load(json_f)
        for people in range(len(js['people'])):
            if len(js['people'][people]['pose_keypoints_2d']) < 3: continue
            else:
                x = js['people'][people]['pose_keypoints_2d'][0::3]
                y = js['people'][people]['pose_keypoints_2d'][1::3]
                x_min, x_max = min(x), max(x)
                y_min, y_max = min(y), max(y)

                if around == 'extremities':
                    dx = (x_max - x_min) * margin_percent
                    dy = (y_max - y_min) * margin_percent
                    bounding_boxes.append([x_min-dx, y_min-dy, x_max+dx, y_max+dy])
                
                elif around == 'center':
                    x_mean, y_mean = np.mean(x), np.mean(y)
                    x_size = (x_max - x_min) * (1 + margin_percent)
                    y_size = (y_max - y_min) * (1 + margin_percent)
                    bounding_boxes.append([x_mean - x_size/2, y_mean - y_size/2, x_mean + x_size/2, y_mean + y_size/2])

    return bounding_boxes   


def retrieve_calib_params(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - S: (h,w) vectors as list of 2x1 arrays
    - K: intrinsic matrices as list of 3x3 arrays
    - dist: distortion vectors as list of 4x1 arrays
    - inv_K: inverse intrinsic matrices as list of 3x3 arrays
    - optim_K: intrinsic matrices for undistorting points as list of 3x3 arrays
    - R: rotation rodrigue vectors as list of 3x1 arrays
    - T: translation vectors as list of 3x1 arrays
    '''
    
    calib = toml.load(calib_file)

    S, K, dist, optim_K, inv_K, R, R_mat, T = [], [], [], [], [], [], [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            S.append(np.array(calib[cam]['size']))
            K.append(np.array(calib[cam]['matrix']))
            dist.append(np.array(calib[cam]['distortions']))
            optim_K.append(cv2.getOptimalNewCameraMatrix(K[c], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0])
            inv_K.append(np.linalg.inv(K[c]))
            R.append(np.array(calib[cam]['rotation']))
            R_mat.append(cv2.Rodrigues(R[c])[0])
            T.append(np.array(calib[cam]['translation']))
    calib_params = {'S': S, 'K': K, 'dist': dist, 'inv_K': inv_K, 'optim_K': optim_K, 'R': R, 'R_mat': R_mat, 'T': T}
            
    return calib_params


def computeP(calib_file, undistort=False):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    - undistort: boolean
    
    OUTPUT:
    - P: projection matrix as list of arrays
    '''
    
    calib = toml.load(calib_file)
    
    P = []
    for cam in list(calib.keys()):
        if cam != 'metadata':
            K = np.array(calib[cam]['matrix'])
            if undistort:
                S = np.array(calib[cam]['size'])
                dist = np.array(calib[cam]['distortions'])
                optim_K = cv2.getOptimalNewCameraMatrix(K, dist, [int(s) for s in S], 1, [int(s) for s in S])[0]
                Kh = np.block([optim_K, np.zeros(3).reshape(3,1)])
            else:
                Kh = np.block([K, np.zeros(3).reshape(3,1)])
            R, _ = cv2.Rodrigues(np.array(calib[cam]['rotation']))
            T = np.array(calib[cam]['translation'])
            H = np.block([[R,T.reshape(3,1)], [np.zeros(3), 1 ]])
            
            P.append(Kh @ H)
   
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
        Q = np.array([np.nan,np.nan,np.nan,1])
        
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
        x_calc.append(P_cam[0] @ Q / (P_cam[2] @ Q))
        y_calc.append(P_cam[1] @ Q / (P_cam[2] @ Q))
        
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
    if np.isnan(dist).all():
        dist =  np.empty_like(dist)
        dist[...] = np.inf
    
    euc_dist = np.sqrt(np.nansum( [d**2 for d in dist]))
    
    return euc_dist


def world_to_camera_persp(r, t):
    '''
    Converts rotation R and translation T 
    from Qualisys world centered perspective
    to OpenCV camera centered perspective
    and inversely.

    Qc = RQ+T --> Q = R-1.Qc - R-1.T
    '''

    r = r.T
    t = - r @ t 

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
    r_ax = r_ax_z @ r_ax_y @ r_ax_x

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h @ rt_h
    
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


def sort_stringlist_by_last_number(string_list):
    '''
    Sort a list of strings based on the last number in the string.
    Works if other numbers in the string, if strings after number. Ignores alphabetical order.

    Example: ['json1', 'zero', 'js4on2.b', 'aaaa', 'eypoints_0000003.json', 'ajson0', 'json10']
    gives: ['ajson0', 'json1', 'js4on2.b', 'eypoints_0000003.json', 'json10', 'aaaa', 'zero']
    '''

    def sort_by_last_number(s):
        numbers = re.findall(r'\d+', s)
        if numbers:
            return (False, int(numbers[-1]))
        else:
            return (True, s)
    
    return sorted(string_list, key=sort_by_last_number)


def natural_sort_key(s):
    '''
    Sorts list of strings with numbers in natural order (alphabetical and numerical)
    Example: ['item_1', 'item_2', 'item_10', 'stuff_1']
    '''
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def zup2yup(Q):
    '''
    Turns Z-up system coordinates into Y-up coordinates
    INPUT:
    - Q: pandas dataframe
    N 3D points as columns, ie 3*N columns in Z-up system coordinates
    and frame number as rows
    OUTPUT:
    - Q: pandas dataframe with N 3D points in Y-up system coordinates
    '''

    # X->Y, Y->Z, Z->X
    cols = list(Q.columns)
    cols = np.array([[cols[i*3+1],cols[i*3+2],cols[i*3]] for i in range(int(len(cols)/3))]).flatten()
    Q = Q[cols]

    return Q
    

def extract_trc_data(trc_path):
    '''
    Extract marker names and coordinates from a trc file.

    INPUTS:
    - trc_path: Path to the trc file

    OUTPUTS:
    - marker_names: List of marker names
    - marker_coords: Array of marker coordinates (n_frames, t+3*n_markers)
    '''

    # marker names
    with open(trc_path, 'r') as file:
        lines = file.readlines()
        marker_names_line = lines[3]
        marker_names = marker_names_line.strip().split('\t')[2::3]

    # time and marker coordinates
    trc_data_np = np.genfromtxt(trc_path, skip_header=5, delimiter = '\t')[:,1:] 

    return marker_names, trc_data_np


def create_c3d_file(c3d_path, marker_names, trc_data_np):
    '''
    Create a c3d file from the data extracted from a trc file.

    INPUTS:
    - c3d_path: Path to the c3d file
    - marker_names: List of marker names
    - trc_data_np: Array of marker coordinates (n_frames, t+3*n_markers)

    OUTPUTS:
    - c3d file
    '''

    # retrieve frame rate
    times = trc_data_np[:,0]
    frame_rate = round((len(times)-1) / (times[-1] - times[0]))

    # write c3d file
    writer = c3d.Writer(point_rate=frame_rate, analog_rate=0, point_scale=1.0, point_units='mm', gen_scale=-1.0)
    writer.set_point_labels(marker_names)
    writer.set_screen_axis(X='+Z', Y='+Y')
    
    for frame in trc_data_np:
        residuals = np.full((len(marker_names), 1), 0.0)
        cameras = np.zeros((len(marker_names), 1))
        coords = frame[1:].reshape(-1,3)*1000
        points = np.hstack((coords, residuals, cameras))
        writer.add_frames([(points, np.array([]))])

    writer.set_start_frame(0)
    writer._set_last_frame(len(trc_data_np)-1)

    with open(c3d_path, 'wb') as handle:
        writer.write(handle)


def convert_to_c3d(trc_path):
    '''
    Make Visual3D compatible c3d files from a trc path

    INPUT:
    - trc_path: string, trc file to convert

    OUTPUT:
    - c3d file
    '''

    c3d_path = trc_path.replace('.trc', '.c3d')
    marker_names, trc_data_np = extract_trc_data(trc_path)
    create_c3d_file(c3d_path, marker_names, trc_data_np)

    return c3d_path


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
