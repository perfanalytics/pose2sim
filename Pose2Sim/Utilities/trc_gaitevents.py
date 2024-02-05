#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## GAIT EVENTS DETECTION                        ##
    ##################################################
    
    Determine gait events according to Zeni et al. (2008).
    Write them in gaitevents.txt (append results if file already exists).

    t_HeelStrike = max(XHeel - Xsacrum)
    t_ToeOff = min(XToe - XSacrum)

    Reference:
    “Two simple methods for determining gait events during treadmill and 
    overground walking using kinematic data.” 
    Gait & posture vol. 27,4 (2008): 710-4. doi:10.1016/j.gaitpost.2007.07.007

    Usage: 
        Replace constants with the appropriate marker names.
        If direction is negative, you need to include an equal sign in the argument, 
        eg -d=-Z or --gait_direction=-Z
        
        from Pose2Sim.Utilities import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(r'<input_trc_file>', '<gait_direction>')
        OR python -m trc_gaitevents -i input_trc_file
        OR python -m trc_gaitevents -i input_trc_file --gait_direction=-Z
'''


## CONSTANTS
R_SACRUM_MARKER = 'RHip'
R_HEEL_MARKER = 'RHeel'
R_TOE_MARKER = 'RBigToe'
L_SACRUM_MARKER = 'LHip'
L_HEEL_MARKER = 'LHeel'
L_TOE_MARKER = 'LBigToe'


## INIT
import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.6'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def df_from_trc(trc_path):
    '''
    Retrieve header and data from trc path.
    '''

    # DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
    df_header = pd.read_csv(trc_path, sep="\t", skiprows=1, header=None, nrows=2, encoding="ISO-8859-1")
    header = dict(zip(df_header.iloc[0].tolist(), df_header.iloc[1].tolist()))
    
    # Label1_X  Label1_Y    Label1_Z    Label2_X    Label2_Y
    df_lab = pd.read_csv(trc_path, sep="\t", skiprows=3, nrows=1)
    labels = df_lab.columns.tolist()[2:-1:3]
    labels_XYZ = np.array([[labels[i]+'_X', labels[i]+'_Y', labels[i]+'_Z'] for i in range(len(labels))], dtype='object').flatten()
    labels_FTXYZ = np.concatenate((['Frame#','Time'], labels_XYZ))
    
    data = pd.read_csv(trc_path, sep="\t", skiprows=5, index_col=False, header=None, names=labels_FTXYZ)
    
    return header, data


def gait_events(trc_path, gait_direction):
    '''
    Determine gait events according to Zeni et al. (2008).
    t_HellStrike = max(XHeel - Xsacrum)
    t_ToeOff = min(XToe - XSacrum)
    '''

    # Read trc
    header, data = df_from_trc(trc_path)

    # In case of a sign in direction (eg -Z)
    sign = ''
    if any(x in gait_direction for x in ['-', '+']):
        sign = gait_direction[0]
        gait_direction = gait_direction[-1]
        
    # Retrieve data of interest
    XRSacrum = data['_'.join((R_SACRUM_MARKER, gait_direction))]
    XRHeel = data['_'.join((R_HEEL_MARKER, gait_direction))]
    XRToe = data['_'.join((R_TOE_MARKER, gait_direction))]
    XLSacrum = data['_'.join((L_SACRUM_MARKER, gait_direction))]
    XLHeel = data['_'.join((L_HEEL_MARKER, gait_direction))]
    XLToe = data['_'.join((L_TOE_MARKER, gait_direction))]

    # Prominence of the peaks
    unit = header['Units']
    peak_prominence = .1 if unit=='m' else 1 if unit=='dm' else 10 if unit=='cm' else 100 if unit=='mm' else np.inf

    # Right and left heel strikes
    frame_RHS = signal.find_peaks(eval(sign+'(XRHeel-XRSacrum)'),prominence=peak_prominence)[0]
    t_RHS = data.loc[frame_RHS, 'Time'].tolist()

    frame_LHS = signal.find_peaks(eval(sign+'(XLHeel-XLSacrum)'),prominence=peak_prominence)[0]
    t_LHS = data.loc[frame_LHS, 'Time'].tolist()

    # Right and left toe offs
    frame_RTO = signal.find_peaks(eval(sign+'-(XRToe-XRSacrum)'),prominence=peak_prominence)[0]
    t_RTO = data.loc[frame_RTO, 'Time'].tolist()

    frame_LTO = signal.find_peaks(eval(sign+'-(XLToe-XLSacrum)'),prominence=peak_prominence)[0]
    t_LTO = data.loc[frame_LTO, 'Time'].tolist()

    return t_RHS, t_LHS, t_RTO, t_LTO


def trc_gaitevents_func(*args):
    '''
    Determine gait events according to Zeni et al. (2008).
    Write them in gaitevents.txt (append results if file already exists).

    t_HeelStrike = max(XHeel - Xsacrum)
    t_ToeOff = min(XToe - XSacrum)

    Reference:
    “Two simple methods for determining gait events during treadmill and 
    overground walking using kinematic data.” 
    Gait & posture vol. 27,4 (2008): 710-4. doi:10.1016/j.gaitpost.2007.07.007
    
    Usage: 
        Replace constants with the appropriate marker names in trc_gaitevents.py.
        If direction is negative, you need to include an equal sign in the argument, 
        eg -d=-Z or --gait_direction=-Z
        
        import trc_gaitevents; trc_gaitevents.trc_gaitevents_func(r'<input_trc_file>', '<gait_direction>')
        OR trc_gaitevents -i input_trc_file --gait_direction Z
        OR trc_gaitevents -i input_trc_file --gait_direction=-Z
    '''

    try:
        trc_path = args[0].get('input_file') # invoked with argparse
        gait_direction = args[0]['gait_direction']
    except:
        trc_path = args[0] # invoked as a function
        try:
            gait_direction = args[1]
        except:
            gait_direction = 'Z'

    trc_dir = os.path.dirname(trc_path)
    trc_name = os.path.basename(trc_path)

    t_RHS, t_LHS, t_RTO, t_LTO = gait_events(trc_path, gait_direction)

    with open(os.path.join(trc_dir, 'gaitevents.txt'), 'a') as gaitevents:
        L = trc_name + '\n'
        L += 'Right Heel strikes: ' + str(t_RHS) + '\n'
        L += 'Left Heel strikes: ' + str(t_LHS) + '\n'
        L += 'Right Toe off: ' + str(t_RTO) + '\n'
        L += 'Left Toe off: ' + str(t_LTO) + '\n\n'

        gaitevents.write(L)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='trc input file')
    parser.add_argument('-d', '--gait_direction', default = 'Z', required = False, help='direction of the gait. If negative, you need to include an equal sign in the argument, eg -d=-Z')
    args = vars(parser.parse_args())
    
    trc_gaitevents_func(args)
    