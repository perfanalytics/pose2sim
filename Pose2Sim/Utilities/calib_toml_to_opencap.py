#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## TOML CALIBRATION TO OPENCAP CALIBRATION      ##
    ##################################################
    
    Convert an OpenCV .toml calibration file 
    to OpenCap .pickle calibration files.
    One file will be created for each camera.

    Usage: 
        from Pose2Sim.Utilities import calib_toml_to_opencap; calib_toml_to_opencap.calib_toml_to_opencap_func(r'<input_toml_file>')
        OR calib_toml_to_opencap -t input_toml_file
        OR calib_toml_to_opencap -t input_toml_file -o output_calibration_folder>
'''

## INIT
import os
import pickle
import argparse
import numpy as np
import toml
import cv2


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
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--toml_file', required = True, help='Input OpenCV .toml calibration file')
    parser.add_argument('-o', '--output_calibration_folder', required = False, help='OpenCap calibration folder')
    args = vars(parser.parse_args())
    
    calib_toml_to_opencap_func(args)

    
def world_to_camera_persp(r, t):
    '''
    Converts rotation R and translation T 
    from Qualisys object centered perspective
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


def read_toml(toml_path):
    '''
    Read an OpenCV .toml calibration file
    Returns 5 lists of size N (N=number of cameras):
    - S (image size),
    - D (distorsion), 
    - K (intrinsic parameters),
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''

    calib = toml.load(toml_path)
    C, S, D, K, R, T = [], [], [], [], [], []
    for cam in list(calib.keys()):
        if cam != 'metadata':
            C += [calib[cam]['name']]
            S += [np.array(calib[cam]['size'])]
            D += [np.array(calib[cam]['distortions'])]
            K += [np.array(calib[cam]['matrix'])]
            R += [np.array(calib[cam]['rotation'])]
            T += [np.array(calib[cam]['translation'])]

    return C, S, D, K, R, T


def write_opencap_pickle(output_calibration_folder, C, S, D, K, R, T):
    '''
    Writes OpenCap .pickle calibration files

    Extrinsics in OpenCap are calculated with a vertical board for the world frame.
    As we want the world frame to be horizontal, we need to rotate cameras by -Pi/2 around x in the world frame. 
    T is good the way it is.

    INPUTS:
    - Path of the output calibration folder
    - C: list of camera names
    - S: list of image sizes
    - D: list of distortion coefficients
    - K: list of intrinsic parameters
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''
    
    for i in range(len(C)):
        # Transform rotation for vertical frame of reference (checkerboard vertical with OpenCap)
        R_mat = cv2.Rodrigues(R[i])[0] # transform in matrix
        R_w, T_w = world_to_camera_persp(R_mat, T[i]) # transform in world centered perspective
        R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi) # rotate cam wrt world frame
        R_c, T_c = world_to_camera_persp(R_w_90, T_w_90) # transform in camera centered perspective

        # retrieve data
        calib_data = {'distortion': np.append(D[i],np.array([0])),
                      'intrinsicMat': K[i],
                      'imageSize': np.expand_dims(S[i][::-1], axis=1),
                      'rotation': R_c,
                      'translation': np.expand_dims(T[i], axis=1)*1000,
                      'rotation_EulerAngles': cv2.Rodrigues(R_c)[0] # OpenCap calls these Euler angles but they are actually the Rodrigues vector (Euler is ambiguous)
                      }

        # write pickle
        with open(os.path.join(output_calibration_folder, f'cam{i:02d}.pickle'), 'wb') as f_out:
            pickle.dump(calib_data, f_out)
    

def calib_toml_to_opencap_func(*args):
    '''
    Convert an OpenCV .toml calibration file 
    to OpenCap .pickle calibration files.
    One file will be created for each camera.

    Usage: 
        from Pose2Sim.Utilities import calib_toml_to_opencap; calib_toml_to_opencap.calib_toml_to_opencap_func(r'<input_toml_file>')
        OR calib_toml_to_opencap -t input_toml_file
        OR calib_toml_to_opencap -t input_toml_file -o output_calibration_folder
    '''
    
    try:
        toml_path = os.path.realpath(args[0].get('toml_file')) # invoked with argparse
        if args[0]['output_calibration_folder'] == None:
            output_calibration_folder = os.path.dirname(toml_path)
        else:
            output_calibration_folder = os.path.realpath(args[0]['output_calibration_folder'])
    except:
        toml_path = os.path.realpath(args[0]) # invoked as a function
        output_calibration_folder = os.path.dirname(toml_path)
        
    C, S, D, K, R, T = read_toml(toml_path)
    write_opencap_pickle(output_calibration_folder, C, S, D, K, R, T)

    print(f'OpenCap calibration files generated at {output_calibration_folder}.\n')


if __name__ == '__main__':
    main()
