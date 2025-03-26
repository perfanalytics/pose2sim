#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## TOML CALIBRATION TO EASYMOCAP CALIBRATION    ##
    ##################################################
    
    Convert a Pose2Sim .toml calibration file 
    to EasyMocap intrinsic and extrinsic .yml calibration files 

    Usage: 
        from Pose2Sim.Utilities import calib_toml_to_easymocap; calib_toml_to_easymocap.calib_toml_to_easymocap_func(r'<input_toml_file>')
        OR calib_easymocap_to_toml -t input_toml_file
        OR calib_easymocap_to_toml -t input_toml_file -i intrinsic_yml_file -e extrinsic_yml_file
'''

## INIT
import os
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
    parser.add_argument('-i', '--intrinsic_yml_file', required = False, help='OpenCV intrinsic .yml calibration file')
    parser.add_argument('-e', '--extrinsic_yml_file', required = False, help='OpenCV extrinsic .yml calibration file')
    args = vars(parser.parse_args())
    
    calib_toml_to_easymocap_func(args)

    
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


def write_intrinsic_yml(intrinsic_yml_path, C, D, K):
    '''
    Writes an OpenCV .yml intrinsic calibration file

    INPUTS:
    - Path of the intrinsic calibration file
    - C: list of camera names
    - D: list of distortion coefficients
    - K: list of intrinsic parameters
    '''

    # Names
    intrinsic_file = cv2.FileStorage(intrinsic_yml_path, cv2.FILE_STORAGE_WRITE)
    names = [f'{c}' for c in C]
    intrinsic_file.write("names", names)

    # Intrinsics and Distortions
    for i in range(len(C)):
        intrinsic_file.write(f"K_{i+1}", K[i])
        intrinsic_file.write(f"dist_{i+1}", np.append(D[i], [0]))
        
    intrinsic_file.release()


def write_extrinsic_yml(extrinsic_yml_path, C, R, T):
    '''
    Writes an OpenCV .yml extrinsic calibration file

    INPUTS:
    - Path of the extrinsic calibration file
    - C: list of camera names
    - R: list of extrinsic rotation matrices
    - T: list of extrinsic translation vectors
    '''

    # Names
    extrinsic_file = cv2.FileStorage(extrinsic_yml_path, cv2.FILE_STORAGE_WRITE)
    names = [f'{c}' for c in C]
    extrinsic_file.write("names", names)

    # Intrinsics and Distortions
    for i in range(len(C)):
        extrinsic_file.write(f"R_{i+1}", R[i])
        extrinsic_file.write(f"Rot_{i+1}", cv2.Rodrigues(R[i])[0])
        extrinsic_file.write(f"T_{i+1}", T[i])
        
    extrinsic_file.release()


def calib_toml_to_easymocap_func(*args):
    '''
    Convert a Pose2Sim .toml calibration file 
    to EasyMocap intrinsic and extrinsic .yml calibration files 

    Usage: 
        from Pose2Sim.Utilities import calib_toml_to_easymocap; calib_toml_to_easymocap.calib_toml_to_easymocap_func(r'<input_toml_file>')
        OR calib_easymocap_to_toml -t input_toml_file
        OR calib_easymocap_to_toml -t input_toml_file -i intrinsic_yml_file -e extrinsic_yml_file
    '''
    
    try:
        toml_path = os.path.realpath(args[0].get('toml_file')) # invoked with argparse
        if args[0]['intrinsic_yml_file'] == None or args[0]['extrinsic_yml_file'] == None:
            intrinsic_yml_path = os.path.join(os.path.dirname(toml_path), 'Intrinsic.yml')
            extrinsic_yml_path = os.path.join(os.path.dirname(toml_path), 'Extrinsic.yml')
        else:
            intrinsic_yml_path = os.path.realpath(args[0]['intrinsic_yml_file'])
            extrinsic_yml_path = os.path.realpath(args[0]['extrinsic_yml_file'])
    except:
        toml_path = os.path.realpath(args[0]) # invoked as a function
        intrinsic_yml_path = os.path.join(os.path.dirname(toml_path), 'Intrinsic.yml')
        extrinsic_yml_path = os.path.join(os.path.dirname(toml_path), 'Extrinsic.yml')
        
    C, _, D, K, R, T = read_toml(toml_path)
    write_intrinsic_yml(intrinsic_yml_path, C, D, K)
    write_extrinsic_yml(extrinsic_yml_path, C, R, T)

    print(f'Calibration files generated at {intrinsic_yml_path}\n and {extrinsic_yml_path}.\n')


if __name__ == '__main__':
    main()

