#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## EASYMOCAP CALIBRATION TO TOML CALIBRATION    ##
    ##################################################
    
    Converts OpenCV intrinsic and extrinsic .yml calibration files 
    to an OpenCV .toml calibration file
    
    N.B. : Size is calculated as twice the position of the optical center. 
    Please correct in the resulting .toml file if needed. Take your image size as a reference.
    
    Usage: 
        import calib_easymocap_to_toml; calib_yml_to_toml.calib_easymocap_to_toml_func(r'<intrinsic_yml_file>', r'<extrinsic_yml_file>')
        OR calib_easymocap_to_toml -i intrinsic_yml_file -e extrinsic_yml_file
        OR calib_easymocap_to_toml -i intrinsic_yml_file -e extrinsic_yml_file -o output_toml_file
'''


## INIT
import os
import argparse
import numpy as np
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
    '''
    Main function for command line use
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--intrinsic_file', required = True, help='EasyMocap intrinsic .yml calibration file')
    parser.add_argument('-e', '--extrinsic_file', required = True, help='EasyMocap extrinsic .yml calibration file')
    parser.add_argument('-t', '--toml_file', required=False, help='OpenCV .toml output calibration file')
    args = vars(parser.parse_args())
    
    calib_easymocap_to_toml_func(args)


def read_intrinsic_yml(intrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - S (image size)
    - K (intrinsic parameters)
    - D (distorsion)

    N.B. : Size is calculated as twice the position of the optical center. Please correct in the .toml file if needed.
    '''
    intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
    N = intrinsic_yml.getNode('names').size()
    S, D, K = [], [], []
    for i in range(N):
        name = intrinsic_yml.getNode('names').at(i).string()
        K.append(intrinsic_yml.getNode(f'K_{name}').mat())
        D.append(intrinsic_yml.getNode(f'dist_{name}').mat().flatten()[:-1])
        S.append([K[i][0,2]*2, K[i][1,2]*2])
    return S, K, D
    

def read_extrinsic_yml(extrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - R (extrinsic rotation, Rodrigues vector)
    - T (extrinsic translation)
    '''
    extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)
    N = extrinsic_yml.getNode('names').size()
    R, T = [], []
    for i in range(N):
        name = extrinsic_yml.getNode('names').at(i).string()
        R.append(extrinsic_yml.getNode(f'R_{name}').mat().flatten()) # R_1 pour Rodrigues, Rot_1 pour matrice
        T.append(extrinsic_yml.getNode(f'T_{name}').mat().flatten())
    return R, T


def read_calib_yml(intrinsic_path, extrinsic_path):
    '''
    Reads OpenCV .yml calibration files
    Returns 5 lists of size N (N=number of cameras):
    - C (camera name),
    - S (image size),
    - D (distorsion), 
    - K (intrinsic parameters),
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''
    S, K, D = read_intrinsic_yml(intrinsic_path)
    R, T = read_extrinsic_yml(extrinsic_path)
    C = np.array(range(len(S)))+1
    return C, S, D, K, R, T


def toml_write(toml_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file.
    '''
    with open(os.path.join(toml_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam=f'[cam_{c+1}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]},]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]},], [ 0.0, {K[c][1,1]}, {K[c][1,2]},], [ 0.0, 0.0, 1.0,],]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]},]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]},]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]},]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def calib_easymocap_to_toml_func(*args):
    '''
    Converts OpenCV intrinsic and extrinsic .yml calibration files 
    to an OpenCV .toml calibration file
    
    N.B. : Size is calculated as twice the position of the optical center. 
    Please correct in the resulting .toml file if needed. Take your image size as a reference.
    
    Usage: 
        import calib_yml_to_toml; calib_yml_to_toml.calib_yml_to_toml_func(r'<intrinsic_yml_file>', r'<extrinsic_yml_file>')
        OR calib_yml_to_toml -i intrinsic_yml_file -e extrinsic_yml_file
        OR calib_yml_to_toml -i intrinsic_yml_file -e extrinsic_yml_file -o output_toml_file
    '''
    try:
        intrinsic_path = os.path.realpath(args[0].get('intrinsic_file')) # invoked with argparse
        extrinsic_path = os.path.realpath(args[0].get('extrinsic_file'))
        if args[0]['toml_file'] == None:
            toml_path = os.path.join(os.path.dirname(intrinsic_path), 'Calib.toml')
        else:
            toml_path = os.path.realpath(args[0]['toml_file'])
    except:
        intrinsic_path = os.path.realpath(args[0]) # invoked as a function
        extrinsic_path = os.path.realpath(args[1])
        toml_path = os.path.join(os.path.dirname(intrinsic_path), 'Calib.toml')
     
    C, S, D, K, R, T = read_calib_yml(intrinsic_path, extrinsic_path)
    toml_write(toml_path, C, S, D, K, R, T)

    print(f'Calibration file generated at {toml_path}.\n')


if __name__ == '__main__':
    main()
