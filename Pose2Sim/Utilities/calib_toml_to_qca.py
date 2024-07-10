#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## TOML CALIBRATION TO QCA CALIBRATION          ##
    ##################################################
    
    Convert an OpenCV .toml calibration file
    to a Qualisys .qca.txt calibration file

    Usage: 
        from Pose2Sim.Utilities import calib_toml_to_qca; calib_toml_to_qca.calib_toml_to_qca_func(r'<input_toml_file>')
        OR python -m calib_toml_to_qca -i input_toml_file
        OR python -m calib_toml_to_qca -i input_toml_file --binning_factor 2 --pixel_size 5.54e-3 -o output_qca_file
'''


## INIT
import os
import argparse
import numpy as np
import toml
from lxml import etree
import cv2


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
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


def rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    '''

    rt_h = np.block([[r,t.reshape(3,1)], [np.zeros(3), 1 ]]) 
    
    r_ax_x = np.array([1,0,0, 0,np.cos(ang_x),-np.sin(ang_x), 0,np.sin(ang_x),np.cos(ang_x)]).reshape(3,3) 
    r_ax_y = np.array([np.cos(ang_y),0,np.sin(ang_y), 0,1,0, -np.sin(ang_y),0,np.cos(ang_y)]).reshape(3,3)
    r_ax_z = np.array([np.cos(ang_z),-np.sin(ang_z),0, np.sin(ang_z),np.cos(ang_z),0, 0,0,1]).reshape(3,3) 
    r_ax = r_ax_z @ r_ax_y @ r_ax_x

    r_ax_h = np.block([[r_ax,np.zeros(3).reshape(3,1)], [np.zeros(3), 1]])
    r_ax_h__rt_h = r_ax_h @ rt_h
    
    r = r_ax_h__rt_h[:3,:3]
    t = r_ax_h__rt_h[:3,3]

    return r, t


def qca_write(qca_path, C, S, D, K, R, T, binning_factor, pixel_size):
    '''
    Writes calibration parameters to a .qca.txt file.
    '''
    
    # OpenCV to Qualisys variables conversions
    S = [[int(ss*binning_factor) for ss in s] for s in S]
    R = [r.T for r in R]
    fm = [k[0,0]*binning_factor*pixel_size for k in K]
    K = [k*binning_factor*64 for k in K]
    D = [d*binning_factor*64 for d in D]

    # .qca.txt construction
    root = etree.Element('calibration', source=os.path.basename(qca_path), created='sometimes ago', qtmversion='none', type='regular', wandLength='none', maximumFrames="none", shortArmEnd="none", longArmEnd="none", longArmMiddle="none")
    etree.SubElement(root, 'results', stddev='0.', minmaxdiff='0.')
    cams = etree.SubElement(root, 'cameras')
    
    for c in range(len(C)):
        cam = etree.SubElement(cams, 'camera', active='1', pointcount='999999999', avgresidual='0.', serial=C[c], model='none', viewrotation='0')
        etree.SubElement(cam, 'fov_marker', left='0', top='0', right=str(S[c][0]), bottom=str(S[c][1]))
        etree.SubElement(cam, 'fov_marker_max', left='0', top='0', right=str(S[c][0]), bottom=str(S[c][1]))
        etree.SubElement(cam, 'fov_video', left='0', top='0', right=str(S[c][0]), bottom=str(S[c][1]))
        etree.SubElement(cam, 'fov_video_max', left='0', top='0', right=str(S[c][0]), bottom=str(S[c][1]))
        etree.SubElement(cam, 'transform', x=str(T[c][0]), y=str(T[c][1]), z=str(T[c][2]), 
                                           r11=str(R[c][0,0]), r12=str(R[c][0,1]), r13=str(R[c][0,2]), 
                                           r21=str(R[c][1,0]), r22=str(R[c][1,1]), r23=str(R[c][1,2]), 
                                           r31=str(R[c][2,0]), r32=str(R[c][2,1]), r33=str(R[c][2,2]))
        etree.SubElement(cam, 'intrinsic', focallength=str(fm[c]), 
                                           sensorMinU='0.000000', sensorMaxU=str(S[c][0]*64), sensorMinV='0.000000', sensorMaxV=str(S[c][1]*64), 
                                           focalLengthU=str(K[c][0,0]), focalLengthV=str(K[c][1,1]), centerPointU=str(K[c][0,2]), centerPointV=str(K[c][1,2]), skew='0.000000', 
                                           radialDistortion1=str(D[c][0]), radialDistortion2=str(D[c][1]), radialDistortion3='0.000000', tangentalDistortion1=str(D[c][2]), tangentalDistortion2=str(D[c][3]))

    etree.ElementTree(root).write(qca_path, xml_declaration=True, pretty_print=True)

    # python XML file: had to delete hyphens in qtm-version, std-dev, min-max-diff, point-count, avg-residual' -> Replace them now
    with open(qca_path, 'r') as f:
        sample1 = f.read().replace('qtmversion', 'qtm-version', 1)
        sample2 = sample1.replace('stddev', 'std-dev', 1)
        sample3 = sample2.replace('minmaxdiff', 'min-max-diff', 1)
        sample4 = sample3.replace('pointcount', 'point-count')
        sample5 = sample4.replace('avgresidual', 'avg-residual')
    with open(qca_path, 'w') as f:    
        f.write(sample5)


def calib_toml_to_qca_func(**args):
    '''
    Convert an OpenCV .toml calibration file
    to a Qualisys .qca.txt calibration file

    Usage: 
        import calib_toml_to_qca; calib_toml_to_qca.calib_toml_to_qca_func(input_file=r'<input_toml_file>')
        OR calib_toml_to_qca -i input_toml_file
        OR calib_toml_to_qca -i input_toml_file --binning_factor 2 --pixel_size 5.54e-3 -o output_qca_file
    '''
    
    toml_path = args.get('input_file')
    qca_path = args.get('output_file')
    if qca_path == None:
        qca_path = toml_path.replace('.toml', '.qca.txt')
    
    binning_factor = args.get('binning_factor')
    if binning_factor == None:
        binning_factor = 1
    binning_factor = int(binning_factor)
    
    pixel_size = args.get('pixel_size')
    if pixel_size == None:
        pixel_size = 5.54e-3
    pixel_size = float(pixel_size)

    C, S, D, K, R, T = read_toml(toml_path)
    
    R = [np.array(cv2.Rodrigues(r)[0]) for r in R]
    T = np.array(T) * 1000

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]
    
    RT = [world_to_camera_persp(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    qca_write(qca_path, C, S, D, K, R, T, binning_factor, pixel_size)

    print('Calibration file generated.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='OpenCV .toml output calibration file')
    parser.add_argument('-b', '--binning_factor', required = False, default = 1, help='Binning factor if applied')
    parser.add_argument('-p', '--pixel_size', required = False, default = 5.54e-3, help='Pixel size in mm, 5.54e-3 mm by default (CMOS CMV2000)')
    parser.add_argument('-o', '--output_file', required=False, help='Qualisys .qca.txt input calibration file')
    args = vars(parser.parse_args())
    
    calib_toml_to_qca_func(**args)
    
