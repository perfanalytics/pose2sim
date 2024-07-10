#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ##################################################
    ## QCA CALIBRATION TO TOML CALIBRATION          ##
    ##################################################
    
    Convert a Qualisys .qca.txt calibration file 
    to an OpenCV .toml calibration file

    Usage: 
        from Pose2Sim.Utilities import calib_qca_to_toml; calib_qca_to_toml.calib_qca_to_toml_func(r'<input_qca_file>')
        OR python -m calib_qca_to_toml -i input_qca_file
        OR python -m calib_qca_to_toml -i input_qca_file --binning_factor 2 -o output_toml_file
'''


## INIT
import os
import argparse
import re
import numpy as np
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
def natural_sort_key(s):
    """
    Key for natural sorting of strings containing numbers.
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def read_qca(qca_path, binning_factor):
    '''
    Read a Qualisys .qca.txt calibration file
    Returns 5 lists of size N (N=number of cameras):
    - ret: residual reprojection error in _mm_: list of floats
    - C (camera name),
    - S (image size),
    - D (distorsion), 
    - K (intrinsic parameters),
    - R (extrinsic rotation),
    - T (extrinsic translation)
    '''

    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    vid_id = []

    # Camera name
    for i, tag in enumerate(root.findall('cameras/camera')):
        ret += [float(tag.attrib.get('avg-residual'))/1000]
        C += [tag.attrib.get('serial')]
        if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
            vid_id += [i]

    # Image size
    for tag in root.findall('cameras/camera/fov_video'):
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left'))) /binning_factor
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top'))) /binning_factor
        S += [[w, h]]
    
    # Intrinsic parameters: distorsion and intrinsic matrix
    for i, tag in enumerate(root.findall('cameras/camera/intrinsic')):
        k1 = float(tag.get('radialDistortion1'))/64/binning_factor
        k2 = float(tag.get('radialDistortion2'))/64/binning_factor
        p1 = float(tag.get('tangentalDistortion1'))/64/binning_factor
        p2 = float(tag.get('tangentalDistortion2'))/64/binning_factor
        D+= [np.array([k1, k2, p1, p2])]
        
        fu = float(tag.get('focalLengthU'))/64/binning_factor
        fv = float(tag.get('focalLengthV'))/64/binning_factor
        cu = float(tag.get('centerPointU'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('left'))
        cv = float(tag.get('centerPointV'))/64/binning_factor \
            - float(root.findall('cameras/camera/fov_video')[i].attrib.get('top'))
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for tag in root.findall('cameras/camera/transform'):
        tx = float(tag.get('x'))/1000
        ty = float(tag.get('y'))/1000
        tz = float(tag.get('z'))/1000
        r11 = float(tag.get('r11'))
        r12 = float(tag.get('r12'))
        r13 = float(tag.get('r13'))
        r21 = float(tag.get('r21'))
        r22 = float(tag.get('r22'))
        r23 = float(tag.get('r23'))
        r31 = float(tag.get('r31'))
        r32 = float(tag.get('r32'))
        r33 = float(tag.get('r33'))

        # Rotation (by-column to by-line)
        R += [np.array([r11, r21, r31, r12, r22, r32, r13, r23, r33]).reshape(3,3)]
        T += [np.array([tx, ty, tz])]
   
    # Cameras names by natural order
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in sorted(C_vid, key=natural_sort_key)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
   
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


def calib_qca_to_toml_func(*args):
    '''
    Convert a Qualisys .qca.txt calibration file 
    to an OpenCV .toml calibration file

    Usage: 
        import calib_qca_to_toml; calib_qca_to_toml.calib_qca_to_toml_func(r'<input_qca_file>')
        OR calib_qca_to_toml -i input_qca_file
        OR calib_qca_to_toml -i input_qca_file --binning_factor 2 -o output_toml_file
    '''
    
    try:
        qca_path = args[0].get('input_file') # invoked with argparse
        binning_factor = int(args[0]['binning_factor'])
        if args[0]['output_file'] == None:
            toml_path = qca_path.replace('.qca.txt', '.toml')
        else:
            toml_path = args[0]['output_file']
    except:
        qca_path = args[0] # invoked as a function
        toml_path = qca_path.replace('.qca.txt', '.toml')
        try:
            binning_factor = int(args[1])
        except:
            binning_factor = 1

    C, S, D, K, R, T = read_qca(qca_path, binning_factor)
    
    RT = [world_to_camera_persp(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)/1000
    
    toml_write(toml_path, C, S, D, K, R, T)

    print('Calibration file generated.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required = True, help='Qualisys .qca.txt input calibration file')
    parser.add_argument('-b', '--binning_factor', required = False, default = 1, help='Binning factor if applied')
    parser.add_argument('-o', '--output_file', required=False, help='OpenCV .toml output calibration file')
    args = vars(parser.parse_args())
    
    calib_qca_to_toml_func(args)
    
