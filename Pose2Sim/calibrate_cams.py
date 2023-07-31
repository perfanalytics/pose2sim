#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## CAMERAS CALIBRATION                                                   ##
    ###########################################################################
    
    Use this module to calibrate your cameras and save results to a .toml file.
    
    It either converts a Qualisys calibration .qca.txt file,
    Or calibrates cameras from checkerboard images.
    
    Checkerboard calibration is based on 
    https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.
    /!\ Beware that corners must be detected on all frames, or else extrinsic 
    parameters may be wrong. Set show_corner_detection to 1 to verify.

    INPUTS: 
    - a calibration file (.qca.txt extension)
    - OR checkerboard images or videos for each camera
    - a Config.toml file
    
    OUTPUTS: 
    - a calibration file (.toml extension)

'''


## INIT
import os
import logging
import numpy as np
import cv2
import glob
import toml
import re
from lxml import etree

from Pose2Sim.common import RT_qca2cv, rotate_cam, quat2mat, euclidean_distance, natural_sort


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.1"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def calib_qca_fun(config):
    '''
    Convert a Qualisys .qca.txt calibration file
    Converts from camera view to object view, Pi rotates cameras, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - a Config.toml file

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats

    '''
    
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    qca_path = glob.glob(os.path.join(calib_dir, '*.qca.txt'))[0]
    binning_factor = config.get('calibration').get('qca').get('binning_factor')
    
    ret, C, S, D, K, R, T = read_qca(qca_path, binning_factor)
    
    RT = [RT_qca2cv(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)
    
    return ret, C, S, D, K, R, T

    
def read_qca(qca_path, binning_factor):
    '''
    Reads a Qualisys .qca.txt calibration file
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - qca_path: path to .qca.txt calibration file: string
    - binning_factor: usually 1: integer

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
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
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left')) +1) /binning_factor
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top')) +1) /binning_factor
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
    C_vid_id = [C_vid.index(c) for c in natural_sort(C_vid)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
   
    return ret, C, S, D, K, R, T


def calib_optitrack_fun(config):
    '''
    Convert an Optitrack calibration file 

    INPUTS:
    - a Config.toml file

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats

    '''

    pass


def calib_vicon_fun(config):
    '''
    Convert a Vicon .xcp calibration file 
    Converts from camera view to object view, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .qca.text file to convert
    - binning_factor: always 1 with Vicon calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats

    '''
   
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    print(calib_dir)
    vicon_path = glob.glob(os.path.join(calib_dir, '*.xcp'))[0]

    ret, C, S, D, K, R, T = read_vicon(vicon_path)
    
    RT = [RT_qca2cv(r,t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)
    
    return ret, C, S, D, K, R, T


def read_vicon(vicon_path):
    '''
    Reads a Vicon .xcp calibration file 
    Returns 6 lists of size N (N=number of cameras)
    
    INPUTS: 
    - vicon_path: path to .xcp calibration file: string

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of 3x3 arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    root = etree.parse(vicon_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    vid_id = []
    
    # Camera name and image size
    for i, tag in enumerate(root.findall('Camera')):
        C += [tag.attrib.get('DEVICEID')]
        S += [[float(t) for t in tag.attrib.get('SENSOR_SIZE').split()]]
        ret += [float(tag.findall('KeyFrames/KeyFrame')[0].attrib.get('WORLD_ERROR'))]
        # if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
        vid_id += [i]

    # Intrinsic parameters: distorsion and intrinsic matrix
    for cam_elem in root.findall('Camera'):
        try:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL2').split()[3:5]
        except:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL').split()
        D += [[float(d) for d in dist] + [0.0, 0.0]]

        fu = float(cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('FOCAL_LENGTH'))
        fv = fu / float(cam_elem.attrib.get('PIXEL_ASPECT_RATIO'))
        cam_center = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('PRINCIPAL_POINT').split()
        cu, cv = [float(c) for c in cam_center]
        K += [np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3,3)]

    # Extrinsic parameters: rotation matrix and translation vector
    for cam_elem in root.findall('Camera'):
        rot = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('ORIENTATION').split()
        R_quat = [float(r) for r in rot]
        R_mat = quat2mat(R_quat, scalar_idx=3)
        R += [R_mat]

        trans = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('POSITION').split()
        T += [[float(t)/1000 for t in trans]]
   
    # Cameras names by natural order
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in natural_sort(C_vid)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
   
    return ret, C, S, D, K, R, T


def findCorners(img, corners_nb, criteria, show):
    '''
    Find corners _of internal squares_ in the checkerboard

    INPUTS:
    - img: image read by opencv
    - corners_nb: [H, W] internal corners in checkerboard: list of two integers [9,6]
    - criteria: when to stop optimizing corners localization
    - show: choose whether to show corner detections

    OUTPUTS
    - imgp:  2d corner points in image plane
    '''

    # Find corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, corners_nb, None)
    # Refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # Draw and display the corners
        if show:
            cv2.drawChessboardCorners(img, corners_nb, imgp, ret)
            print('Corners found.')
            cv2.imshow('img', img)
            cv2.waitKey(0)
        return imgp
    else:
        if show:
            print('Corners not found.')
        return


def calib_checkerboard_fun(config):
    '''
    Calibrates from images or videos of a checkerboard

    INPUTS:
    - a Config.toml file

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    cam_listdirs_names = next(os.walk(calib_dir))[1]
        
    corners_nb = config.get('calibration').get('checkerboard').get('corners_nb')
    square_size = config.get('calibration').get('checkerboard').get('square_size')
    square_size = [square_size, square_size] if isinstance(square_size, int)==True else square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px

    frame_for_origin = config.get('calibration').get('checkerboard').get('frame_for_origin')
    show = config.get('calibration').get('checkerboard').get('show_corner_detection')
    from_vid_or_img = config.get('calibration').get('checkerboard').get('from_vid_or_img')
    vid_snapshot_every_N_frames = config.get('calibration').get('checkerboard').get('vid_snapshot_every_N_frames')
    vid_extension = config.get('calibration').get('checkerboard').get('vid_extension')
    img_extension = config.get('calibration').get('checkerboard').get('img_extension')
 
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    for cam in cam_listdirs_names:
        print(f'\nCamera {cam}:')
         # Prepare object points
        objp = np.zeros((corners_nb[0]*corners_nb[1],3), np.float32) 
        objp[:,:2] = np.mgrid[0:corners_nb[0],0:corners_nb[1]].T.reshape(-1,2)
        objp[:,0] = objp[:,0]*square_size[0]
        objp[:,1] = objp[:,1]*square_size[1]
        objpoints = [] # 3d points in world space
        imgpoints = [] # 2d points in image plane
    
        # Find corners in vid
        if from_vid_or_img=='vid':
            video = glob.glob(os.path.join(calib_dir, cam, '*.'+ vid_extension))[0]
            cap = cv2.VideoCapture(video)
            ret_vid, img = cap.read()
            while ret_vid:
                count = int(round(cap.get(1)))
                ret_vid, img_vid = cap.read()
                if count % vid_snapshot_every_N_frames == 0:
                    img = img_vid
                    imgp = findCorners(img, corners_nb, criteria, show)
                    if isinstance(imgp, np.ndarray):
                        objpoints.append(objp)
                        imgpoints.append(imgp)
            cap.release()

        # Find corners in images
        elif from_vid_or_img=='img':
            images = glob.glob(os.path.join(calib_dir, cam, '*.'+ img_extension))
            images_sorted = sorted(images, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
            for image_f in images_sorted:
                img = cv2.imread(image_f)
                imgp = findCorners(img, corners_nb, criteria, show)
                if isinstance(imgp, np.ndarray):
                    objpoints.append(objp)
                    imgpoints.append(imgp)

        # Calibration
        r, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
                                        None, None, flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT))
        h, w = [np.float32(i) for i in img.shape[:-1]]
        print(r, repr(mtx), repr(dist))
        print(w,h)
        
        ret.append(r)
        C.append(cam)
        S.append([w, h])
        D.append(dist[0])
        K.append(mtx)
        R.append(rvecs[frame_for_origin].squeeze())
        T.append(tvecs[frame_for_origin].squeeze())

    # Object view to camera view
    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]
    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)/1000


    return ret, C, S, D, K, R, T


def toml_write(calib_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''

    with open(os.path.join(calib_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam=f'[cam_{c+1}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]}]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]}], [ 0.0, {K[c][1,1]}, {K[c][1,2]}], [ 0.0, 0.0, 1.0]]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def calibrate(config):
    '''
    Choose whether to use qca or checkerboard calibration functions

    INPUT:
    - a Config.toml file

    OUTPUTS:
    - ret: residual reprojection error in _px_ (checkerboard) or in _mm_ (qca): list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''

    # Map calib function
    calib_type = config.get('calibration').get('type')
    calib_mapping = {
        'qca': calib_qca_fun,
        'vicon': calib_vicon_fun,
        'checkerboard': calib_checkerboard_fun
        }
    calib_fun = calib_mapping[calib_type]

    # Calibrate
    ret, C, S, D, K, R, T = calib_fun(config)
    return ret, C, S, D, K, R, T
    

def recap_calibrate(ret, calib_path, calib_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            fm = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            if calib_type=='qca':
                ret_m.append( np.around(ret[c]*1000, decimals=3) )
                ret_px.append( np.around(ret[c] / Dm * fm, decimals=3) )
            if calib_type=='vicon':
                ret_m.append( np.around(ret[c], decimals=3) )
                ret_px.append( np.around(ret[c] / (Dm*1000) * fm, decimals=3) )
            elif calib_type=='checkerboard':
                ret_px.append( np.around(ret[c], decimals=3) )
                ret_m.append( np.around(ret[c]*1000 * Dm / fm, decimals=3) )

    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, which corresponds to {ret_m} mm.\n')
    logging.info(f'Calibration file is stored at {calib_path}.')


def calibrate_cams_all(config):
    '''
    Either converts qca.txt calibration file, 
    or calibrates from a checkerboard.
    Stores it in a .toml file
    Prints recap.
    
    INPUTS:
    - a Config.toml file

    OUTPUT:
    - a .toml cameras calibration file
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_type = config.get('calibration').get('type')
    if calib_type=='vicon':
        vicon_path = glob.glob(os.path.join(calib_dir, '*.xcp'))[0]
        calib_path = vicon_path.replace('.xcp', '_xcp.toml')
    if calib_type=='qca':
        qca_path = glob.glob(os.path.join(calib_dir, '*.qca.txt'))[0]
        calib_path = qca_path.replace('.qca.txt', '_qca.toml')
    elif calib_type=='checkerboard':
        calib_path = os.path.join(calib_dir, 'Calib_checkerboard.toml')
    
    # Calibrate
    ret, C, S, D, K, R, T = calibrate(config)

    # Write calibration file
    toml_write(calib_path, C, S, D, K, R, T)

    # Recap message
    recap_calibrate(ret, calib_path, calib_type)
