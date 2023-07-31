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
    - a calibration file in the 'calibration' folder (.qca.txt extension)
    - OR folders 'calibration\intrinsics' (populated with video or about 30 images) and 'calibration\extrinsics' (populated with video or one image)
    - a Config.toml file in the 'User' folder
    
    OUTPUTS: 
    - a calibration file in the 'calibration' folder (.toml extension)

'''


## INIT
import os
import logging
import numpy as np
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
import glob
import toml
import re
from lxml import etree
import warnings
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler

from Pose2Sim.common import RT_qca2cv, rotate_cam, quat2mat, euclidean_distance, natural_sort


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.3"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def calib_qca_fun(file_to_convert_path, binning_factor=1):
    '''
    Convert a Qualisys .qca.txt calibration file
    Converts from camera view to object view, Pi rotates cameras, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .qca.text file to convert
    - binning_factor: when filming in 540p, one out of 2 pixels is binned so that the full frame is used

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats

    '''
    
    ret, C, S, D, K, R, T = read_qca(file_to_convert_path, binning_factor)
    
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
        R += [np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3,3).T]
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


def calib_vicon_fun(file_to_convert_path, binning_factor=1):
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
   
    ret, C, S, D, K, R, T = read_vicon(file_to_convert_path)
    
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


def calib_board_fun(calib_dir, intrinsics_config_dict, extrinsics_config_dict):
    '''
    Calibrates intrinsic and extrinsic parameters
    from images or videos of a checkerboard
    or retrieve them from a file

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (intrinsics_board_type, overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)
    - extrinsics_config_dict: dictionary of extrinsics parameters (extrinsics_board_type, calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    # retrieve intrinsics if calib_file found and if overwrite_intrinsics=False
    overwrite_intrinsics = extrinsics_config_dict.get('overwrite_intrinsics')
    try:
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
    except:
        pass
    if not overwrite_intrinsics and 'calib_file' in locals():
        logging.info(f'Preexisting intrinsics file found: {calib_file}. Retrieving intrinsics.')
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]
        calib_data = toml.load(calib_file)
        K, D = [], []
        for cam in calib_data:
            if cam != 'metadata':
                K += calib_data[cam]['matrix']
                D += calib_data[cam]['distortions']
    
    # calculate intrinsics otherwise
    else:
        logging.info(f'Calculating intrinsics.')
        K, D = calibrate_intrinsics(calib_dir, intrinsics_config_dict)

    # calculate extrinsics
    calculate_extrinsic = extrinsics_config_dict.get('calculate_extrinsics')
    if calculate_extrinsic:
        R, T = calibrate_extrinsics(calib_dir, extrinsics_config_dict)


def calibrate_intrinsics(calib_dir, intrinsics_config_dict):
    '''
    Calculate intrinsic parameters
    from images or videos of a checkerboard
    Extract frames, then detect corners, then calibrate

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (intrinsics_board_type, overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)

    OUTPUTS:
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    '''

    try:
        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'intrinsics')))[1]
    except StopIteration:
        raise Exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')
    intrinsics_board_type = intrinsics_config_dict.get('intrinsics_board_type')
    intrinsics_extension = intrinsics_config_dict.get('intrinsics_extension')
    extract_every_N_sec = intrinsics_config_dict.get('extract_every_N_sec')
    overwrite_extraction = False
    show_detection_intrinsics = intrinsics_config_dict.get('show_detection_intrinsics')
    intrinsics_corners_nb = intrinsics_config_dict.get('intrinsics_corners_nb')
    intrinsics_square_size = intrinsics_config_dict.get('intrinsics_square_size')
    C, S, D, K, R, T = [], [], [], [], [], []

    for i,cam in enumerate(intrinsics_cam_listdirs_names):
        # Prepare object points
        objp = np.zeros((intrinsics_corners_nb[0]*intrinsics_corners_nb[1],3), np.float32) 
        objp[:,:2] = np.mgrid[0:intrinsics_corners_nb[0],0:intrinsics_corners_nb[1]].T.reshape(-1,2)
        objp[:,:2] = objp[:,0:2]*intrinsics_square_size
        objpoints = [] # 3d points in world space
        imgpoints = [] # 2d points in image plane

        logging.info(f'\nCamera {cam}:')
        img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.{intrinsics_extension}'))
        if img_vid_files == []:
            raise ValueError(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
        img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
        # extract frames from video if video
        try:
            # check if file is a video rather than an image
            cap = cv2.VideoCapture(img_vid_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise ValueError('No video in the folder or wrong extension.')
            # extract frames from video
            extract_frames(img_vid_files[0], extract_every_N_sec, overwrite_extraction)
            img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.png'))
            img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)])
        except:
            pass

        # find corners
        for img_path in img_vid_files:
            if show_detection_intrinsics == True:
                imgp_confirmed, objp_confirmed = findCorners(img_path, intrinsics_corners_nb, objp=objp, show=show_detection_intrinsics)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp_confirmed)
            else:
                imgp_confirmed = findCorners(img_path, intrinsics_corners_nb, objp=objp, show=show_detection_intrinsics)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp)
        if len(imgpoints) <= 10:
            logging.info(f'Corners were detected only on {len(imgpoints)} images for camera {cam}. Calibration of intrinsic parameters may not be accurate with less than 20 good images of the board.')

        # calculate intrinsics
        img = cv2.imread(str(img_path))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
                                    None, None, flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT))
        h, w = [np.float32(i) for i in img.shape[:-1]]
        C.append(f'cam_{str(i+1).zfill(2)}')   
        S.append([w, h])
        D.append(dist[0])
        K.append(mtx)
        R.append([0,0,0])
        T.append([0,0,0])

    # Write calibration file with calculated intrinsic parameters and random extrinsicc parameters
    calib_file = os.path.join(calib_dir, f'Calib_int{intrinsics_board_type}_extrandom.toml')
    toml_write(calib_file, C, S, D, K, R, T)

    return K, D


def calibrate_extrinsics(calib_dir, extrinsics_config_dict):
    '''
    Calibrates extrinsic parameters
    from an image or the first frame of a video
    of a checkerboard or of measured clues on the scene

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - extrinsics_config_dict: dictionary of extrinsics parameters (extrinsics_board_type, calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''

    try:
        extrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1]
    except StopIteration:
        raise Exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
    extrinsics_extension = extrinsics_config_dict.get('extrinsics_extension')
    extrinsics_board_type = extrinsics_config_dict.get('extrinsics_board_type')
    extrinsics_corners_nb = extrinsics_config_dict.get('extrinsics_corners_nb')
    show_detection_extrinsics = True

    for i,cam in enumerate(extrinsics_cam_listdirs_names):
        logging.info(f'\nCamera {cam}:')
        img_vid_files = glob.glob(os.path.join(calib_dir, 'extrinsics', cam, f'*.{extrinsics_extension}'))
        if img_vid_files == []:
            raise ValueError(f'The folder {os.path.join(calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
        img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers


        if extrinsics_board_type == 'checkerboard':
            # find corners
            imgp = findCorners(img_vid_files[0], extrinsics_corners_nb, objp=[], show_detection_intrinsics=show_detection_intrinsics=)
            # CHANGE FINDCORNERS: 'O' for okay and next, 'D' for delete, 'C' for click
            # DEFINE OBJECT POINTS
            
            # IF CORNERS NOT FOUND: RAISE ERROR: 'set "show_detection_extrinsics" to true to click corners by hand, or change extrinsic_board_type to "scene"'
            
            if isinstance(imgp, np.ndarray):
                objpoints.append(objp)
                imgpoints.append(imgp)
            



        elif extrinsics_board_type == 'scene':
            
            
            if len(objp) <= 10:
                logging.info(f'Only {len(objp)} reference points for camera {cam}. Calibration of extrinsic parameters may not be accurate with less than 10 refeence points.')


    # try:
    #     intrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'intrinsics')))[1]
    # except StopIteration:
    #     raise Exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')
    # intrinsics_board_type = intrinsics_config_dict.get('intrinsics_board_type')
    # intrinsics_extension = intrinsics_config_dict.get('intrinsics_extension')
    # extract_every_N_sec = intrinsics_config_dict.get('extract_every_N_sec')
    # overwrite_extraction=False
    # show_detection_intrinsics = intrinsics_config_dict.get('show_detection_intrinsics')
    # intrinsics_corners_nb = intrinsics_config_dict.get('intrinsics_corners_nb')
    # intrinsics_square_size = intrinsics_config_dict.get('intrinsics_square_size')
    # C, S, D, K, R, T = [], [], [], [], [], []

    # for i,cam in enumerate(intrinsics_cam_listdirs_names):
    #     # Prepare object points
    #     objp = np.zeros((intrinsics_corners_nb[0]*intrinsics_corners_nb[1],3), np.float32) 
    #     objp[:,:2] = np.mgrid[0:intrinsics_corners_nb[0],0:intrinsics_corners_nb[1]].T.reshape(-1,2)
    #     objp[:,:2] = objp[:,0:2]*intrinsics_square_size
    #     objpoints = [] # 3d points in world space
    #     imgpoints = [] # 2d points in image plane

    #     print(f'\nCamera {cam}:')
    #     img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.{intrinsics_extension}'))
    #     if img_vid_files == []:
    #         raise ValueError(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
    #     img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)]) #sorting paths with numbers
    #     # extract frames from video if video
    #     try:
    #         # check if file is a video rather than an image
    #         cap = cv2.VideoCapture(img_vid_files[0])
    #         cap.read()
    #         if cap.read()[0] == False:
    #             raise ValueError('No video in the folder or wrong extension.')
    #         ## extract frames from video
    #         extract_frames(img_vid_files[0], extract_every_N_sec, overwrite_extraction)
    #         img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.png'))
    #         img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)])
    #     except:
    #         pass

    #     # find corners
    #     for img_path in img_vid_files:
    #         imgp = findCorners(img_path, intrinsics_corners_nb, show_detection_intrinsics)
    #         if isinstance(imgp, np.ndarray):
    #             objpoints.append(objp)
    #             imgpoints.append(imgp)
    #     if len(imgpoints) <= 10:
    #         print(f'Corners were detected only on {len(imgpoints)} images for camera {cam}. Calibration of intrinsic parameters may not be accurate with less than 20 good images of the board.')

    #     # calculate intrinsics
    #     img = cv2.imread(str(img_path))
    #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
    #                                 None, None, flags=(cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_PRINCIPAL_POINT))
    #     h, w = [np.float32(i) for i in img.shape[:-1]]
    #     C.append(f'cam_{str(i+1).zfill(2)}')   
    #     S.append([w, h])
    #     D.append(dist[0])
    #     K.append(mtx)
    #     R.append([0,0,0])
    #     T.append([0,0,0])

    # # Write calibration file with calculated intrinsic parameters and random extrinsicc parameters
    # calib_file = os.path.join(calib_dir, f'Calib_int{intrinsics_board_type}_extrandom.toml')
    # toml_write(calib_file, C, S, D, K, R, T)

    # return K, D




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
        # Try videocapture
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
        print(ret, repr(mtx), repr(dist))
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


def extract_frames(video_path, extract_every_N_sec=1, overwrite_extraction=False):
    '''
    Extract frames if don't exist yet or if overwrite==True
    
    INPUT:
    - video_path: path to video whose frames need to be extracted
    - extract_every_N_sec: extract one frame every N seconds (can be <1)
    - overwrite_extraction: if True, overwrite even if frames have already been extracted
    
    OUTPUT:
    - extracted frames in folder
    '''
    
    if not os.path.exists(os.path.splitext(video_path)[0] + '_00000.png') or overwrite_extraction:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            frame_nb = 0
            logging.info(f'Extracting frames...')
            while cap.isOpened():
                ret, frame = cap.read()
                if ret == True:
                    if frame_nb % (fps*extract_every_N_sec) == 0:
                        img_path = (os.path.splitext(video_path)[0] + '_' +str(frame_nb).zfill(5)+'.png')
                        cv2.imwrite(str(img_path), frame)
                    frame_nb+=1
                else:
                    break


def findCorners(img_path, corner_nb, objp=[], show=True):
    '''
    Find corners in the photo of a checkerboard.
    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.
    
    Make sure that: 
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners
    
    INPUTS:
    - img_path: path to image (or video)
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [9,6]
    - optionnal: show: choose whether to show corner detections
    - optionnal: objp: array [3d corner coordinates]

    OUTPUTS:
    - imgp_confirmed: array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
    # reads image if image, or first frame if video
    try:
        img = cv2.imread(img_path)
    except:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cap = cv2.VideoCapture(img_path)
            ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)
    # If corners are found, refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')
        
        if show:
            # Draw corners
            cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
            # Add corner index 
            for i, corner in enumerate(imgp):
                x, y = corner.ravel()
                cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2) 
                cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1) 
            
            # Visualizer and key press event handler
            for var_to_delete in ['imgp_confirmed', 'objp_confirmed']:
                if var_to_delete in globals():
                    del globals()[var_to_delete]
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=imgp, objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = imgp

    # If corners are not found, dismiss or click points by hand
    else:
        logging.info(f'{os.path.basename(img_path)}: Corners not found.')
        if show:
            # Visualizer and key press event handler
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=[], objp=objp, img_path=img_path)
        else:
            imgp_objp_confirmed = []
            
    # if len(imgp_objp_confirmed) == 1:
    #     imgp_confirmed = imgp_objp_confirmed
    #     objp_confirmed = objp
    # else:
    #     imgp_confirmed, objp_confirmed = imgp_objp_confirmed
    return imgp_objp_confirmed


def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=''):
    '''
    Shows image img. 
    If imgp is given, displays them in green
    If objp is given, can be displayed in a 3D plot if 'C' is pressed.
    If img_path is given, just uses it to name the window

    If 'Y' is pressed, closes all and returns confirmed imgp and (if given) objp
    If 'N' is pressed, closes all and returns nothing
    If 'C' is pressed, allows clicking imgp by hand. If objp is given:
        Displays them in 3D as a helper. 
        Left click to add a point, right click to remove the last point.
        Press 'H' to indicate that one of the objp is not visible on image
        Closes all and returns imgp and objp if all points have been clicked
    Allows for zooming and panning with middle click
    
    INPUTS:
    - img: image opened with openCV
    - optional: imgp: detected image points, to be accepted or not. Array of [[2d corner coordinates]]
    - optionnal: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    def on_key(event):
        '''
        Handles key press events:
        'Y' to return imgp, 'N' to dismiss image, 'C' to click points by hand.
        Left click to add a point, 'H' to indicate it is not visible, right click to remove the last point.
        '''

        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count
        
        if event.key == 'y':
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if 'scat' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = [imgp.astype('float32') for imgp in imgp_confirmed]
                objp_confirmed = np.array(objp_confirmed)
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            if len(objp) == 0:
                objp_confirmed = []

        if event.key == 'c':
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([],[],s=100,marker='+',color='g')
            plt.connect('button_press_event', on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title('Object points to be clicked')
                ax_3d = fig_3d.add_subplot(projection='3d')
                plt.rc('xtick', labelsize=5)
                plt.rc('ytick', labelsize=5)
                for i, (xs,ys,zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs,ys,zs, marker='.', color='k')
                    ax_3d.text(xs,ys,zs,  f'{str(i+1)}', size=10, zorder=1, color='k') 
                set_axes_equal(ax_3d)
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=90, azim=-90)
                fig_3d.show()

        if event.key == 'h':
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0  and 'ax_3d' in globals():
                count = [0 if 'count' not in globals() else count+1][0]
                if 'events' not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # if all objp have been clicked or indicated as not visible, close all
                    imgp_confirmed = [imgp.astype('float32') for imgp in imgp_confirmed]
                    plt.close('all')
                    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
            else:
                pass


    def on_click(event):
        '''
        Detect click position on image
        If right click, last point is removed
        '''
        
        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count, xydata
        
        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1: 
            # To remember the event to cancel after right click
            if 'events' in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata,[[event.xdata,event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)    
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if 'count' not in globals() else count+1][0]
                if count==0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # retrieve objp_confirmed
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    imgp_confirmed = [imgp.astype('float32') for imgp in imgp_confirmed]
                    # close all, delete all
                    plt.close('all')
                    for var_to_delete in ['events', 'count', 'scat', 'scat_3d', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()

        # Right click: 
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3: # right click
            # If last event was left click: 
            if 'button' in dir(events[-1]):
                if events[-1].button == 1: 
                    # Remove lastpoint from image
                    new_xydata = scat.get_offsets()[:-1]
                    scat.set_offsets(new_xydata)
                    plt.draw()
                    # Remove last point from imgp_confirmed
                    imgp_confirmed = imgp_confirmed[:-1]
                    if len(objp) != 0:
                        if count >= 1: count -= 1
                        # Remove last point from objp_confirmed
                        objp_confirmed = objp_confirmed[:-1]
                        # remove from plot 
                        if len(ax_3d.collections) > len(objp):
                            ax_3d.collections[-1].remove()
                            fig_3d.canvas.draw()
                        
            # If last event was 'h' key
            elif events[-1].key == 'h':
                if len(objp) != 0:
                    if count >= 1: count -= 1
                    # Remove last point from objp_confirmed_notok
                    objp_confirmed_notok = objp_confirmed_notok[:-1]
                    # remove from plot  
                    if len(ax_3d.collections) > len(objp):
                        ax_3d.collections[-1].remove()
                        fig_3d.canvas.draw()                
    

    def set_axes_equal(ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 2, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, lineType = cv2.LINE_AA)    
    
    # Put image in a matplotlib figure for more controls
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x),int(y)), (128,128,128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    
    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['toolbar'] = 'toolmanager'

    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if 'imgp_confirmed' in globals() and 'objp_confirmed' in globals():
        return imgp_confirmed, objp_confirmed
    elif 'imgp_confirmed' in globals() and not 'objp_confirmed' in globals():
        return imgp_confirmed
    else:
        return


def calib_points_fun(config):
    '''
    Not implemented yet
    '''
    pass


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


def recap_calibrate(ret, calib_path, calib_full_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            f_px = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            if calib_full_type=='convert_qualisys' or calib_full_type=='convert_vicon':
                ret_m.append( np.around(ret[c], decimals=3) )
                ret_px.append( np.around(ret[c] / (Dm*1000) * f_px, decimals=3) )
            elif calib_full_type=='calculate_board':
                ret_px.append( np.around(ret[c], decimals=3) )
                ret_m.append( np.around(ret[c]*(Dm*1000) / f_px, decimals=3) )

    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, \nwhich corresponds to {ret_m} mm.\n')
    logging.info(f'Calibration file is stored at {calib_path}.')


def calibrate_cams_all(config):
    '''
    Either converts a preexisting calibration file, 
    or calculates calibration from scratch (from a board or from points).
    Stores calibration in a .toml file
    Prints recap.
    
    INPUTS:
    - a Config.toml file

    OUTPUT:
    - a .toml camera calibration file
    '''

    # Read config
    project_dir = config.get('project').get('project_dir')
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_type = config.get('calibration').get('calibration_type')

    if calib_type=='convert':
        convert_filetype = config.get('calibration').get('convert').get('convert_from')
        if convert_filetype=='qualisys':
            convert_ext = '.qca.txt'
            binning_factor = config.get('calibration').get('convert').get('qualisys').get('binning_factor')
        elif convert_filetype=='optitrack':
            print('See Readme.md to retrieve calibration values.')
        elif convert_filetype=='vicon':
            convert_ext = '.xcp'
            binning_factor = 1
            print('Not supported yet')

        file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}*'))[0]
        calib_output_path = os.path.join(calib_dir, f'Calib_{convert_filetype}.toml')

        calib_full_type = '_'.join([calib_type, convert_filetype])
        args_calib_fun = [file_to_convert_path, binning_factor]
        
    elif calib_type=='calculate':
        calculate_method = config.get('calibration').get('calculate').get('calculate_method')
        if calculate_method=='board':
            intrinsics_board_type = config.get('calibration').get('calculate').get('board').get('intrinsics').get('intrinsics_board_type')
            intrinsics_config_dict = config.get('calibration').get('calculate').get('board').get('intrinsics')
            extrinsics_board_type = config.get('calibration').get('calculate').get('board').get('extrinsics').get('extrinsics_board_type')
            extrinsics_config_dict = config.get('calibration').get('calculate').get('board').get('extrinsics')
            calib_output_path = os.path.join(calib_dir, f'Calib_int{intrinsics_board_type}_ext{extrinsics_board_type}.toml')
            
            calib_full_type = '_'.join([calib_type, calculate_method])
            args_calib_fun = [calib_dir, intrinsics_config_dict, extrinsics_config_dict]

        elif calculate_method=='points':
            print('Not supported yet')

        else:
            print('Wrong calculate_method in Config.toml')

    else:
        print('Wrong calibration_type in Config.toml')
    
    # Map calib function
    calib_mapping = {
        'convert_qualisys': calib_qca_fun,
        'convert_optitrack': calib_optitrack_fun,
        'convert_vicon': calib_vicon_fun,
        'calculate_board': calib_board_fun,
        'calculate_points': calib_points_fun
        }
    calib_fun = calib_mapping[calib_full_type]

    # Calibrate
    ret, C, S, D, K, R, T = calib_fun(*args_calib_fun)

    # Write calibration file
    toml_write(calib_output_path, C, S, D, K, R, T)

    # Recap message
    recap_calibrate(ret, calib_output_path, calib_full_type)
