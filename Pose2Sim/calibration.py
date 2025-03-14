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

INPUTS: 
- a calibration file in the 'calibration' folder (.qca.txt extension)
- OR folders 'calibration/intrinsics' (populated with video or about 30 images) and 'calibration/extrinsics' (populated with video or one image)
- a Config.toml file in the 'User' folder

OUTPUTS: 
- a calibration file in the 'calibration' folder (.toml extension)
'''

# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS


## INIT
from Pose2Sim.common import world_to_camera_persp, rotate_cam, quat2mat, euclidean_distance, zup2yup

import os
import toml
import logging
import pickle
import numpy as np
import pandas as pd
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
from lxml import etree
import warnings
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from PIL import Image


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
def calib_qca_fun(config, file_to_convert_path, binning_factor=1):
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
    
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    root = etree.parse(file_to_convert_path).getroot()

    camera_tags = root.findall('cameras/camera')
    fov_tags = root.findall('cameras/camera/fov_video')
    intrinsic_tags = root.findall('cameras/camera/intrinsic')
    transform_tags = root.findall('cameras/camera/transform')
    
    # Camera name
    for i, tag in enumerate(camera_tags):
        name = tag.attrib.get('serial')

        source = next((s for s in config.sources if s.name == name), None)
        if source is None:
            logging.warning(f"No source found for config: '{name}'.")
            continue

        source.ret = float(tag.attrib.get('avg-residual'))

        video_res = tag.attrib.get('video_resolution')

        res_value = int(video_res[:-1]) if video_res is not None else 1080

        # Image size
        fov_tag = fov_tags[i]
        w = (float(fov_tag.attrib.get('right')) - float(fov_tag.attrib.get('left')) + 1) / binning_factor / (1080 / res_value)
        h = (float(fov_tag.attrib.get('bottom')) - float(fov_tag.attrib.get('top')) + 1) / binning_factor / (1080 / res_value)
        source.S = np.array([w, h], dtype=np.float32)

        # Intrinsic parameters: distortion and intrinsic matrix
        fu = float(intrinsic_tag.get('focalLengthU')) / 64 / binning_factor / (1080 / res_value)
        fv = float(intrinsic_tag.get('focalLengthV')) / 64 / binning_factor / (1080 / res_value)
        left_val = float(fov_tag.attrib.get('left'))
        top_val = float(fov_tag.attrib.get('top'))
        cu = (float(intrinsic_tag.get('centerPointU')) / 64 / binning_factor - left_val) / (1080 / res_value)
        cv = (float(intrinsic_tag.get('centerPointV')) / 64 / binning_factor - top_val) / (1080 / res_value)
        source.K = np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.], dtype=np.float32).reshape(3, 3)

        intrinsic_tag = intrinsic_tags[i]
        k1 = float(intrinsic_tag.get('radialDistortion1')) / 64 / binning_factor
        k2 = float(intrinsic_tag.get('radialDistortion2')) / 64 / binning_factor
        p1 = float(intrinsic_tag.get('tangentalDistortion1')) / 64 / binning_factor
        p2 = float(intrinsic_tag.get('tangentalDistortion2')) / 64 / binning_factor
        source.D = np.array([k1, k2, p1, p2], dtype=np.float32)

        # Extrinsic parameters: rotation matrix and translation vector
        r11 = float(transform_tag.get('r11'))
        r12 = float(transform_tag.get('r12'))
        r13 = float(transform_tag.get('r13'))
        r21 = float(transform_tag.get('r21'))
        r22 = float(transform_tag.get('r22'))
        r23 = float(transform_tag.get('r23'))
        r31 = float(transform_tag.get('r31'))
        r32 = float(transform_tag.get('r32'))
        r33 = float(transform_tag.get('r33'))
        R_value = np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33], dtype=np.float32).reshape(3, 3).T

        transform_tag = transform_tags[i]
        tx = float(transform_tag.get('x')) / 1000
        ty = float(transform_tag.get('y')) / 1000
        tz = float(transform_tag.get('z')) / 1000
        T_value = np.array([tx, ty, tz], dtype=np.float32)

        r_transf, t_transf = world_to_camera_persp(R_value, T_value)
        r_transf, t_transf = rotate_cam(r_transf, t_transf, ang_x=np.pi, ang_y=0, ang_z=0)
        source.R = cv2.Rodrigues(r_transf.astype(np.float32))[0].flatten()
        source.T = t_transf.astype(np.float32)


def calib_optitrack_fun(config, file_to_convert_path, binning_factor=1):
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
    
    logging.warning('See Readme.md to retrieve Optitrack calibration values.')
    raise NameError("See Readme.md to retrieve Optitrack calibration values.")


def calib_vicon_fun(config, file_to_convert_path, binning_factor=1):
    '''
    Convert a Vicon .xcp calibration file 
    Converts from camera view to object view, 
    and converts rotation with Rodrigues formula

    INPUTS:
    - file_to_convert_path: path of the .xcp file to convert
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
   
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    root = etree.parse(file_to_convert_path).getroot()

    for cam_elem in root.findall('Camera'):
        # Camera name and image size
        name = cam_elem.attrib.get('DEVICEID')
        source = next((s for s in config.sources if s.name == name), None)
        if source is None:
            logging.warning(f"No source found for config: '{name}'.")
            continue
        
        keyframe = cam_elem.findall('KeyFrames/KeyFrame')[0]
        
        source.ret = float(keyframe.attrib.get('WORLD_ERROR'))
        
        source.S = [float(t) for t in cam_elem.attrib.get('SENSOR_SIZE').split()]
        
        # Intrinsic parameters: distorsion and intrinsic matrix
        fu = float(keyframe.attrib.get('FOCAL_LENGTH'))
        pixel_aspect = float(cam_elem.attrib.get('PIXEL_ASPECT_RATIO'))
        fv = fu / pixel_aspect
        cam_center = keyframe.attrib.get('PRINCIPAL_POINT').split()
        cu, cv = [float(c) for c in cam_center]
        source.K = np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3, 3)

        try:
            dist = keyframe.attrib.get('VICON_RADIAL2').split()[3:5]
        except Exception:
            dist = keyframe.attrib.get('VICON_RADIAL').split()
        source.D = [float(d) for d in dist] + [0.0, 0.0]
        
        # Extrinsic parameters: rotation matrix and translation vector
        rot = keyframe.attrib.get('ORIENTATION').split()
        R_quat = [float(r) for r in rot]
        r_orig = quat2mat(R_quat, scalar_idx=3)

        t_orig = [float(t) for t in keyframe.attrib.get('POSITION').split()]
        
        r_trans, t_trans = world_to_camera_persp(r_orig, t_orig)
        
        source.R = np.array(cv2.Rodrigues(r_trans)[0]).flatten()
        source.T = np.array(t_trans)


def read_intrinsic_yml(config, intrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - S (image size)
    - K (intrinsic parameters)
    - D (distorsion)

    N.B. : Size is calculated as twice the position of the optical center. Please correct in the .toml file if needed.
    '''
    intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = intrinsic_yml.getNode('names').size()

    for i in range(cam_number):
        name = intrinsic_yml.getNode('names').at(i).string()
        source = next((s for s in config.sources if s.name == name), None)
        if source is None:
            logging.warning(f"No source found for config: '{name}'.")
            continue

    K_mat = intrinsic_yml.getNode(f'K_{name}').mat().astype(float)
    source.S = [float(K_mat[0, 2]) * 2, float(K_mat[1, 2]) * 2]
    source.K = K_mat
    source.D = intrinsic_yml.getNode(f'dist_{name}').mat().astype(float).flatten()[:-1]


def read_extrinsic_yml(config, extrinsic_path):
    '''
    Reads an intrinsic .yml calibration file
    Returns 3 lists of size N (N=number of cameras):
    - R (extrinsic rotation, Rodrigues vector)
    - T (extrinsic translation)
    '''
    extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)
    cam_number = extrinsic_yml.getNode('names').size()

    for i in range(cam_number):
        name = extrinsic_yml.getNode('names').at(i).string()
        
        source = next((s for s in config.sources if s.name == name), None)
        if source is None:
            logging.warning(f"No source found for config: '{name}'.")
            continue

        source.R = extrinsic_yml.getNode(f'R_{name}').mat().astype(float).flatten()
        source.T = extrinsic_yml.getNode(f'T_{name}').mat().astype(float).flatten()
        source.ret = np.nan


def calib_easymocap_fun(config, files_to_convert_paths, binning_factor=1):
    '''
    Reads EasyMocap .yml calibration files

    INPUTS:
    - files_to_convert_paths: paths of the intri.yml and extri.yml calibration files to convert
    - binning_factor: always 1 with easymocap calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''

    extrinsic_path, intrinsic_path = files_to_convert_paths
    read_intrinsic_yml(config, intrinsic_path)
    read_extrinsic_yml(config, extrinsic_path)


def calib_biocv_fun(config, files_to_convert_paths, binning_factor=1):
    '''
    Convert bioCV calibration files.

    INPUTS:
    - files_to_convert_paths: paths of the calibration files to convert (no extension)
    - binning_factor: always 1 with biocv calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')

    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split('\n')
            
            name = f'cam_{str(i).zfill(2)}'

            source = next((s for s in config.sources if s.name == name), None)
            if source is None:
                logging.warning(f".")
                continue

            source.S = [float(calib_data[0]), float(calib_data[1])]
            source.K = np.array([list(map(float, k.strip().split(' '))) for k in calib_data[2:5]], dtype=np.float32)
            source.D = np.array([float(d) for d in calib_data[-2].split(' ')[:4]], dtype=np.float32)

            RT = np.array([list(map(float, k.strip().split(' '))) for k in calib_data[6:9]], dtype=np.float32)
            source.R = cv2.Rodrigues(RT[:, :3])[0].squeeze().astype(np.float32)
            source.T = (RT[:, 3] / 1000).astype(np.float32)



def calib_opencap_fun(config, files_to_convert_paths, binning_factor=1):
    '''
    Convert OpenCap calibration files.
    
    Extrinsics in OpenCap are calculated with a vertical board for the world frame.
    As we want the world frame to be horizontal, we need to rotate cameras by -Pi/2 around x in the world frame. 
    T is good the way it is.

    INPUTS:
    - files_to_convert_paths: paths of the .pickle calibration files to convert
    - binning_factor: always 1 with opencap calibration

    OUTPUTS:
    - ret: residual reprojection error in _mm_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats
    - T: extrinsic translation: list of arrays of floats
    '''
    
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')
    
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path, 'rb') as f_pickle:
            calib_data = pickle.load(f_pickle)
            name = f'cam_{str(i).zfill(2)}'

            source = next((s for s in config.sources if s.name == name), None)
            if source is None:
                logging.warning(f"No source found for config: '{name}'.")
                continue

            source.ret = np.nan
            source.S = list(map(float, calib_data['imageSize'].squeeze()[::-1]))
            source.D = list(map(float, calib_data['distortion'][0][:-1]))
            source.K = calib_data['intrinsicMat'].astype(np.float32)

            R_cam = calib_data['rotation'].astype(np.float32)
            T_cam = calib_data['translation'].squeeze().astype(np.float32)

            # Rotate cameras by Pi/2 around x in world frame -> could have just switched some columns in matrix
            # camera frame to world frame
            R_w, T_w = world_to_camera_persp(R_cam, T_cam)
            # x_rotate -Pi/2 and z_rotate Pi
            R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi)
            # world frame to camera frame
            R_c_90, T_c_90 = world_to_camera_persp(R_w_90, T_w_90)

            # store in R and T
            source.R = cv2.Rodrigues(R_c_90.astype(np.float32))[0].squeeze()
            source.T = (T_cam / 1000).astype(np.float32)


def calib_calc_fun(config):
    '''
    Calibrates intrinsic and extrinsic parameters
    from images or videos of a checkerboard
    or retrieve them from a file

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)
    - extrinsics_config_dict: dictionary of extrinsics parameters (calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''

    # retrieve intrinsics if calib_file found and if overwrite_intrinsics=False
    if config.use_existing_intrinsics:
        config.calibrate_sources()
    
    # calculate intrinsics otherwise
    else:
        logging.info(f'\nCalculating intrinsic parameters...')
        calibrate_intrinsics(config)

    # calculate extrinsics
    if config.calculate_extrinsics:
        logging.info(f'\nCalculating extrinsic parameters...')
        calibrate_extrinsics(config)
    else:
        logging.info(f'\nExtrinsic parameters won\'t be calculated. Set "calculate_extrinsics" to true in Config.toml to calculate them.')


def calibrate_intrinsics(config):
    '''
    Calculate intrinsic parameters
    from images or videos of a checkerboard
    Extract frames, then detect corners, then calibrate

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters (overwrite_intrinsics, show_detection_intrinsics, intrinsics_extension, extract_every_N_sec, intrinsics_corners_nb, intrinsics_square_size, intrinsics_marker_size, intrinsics_aruco_dict)

    OUTPUTS:
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    '''

    for source in config.sources:
        # Prepare object points
        objp = np.zeros((config.intrinsics_corners_nb[0]*config.intrinsics_corners_nb[1],3), np.float32) 
        objp[:,:2] = np.mgrid[0:config.intrinsics_corners_nb[0],0:config.intrinsics_corners_nb[1]].T.reshape(-1,2)
        objp[:,:2] = objp[:,0:2]*config.intrinsics_square_size
        objpoints = [] # 3d points in world space
        imgpoints = [] # 2d points in image plane

        logging.info(f'\nCamera {source.name}:')

        # extract frames from video if video
        source.extract_frames('intrinsic')

        # find corners
        for img_path in source.intrinsic_files[0]:
            imgp_confirmed, objp_confirmed = findCorners(img_path, config.intrinsics_corners_nb, objp=objp, show=config.show_detection_intrinsics)
            if config.show_detection_intrinsics == True:
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp_confirmed)
            else:
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp)
        if len(imgpoints) < 10:
            logging.info(f'Corners were detected only on {len(imgpoints)} images for camera {source.name}. Calibration of intrinsic parameters may not be accurate with fewer than 10 good images of the board.')

        # calculate intrinsics
        img = cv2.imread(str(img_path))
        objpoints = np.array(objpoints)
        ret_cam, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
                                    None, None, flags=(cv2.CALIB_FIX_K3+cv2.CALIB_USE_LU))# + cv2.CALIB_FIX_PRINCIPAL_POINT))
        h, w = [np.float32(i) for i in img.shape[:-1]]
        source.ret = ret_cam
        source.S = [float(w), float(h)]
        source.K = np.array(dist[0], dtype=np.float32)
        source.D = np.array(mtx, dtype=np.float32)

        logging.info(f'Intrinsics error: {np.around(ret_cam, decimals=3)} px for each cameras.')


def calibrate_extrinsics(config):
    '''
    Calibrates extrinsic parameters
    from an image or the first frame of a video
    of a checkerboard or of measured clues on the scene

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - extrinsics_config_dict: dictionary of extrinsics parameters (extrinsics_method, calculate_extrinsics, show_detection_extrinsics, extrinsics_extension, extrinsics_corners_nb, extrinsics_square_size, extrinsics_marker_size, extrinsics_aruco_dict, object_coords_3d)

    OUTPUTS:
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    if config.extrinsics_method in {'board', 'scene'}:

        # Define 3D object points
        if config.extrinsics_method == 'board':
            extrinsics_corners_nb = config.extrinsics_square_size
            extrinsics_square_size = config.extrinsics_corners_nb
            object_coords_3d = np.zeros((extrinsics_corners_nb[0] * extrinsics_corners_nb[1], 3), np.float32)
            object_coords_3d[:, :2] = np.mgrid[0:extrinsics_corners_nb[0], 0:extrinsics_corners_nb[1]].T.reshape(-1, 2)
            object_coords_3d[:, :2] = object_coords_3d[:, 0:2] * extrinsics_square_size
        elif config.extrinsics_method == 'scene':
            object_coords_3d = config.object_coords_3d

        # Save reference 3D coordinates as trc
        trc_write(object_coords_3d, config.calib_output_path)

        for source in config.sources:
            logging.info(f'\nCamera {source.name}:')
            
            # Read images or video
            source.extract_frames('extrinsic')

            # Find corners or label by hand
            if config.extrinsics_method == 'board':
                imgp, objp = findCorners(source.extrinsic_files[0], extrinsics_corners_nb, objp=object_coords_3d, show=config.show_reprojection_error)
                if len(imgp) == 0:
                    logging.exception('No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsic_board_type to "scene"')
                    raise ValueError('No corners found. Set "show_detection_extrinsics" to true to click corners by hand, or change extrinsic_board_type to "scene"')

            elif config.extrinsics_method == 'scene':
                imgp, objp = imgp_objp_visualizer_clicker(source.extrinsic_files[0], imgp, objp=object_coords_3d)
                if len(imgp) == 0:
                    logging.exception('No points clicked (or fewer than 6). Press \'C\' when the image is displayed, and then click on the image points corresponding to the \'object_coords_3d\' you measured and wrote down in the Config.toml file.')
                    raise ValueError('No points clicked (or fewer than 6). Press \'C\' when the image is displayed, and then click on the image points corresponding to the \'object_coords_3d\' you measured and wrote down in the Config.toml file.')
                if len(objp) < 10:
                    logging.info(f'Only {len(objp)} reference points for camera {source.name}. Calibration of extrinsic parameters may not be accurate with fewer than 10 reference points, as spread out in the captured volume as possible.')

            # Calculate extrinsics
            mtx, dist = np.array(source.K), np.array(source.D)
            _, r, t = cv2.solvePnP(np.array(objp, dtype=np.float32) * 1000, imgp, np.array(mtx, dtype=np.float32), np.array(dist, dtype=np.float32))

            source.R = r.flatten().astype(np.float32)
            source.T = (t.flatten() / 1000).astype(np.float32)

            # Projection of object points to image plane
            # # Former way, distortions used to be ignored
            # Kh_cam = np.block([mtx, np.zeros(3).reshape(3,1)])
            # r_mat, _ = cv2.Rodrigues(r)
            # H_cam = np.block([[r_mat,t.reshape(3,1)], [np.zeros(3), 1 ]])
            # P_cam = Kh_cam @ H_cam
            # proj_obj = [ ( P_cam[0] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)),  P_cam[1] @ np.append(o, 1) /  (P_cam[2] @ np.append(o, 1)) ) for o in objp]
            proj_obj = np.squeeze(cv2.projectPoints(objp,r,t,mtx,dist)[0])

            # Check calibration results
            if config.show_reprojection_error:
                # Reopen image, otherwise 2 sets of text are overlaid
                img = cv2.imread(source.extrinsic_files[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                for o in proj_obj:
                    cv2.circle(img, (int(o[0]), int(o[1])), 8, (0,0,255), -1) 
                for i in imgp:
                    cv2.drawMarker(img, (int(i[0][0]), int(i[0][1])), (0,255,0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, 'Verify calibration results, then close window.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA) 
                cv2.drawMarker(img, (20,40), (0,255,0), cv2.MARKER_CROSS, 15, 2)
                cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, '    Clicked points', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
                cv2.circle(img, (20,60), 8, (0,0,255), -1)    
                cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
                cv2.putText(img, '    Reprojected object points', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
                im_pil = Image.fromarray(img)
                im_pil.show(title = os.path.basename(source.extrinsic_files[0]))

            # Calculate reprojection error
            imgp_to_objreproj_dist = [euclidean_distance(proj_obj[n], imgp[n]) for n in range(len(proj_obj))]
            rms_px = np.sqrt(np.sum([d**2 for d in imgp_to_objreproj_dist]))
            source.ret = rms_px

    elif config.extrinsics_method == 'keypoints':
        raise NotImplementedError('This has not been integrated yet.')

    else:
        raise ValueError('Wrong value for extrinsics_method')


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
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [4,7]
    - optional: show: choose whether to show corner detections
    - optional: objp: array [3d corner coordinates]

    OUTPUTS:
    - imgp_confirmed: array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)
    # If corners are found, refine corners
    if ret == True: 
        imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')

        if show:
            # Draw corners
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.drawChessboardCorners(img, corner_nb, imgp, ret)
            # Add corner index 
            for i, corner in enumerate(imgp):
                if i in [0, corner_nb[0]-1, corner_nb[0]*(corner_nb[1]-1), corner_nb[0]*corner_nb[1] -1]:
                    x, y = corner.ravel()
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 7) 
                    cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2) 
            
            return imgp_objp_visualizer_clicker(img_path, imgp=imgp, objp=objp)
        else:
            return imgp
            

    # If corners are not found, dismiss or click points by hand
    else:
        if show:
            # Visualizer and key press event handler
            logging.info(f'{os.path.basename(img_path)}: Corners not found: please label them by hand.')
            return imgp_objp_visualizer_clicker(img_path, imgp=[], objp=objp)
        else:
            logging.info(f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in the Config.toml file.')
            return []


## TODO: NEED A REWORK !
def imgp_objp_visualizer_clicker(img_path, imgp=[], objp=[]):
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
    - optional: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''                
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
            if 'scat' not in globals() or 'imgp_confirmed' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array([imgp.astype('float32') for imgp in imgp_confirmed])
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == 'c':
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            ax.imshow(img)
            # To update the image
            plt.draw()

            if 'objp_confirmed' in globals():
                del objp_confirmed
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
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=-90, azim=0)
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
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])[:-1]
                    imgp_confirmed = np.array(np.expand_dims(scat.get_offsets(), axis=1), np.float32) 
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
                    # close all
                    plt.close('all')
                    # retrieve objp_confirmed
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])
                    imgp_confirmed = np.array(imgp_confirmed, np.float32)
                    # delete all
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
            if 'events' in globals():
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
                            if count >= 0: 
                                count -= 1
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
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

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
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    
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


def trc_write(object_coords_3d, trc_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - object_coords_3d: list of 3D point lists
    - trc_path: output path of the trc file

    OUTPUT:
    - trc file with 2 frames of the same 3D points
    '''

    #Header
    DataRate = CameraRate = OrigDataRate = 1
    NumFrames = 2
    NumMarkers = len(object_coords_3d)
    keypoints_names = np.arange(NumMarkers)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(str(k) for k in keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(NumMarkers)])]
    
    # Zup to Yup coordinate system
    object_coords_3d = pd.DataFrame([np.array(object_coords_3d).flatten(), np.array(object_coords_3d).flatten()])
    object_coords_3d = zup2yup(object_coords_3d)
    
    #Add Frame# and Time columns
    object_coords_3d.index = np.array(range(0, NumFrames)) + 1
    object_coords_3d.insert(0, 't', object_coords_3d.index / DataRate)

    #Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        object_coords_3d.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def toml_write(config, calib_output_path):
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

    if config.calib_file != None:
        data = toml.load(calib_output_path)
    else:
        data = {}

    for source in config.sources:
        if len(source.K) == 0 and len(source.D) == 0 and len(source.R) == 0 and len(source.T) == 0 and len(source.S) == 0:
            logging.warning(f"The source '{source.name}' has not been calibrated")
    
        else:
            data[source.name] = {
                "name": source.name,
                "size": [float(source.S[0]), float(source.S[1])],
                "matrix": [
                    [float(source.K[0, 0]), 0.0,                float(source.K[0, 2])],
                    [0.0,                   float(source.K[1, 1]), float(source.K[1, 2])],
                    [0.0,                   0.0,                1.0]
                ],
                "distortions": [
                    float(source.D[0]), float(source.D[1]),
                    float(source.D[2]), float(source.D[3])
                ],
                "rotation": [
                    float(source.R[0]), float(source.R[1]), float(source.R[2])
                ],
                "translation": [
                    float(source.T[0]), float(source.T[1]), float(source.T[2])
                ],
                "fisheye": False
            }
    
    data["metadata"] = {
        "adjusted": False,
        "error": 0.0
    }
    
    with open(calib_output_path, 'w') as f:
        toml.dump(data, f)


def recap_calibrate(config, calib_full_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''

    for source in config.sources:
        if len(source.K) != 0 and len(source.D) != 0 and len(source.R) != 0 and len(source.T) != 0 and len(source.S) != 0:
            r_trans, t_trans = world_to_camera_persp(source.R, source.T)
            source.R = np.array(cv2.Rodrigues(r_trans)[0]).flatten()
            source.T = np.array(t_trans)
            
            f_px = source.K[0, 0]
            Dm = euclidean_distance(source.T, [0, 0, 0])
            
            if calib_full_type in ['convert_qualisys', 'convert_vicon', 'convert_opencap', 'convert_biocv']:
                source.ret_mm = np.around(source.ret, decimals=3)
                source.ret_px = np.around(source.ret / (Dm * 1000) * f_px, decimals=3)
            elif calib_full_type == 'calculate':
                source.ret_px = np.around(source.ret, decimals=3)
                source.ret_mm = np.around(source.ret * Dm * 1000 / f_px, decimals=3)

    ret_px_list = [s.ret_px for s in config.sources if hasattr(s, 'ret_px')]
    ret_mm_list = [s.ret_mm for s in config.sources if hasattr(s, 'ret_mm')]

    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px_list} px, \nwhich corresponds to {ret_mm_list} mm.\n')


def calibrate_cams_all(config):
    '''
    Either converts a preexisting calibration file, 
    or calculates calibration from scratch (from a board or from points).
    Stores calibration in a .toml file
    Prints recap.
    
    INPUTS:
    - a config_dict dictionary

    OUTPUT:
    - a .toml camera calibration file
    '''

    # Read config_dict
    calib_type, args_calib, calib_output_path = config.get_calibration_params()

    # Map calib function
    calib_mapping = {
        'convert_qualisys': calib_qca_fun,
        'convert_optitrack': calib_optitrack_fun,
        'convert_vicon': calib_vicon_fun,
        'convert_opencap': calib_opencap_fun,
        'convert_easymocap': calib_easymocap_fun,
        'convert_biocv': calib_biocv_fun,
        'calculate': calib_calc_fun,
        }
    calib_fun = calib_mapping[calib_type]

    # Calibrate
    calib_fun(*args_calib)

    # Write calibration file
    toml_write(config, calib_output_path)

    # Recap message
    recap_calibrate(config, calib_type)
    
    logging.info(f'Calibration file is stored at {calib_output_path}.')
