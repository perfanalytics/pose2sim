#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## CAMERAS CALIBRATION                                                   ##
    ###########################################################################
    
    Use this module to calibrate your cameras from checkerboard images and 
    save results to a .toml file.
    Based on https://docs.opencv.org/3.4.15/dc/dbb/tutorial_py_calibration.html.

    /!\ Beware that corners must be detected on all frames, or else extrinsic 
    parameters may be wrong. Set show_corner_detection to 1 to verify.

    Usage: 
    calib_from_checkerboard -d "calib_path" -v False -e jpg -n 6 9 -S 1000
    OR calib_from_checkerboard -d "calib_path" -v True -e mp4 -n 6 9 -S 1000 1000 -s 1 -O 0 -f 50 -o Test.toml
    OR from Pose2Sim.Utilities import calib_from_checkerboard; calib_from_checkerboard.calibrate_cams_func(calib_dir=r"calib_path", 
                video=False, extension="jpg", corners_nb=(6,9), square_size=[1000])
    
'''


## INIT
import os
import numpy as np
import cv2
import glob
import toml
import argparse


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
    parser.add_argument('-d', '--calib_dir', required = True, help='Directory of checkerboard images or videos (one folder per camera)')
    parser.add_argument('-v', '--video', required = True, help='True if calibrate from videos, False if calibrate from images')
    parser.add_argument('-e', '--extension', required=True, help='Video or image files extension (jpg, png, mp4, etc)')
    parser.add_argument('-n', '--corners_nb', nargs=2, type=int, required=True, help='Number of (internal) square corners in the checkerboard: h,w')
    parser.add_argument('-S', '--square_size', nargs='*', type=int, required=True, help='Square or rectangle size in mm (int or int int)')
    parser.add_argument('-O', '--frame_for_origin', required=False, type=int, default=0, help='Checkerboard placed at world origin at frame N (-1 if last frame)')
    parser.add_argument('-f', '--vid_snapshot_every_N_frames', type=int, required=False, help='Calibrate on each N frame of the video (if applicable)')
    parser.add_argument('-s', '--show_corner_detection', type=int, required=False, default=0, help='Display corners detection overlayed on image')
    parser.add_argument('-o', '--output_file', required=False, default="Calib.toml", help='Output calibration file name')
    args = vars(parser.parse_args())

    
    calibrate_cams_func(**args)


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
    
    euc_dist = np.sqrt(np.sum( [d**2 for d in dist] ))
    
    return euc_dist


def rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0):
    '''
    Apply rotations around x, y, z in cameras coordinates
    '''

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

def calib_checkerboard(criteria, **args):
    '''
    Calibrates from images or videos of a checkerboard

    INPUTS:
    - criteria: (int, int, float): Type of criteria, max iterations, min error
    - calib_dir: string: directory of camera folders with checkerboard images
    - video: bool: True if video, False if images
    - extension: string: jpg, mpa, etc
    - corners_nb: (int, int): number of internal corners in the checkerboard (h, w)
    - square_size: (int) or (int, int): square or rectangle size in mm (h,w)
    - frame_for origin: int: checkerboard placed at world origin at frame N
    - vid_snapshot_every_N_frames: int: if video, calibrate on each N frame 

    OUTPUTS:
    - ret: residual reprojection error in _px_: list of floats
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats
    '''
    
    calib_dir = args.get('calib_dir')
    cam_listdirs_names = next(os.walk(calib_dir))[1]
    video = True if args.get('video')==True or args.get('video')=='True' else False
    extension = args.get('extension')
    corners_nb = args.get('corners_nb')
    square_size = args.get('square_size')
    square_size = square_size*2 if len(square_size)==1 else square_size
    frame_for_origin = 0 if args.get('frame_for_origin')==None else args.get('frame_for_origin')
    if video:
        vid_snapshot_every_N_frames = args.get('vid_snapshot_every_N_frames')
    show = args.get('show_corner_detection')

    # Prepare object points
    objp = np.zeros((corners_nb[0]*corners_nb[1],3), np.float32) 
    objp[:,:2] = np.mgrid[0:corners_nb[0],0:corners_nb[1]].T.reshape(-1,2)
    objp[:,0] = objp[:,0]*square_size[0]
    objp[:,1] = objp[:,1]*square_size[1]


    objpoints = [] # 3d points in world space
    imgpoints = [] # 2d points in image plane
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    for cam in cam_listdirs_names:
        # Find corners in vid
        if video:
            video = glob.glob(os.path.join(calib_dir, cam, '*.'+ extension))[0]
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
        else:
            images = glob.glob(os.path.join(calib_dir, cam, '*.'+ extension))
            for image_f in images:
                img = cv2.imread(image_f)
                imgp = findCorners(img, corners_nb, criteria, show)
                if isinstance(imgp, np.ndarray):
                    objpoints.append(objp)
                    imgpoints.append(imgp)
            
        # Calibration
        r, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], 
                                        None, None, flags=cv2.CALIB_FIX_K3)
        h, w = img.shape[:-1]
        
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
            size = f'size = [ {S[c][0]}, {S[c][1]},]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]},], [ 0.0, {K[c][1,1]}, {K[c][1,2]},], [ 0.0, 0.0, 1.0,],]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]},]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]},]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]},]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def recap_calibrate(ret, calib_path):
    '''
    Print a log message giving filtering parameters.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            fm = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            ret_px.append( np.around(ret[c], decimals=3) )
            ret_m.append( np.around(ret[c]*1000 * Dm / fm, decimals=3) )

    print(f'--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, which corresponds to {ret_m} mm.')
    print(f'Calibration file is stored at {calib_path}.')


def calibrate_cams_func(**args):
    '''
    Calibration from a checkerboard.
    Stores it in a .toml file
    Prints recap.
    
    Usage: 
    calib_from_checkerboard -d "calib_path" -v False -e jpg -n 6 9 -s 1000
    OR calib_from_checkerboard -d "calib_path" -v True -e mp4 -n 6 9 -s 1000 1000 -O 0 -f 50 -o Test.toml
    OR import calib_from_checkerboard; calib_from_checkerboard.calibrate_cams_func(calib_dir=r"calib_path", 
                video=False, extension="jpg", corners_nb=(6,9), square_size=[1000])
    '''
    
    calib_dir = args.get('calib_dir')
    output_file = 'Calib.toml' if args.get('output_file')==None else args.get('output_file')
    calib_path = os.path.join(calib_dir, output_file)
    
    # Calibrate
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # stop refining after 30 iterations or if error less than 0.001px
    ret, C, S, D, K, R, T = calib_checkerboard(criteria, **args)

    # Write calibration file
    toml_write(calib_path, C, S, D, K, R, T)

    # Recap message
    recap_calibrate(ret, calib_path)


if __name__ == '__main__':
    main()
