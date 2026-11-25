#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ##################################################
    ## Calibrate cameras from known points on image ##
    ##################################################

    Calibration from known scene coordinates.
    Also works without intrinsic parameters.

    INPUTS:
    - img_path: path of an image (or video, first frame), or list of one path per camera
    - object_coords_3d: list of 3D coordinates in meters, [[x1,y1,z1], [x2,y2,z2], ...]
      OR coord_path: path to a .toml file containing an `object_coords_3d` item. 
    - calib_path (optional): path to a Pose2Sim calibration file from which intrinsics are retrieved. 
    - out_path (optional): path of the output calibration file (.toml). 

    OUTPUTS:
    - a Pose2Sim calibration file (.toml)
    - one debug image per camera, in a 'debug_images' folder next to the calibration file
    - a Calibration_points.trc file next to the calibration file, representing object_coords_3d

    N.B.: Requires Pose2Sim

    Usage:
    from Pose2Sim.Utilities import calib_from_image; calib_from_image.calib_from_image_func(img_path=['cam1.png', 'cam2.png'], coord_path='scene.toml')
    calib_from_image -i cam01_ext.png -c config.toml
    calib_from_image -i cam01_ext.png --object_coords_3d "[[-2.0,  0.3,  0.0], [-2.0 , 0.0,  0.0], [-2.0, 0.0,  0.05], [-2.0, -0.3 ,  0.0], [0.0,  0.3,  0.0], [0.0, 0.0,  0.0], [0.0, 0.0,  0.05], [0.0, -0.3,  0.0]]"
    calib_from_image -i cam01_ext.png cam02_ext.png -c Config.toml -p calib.toml -o Calib_scene.toml
'''


## INIT
import argparse
import ast
import json
import logging
import re
from collections import deque
from pathlib import Path

import numpy as np
import cv2
import rtoml

import Pose2Sim.calibration as p2s_calib # also sets the matplotlib backend
from Pose2Sim.calibration import imgp_objp_visualizer_clicker, calc_extrinsics, \
                                 create_image_labels, toml_write, trc_write
import matplotlib.pyplot as plt


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2026"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.0.2'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def find_object_coords_3d(data, key='object_coords_3d'):
    '''
    Breadth-first search of a key anywhere in a nested dict/list structure.
    The shallowest occurrence is returned.

    INPUTS:
    - data: dictionary loaded from a toml file
    - key: name of the item to look for

    OUTPUTS:
    - value of the item, or None if not found
    - location of the item (e.g. 'calibration.calculate.extrinsics.scene.object_coords_3d')
    '''

    queue = deque([('', data)])
    while queue:
        path, node = queue.popleft()
        if isinstance(node, dict):
            if key in node:
                return node[key], f'{path}.{key}'.lstrip('.')
            for k, v in node.items():
                queue.append((f'{path}.{k}' if path else str(k), v))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, (dict, list)):
                    queue.append((f'{path}[{i}]', v))
    return None, None


def to_coords_array(coords, source='object_coords_3d'):
    '''
    Check and convert 3D coordinates to an (N, 3) float array.
    '''

    try:
        arr = np.asarray(coords, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(f'{source} must be a list of [x, y, z] coordinates.')
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f'{source} must be a list of [x, y, z] coordinates, but has shape {arr.shape}.')
    if len(arr) < 4:
        raise ValueError(f'{source} has only {len(arr)} points. At least 4 are needed (6 or more recommended).')
    return arr


def load_object_coords_3d(coord_path):
    '''
    Retrieve object_coords_3d from a .toml file, wherever it is in the hierarchy.
    '''

    coord_path = Path(coord_path)
    if coord_path.suffix.lower() != '.toml':
        raise ValueError(f'coord_path must be a .toml file, but got {coord_path}.')
    if not coord_path.is_file():
        raise FileNotFoundError(f'{coord_path} not found.')

    coords, location = find_object_coords_3d(rtoml.load(coord_path))
    if coords is None:
        raise ValueError(f'No "object_coords_3d" item found in {coord_path}.')
    logging.info(f'Found object_coords_3d in {coord_path.name} ("{location}").')

    return to_coords_array(coords, source=f'object_coords_3d ("{location}")')


def read_intrinsics(calib_path):
    '''
    Read intrinsic parameters of all cameras of a Pose2Sim calibration file.
    Cameras with missing or invalid intrinsics are skipped.

    OUTPUTS:
    - list of dictionaries with keys 'name', 'K' (3x3), 'D' (>=4), 'size'
    '''

    calib_path = Path(calib_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f'{calib_path} not found.')
    data = rtoml.load(calib_path)
    if isinstance(data.get('cameras'), dict):
        data = data['cameras']

    cams = []
    for key, cam in data.items():
        if key == 'metadata' or not isinstance(cam, dict):
            continue
        try:
            K = np.array(cam['matrix'], dtype=float)
            D = np.array(cam['distortions'], dtype=float).flatten()
            assert K.shape == (3, 3) and K[0, 0] > 0 and len(D) >= 4
        except Exception:
            logging.warning(f'No valid intrinsics for camera "{key}" in {calib_path.name}: skipped.')
            continue
        cams.append({'name': cam.get('name', key), 'K': K, 'D': D, 'size': cam.get('size')})
    if not cams:
        raise ValueError(f'No camera with valid intrinsics found in {calib_path}.')

    return cams


def match_intrinsics(img_names, cams):
    '''
    Match each image with a camera of the calibration file.
    By name first. If some names do not match but the number of cameras equals 
    the number of images, by order.

    OUTPUTS:
    - list of camera dictionaries (or None where intrinsics have to be estimated), one per image
    '''

    by_name = {c['name']: c for c in cams}
    matched = [by_name.get(n) for n in img_names]
    if None in matched:
        if len(cams) == len(img_names):
            logging.warning('Image names do not all match camera names of the calibration file: matching them by order.')
            matched = list(cams)
        else:
            missing = [n for n, m in zip(img_names, matched) if m is None]
            logging.warning(f'No intrinsics found for {missing}: focal length will be estimated.')
    return matched


def make_cam_names(img_paths):
    '''
    Camera names from image names, made valid and unique for a .toml table name.
    '''

    names, count = [], {}
    for p in img_paths:
        base = re.sub(r'[^\w\-]', '_', p.stem)
        count[base] = count.get(base, 0) + 1
        names.append(base if count[base] == 1 else f'{base}_{count[base]}')
    return names


def read_image(img_path):
    '''
    Read an image, or the first frame of a video. Returns a BGR image.
    '''

    img = cv2.imread(str(img_path))
    if img is None:
        cap = cv2.VideoCapture(str(img_path))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f'Could not read image or video: {img_path}')
    return img


def reset_clicker_state():
    '''
    imgp_objp_visualizer_clicker stores its results in module globals. 
    Clear them so that a previous image cannot leak into the next one 
    (e.g. if the window is closed without confirming).
    '''

    for var in ['imgp_confirmed', 'objp_confirmed', 'objp_confirmed_notok', 
                'scat', 'ax_3d', 'fig_3d', 'events', 'count']:
        p2s_calib.__dict__.pop(var, None)
    plt.close('all')


def calib_from_image_func(**args):
    '''
    Calibrate cameras from known 3D points of the scene and the corresponding 
    points clicked by hand on one image per camera.

    INPUTS (keyword arguments):
    - img_path: path or list of paths of images (or videos), one per camera
    - object_coords_3d: list of [x, y, z] coordinates in meters
    - coord_path: .toml file containing an `object_coords_3d` item (alternative to object_coords_3d)
    - optional: calib_path: Pose2Sim calibration file to retrieve intrinsics from
    - optional: out_path: output calibration file (.toml), or folder
    - optional: display: display reprojection on each image after calibration (default True)

    OUTPUTS:
    - calibration .toml file, debug images, and Calibration_points.trc
    - returns the path of the calibration file
    '''

    img_paths = args.get('img_path')
    object_coords_3d = args.get('object_coords_3d')
    coord_path = args.get('coord_path')
    calib_path = args.get('calib_path')
    out_path = args.get('out_path')
    show = args.get('display', True)

    # Images
    if img_paths is None:
        raise ValueError('img_path is required.')
    if isinstance(img_paths, (str, Path)):
        img_paths = [img_paths]
    img_paths = [Path(p) for p in img_paths]
    not_found = [str(p) for p in img_paths if not p.is_file()]
    if not_found:
        raise FileNotFoundError(f'Image(s) not found: {not_found}')
    cam_names = make_cam_names(img_paths)

    # Object points
    if (object_coords_3d is None) == (coord_path is None):
        raise ValueError('Provide either object_coords_3d or coord_path (not both).')
    if coord_path is not None:
        object_coords_3d = load_object_coords_3d(coord_path)
    else:
        object_coords_3d = to_coords_array(object_coords_3d)

    # Intrinsics
    if calib_path is not None:
        matched = match_intrinsics([p.stem for p in img_paths], read_intrinsics(calib_path))
    else:
        matched = [None] * len(img_paths)

    # Output paths
    out_path = Path(out_path) if out_path else img_paths[0].parent / 'Calib_scene.toml'
    if out_path.suffix == '':
        out_path = out_path / 'Calib_scene.toml'
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    trc_path = out_dir / 'Calibration_points.trc'

    # Save reference 3D coordinates as trc
    trc_write(object_coords_3d, trc_path)

    # Calibrate each camera
    C, S, D, K, R, T, rms_list = [], [], [], [], [], [], []
    for img_path, cam_name, cam in zip(img_paths, cam_names, matched):
        logging.info(f'\nCamera {cam_name} ({img_path.name}):')
        img = read_image(img_path)
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Click image points
        reset_clicker_state()
        res = imgp_objp_visualizer_clicker(img_rgb.copy(), imgp=[], objp=object_coords_3d, img_path=str(img_path))
        if not (isinstance(res, tuple) and len(res) == 2):
            raise ValueError(f'No points validated for {img_path.name}.')
        imgp = np.asarray(res[0], dtype=float).reshape(-1, 2)
        objp_selected = np.asarray(res[1], dtype=float).reshape(-1, 3)
        if len(imgp) < 4 or len(imgp) != len(objp_selected):
            raise ValueError(f'{img_path.name}: {len(imgp)} image points for {len(objp_selected)} object points. '
                             f'At least 4 matching points are needed (6 or more recommended). '
                             f'Press "C", click the points in the order they are numbered, then confirm.')

        # Calibrate: intrinsics are estimated if not provided
        K_i = D_i = None
        if cam is not None:
            K_i, D_i = cam['K'], cam['D']
            if cam['size'] is not None and (abs(cam['size'][0] - w) > 1 or abs(cam['size'][1] - h) > 1):
                logging.warning(f'Image size ({w}x{h}) differs from the one of the calibration file '
                                f'({cam["size"][0]}x{cam["size"][1]}): intrinsics may not be valid.')
        estimated, D_i, K_i, r, t = calc_extrinsics(objp_selected, imgp, K_i, D_i, (w, h))
        K_i, D_i = np.asarray(K_i, dtype=float), np.asarray(D_i, dtype=float)

        # Reprojection error
        proj_selected = cv2.projectPoints(objp_selected, r, t, K_i, D_i)[0].reshape(-1, 2)
        proj_all = cv2.projectPoints(object_coords_3d, r, t, K_i, D_i)[0].reshape(-1, 2) # including hidden points
        rms = float(np.sqrt(np.mean(np.sum((proj_selected - imgp) ** 2, axis=1))))
        logging.info(f'Reprojection error: {rms:.2f} px for f = {K_i[0,0]:.1f} px ' 
                     f'({"estimated" if estimated else "from calibration file"}).')

        # Debug image
        create_image_labels(str(img_path), imgp, out_dir, 'calib', reprojected_points=proj_all, show=show, save=True)
        if show:
            plt.gcf().canvas.manager.set_window_title(f'Close {cam_name} to continue')
            plt.show(block=True)

        C.append(cam_name)
        S.append([w, h])
        D.append(D_i)
        K.append(K_i)
        R.append(r)
        T.append(t)
        rms_list.append(rms)

    # Write calibration file
    toml_write(out_path, C, S, D, K, R, T)

    logging.info(f'\n--> Reprojection errors for each camera are respectively {[round(e, 2) for e in rms_list]} px.')
    logging.info(f'Calibration file is stored at {out_path}.')
    logging.info(f'Debug images are stored in {out_dir / "debug_images"}.')
    logging.info(f'Object points are stored at {trc_path}.')

    return out_path


def parse_coords_string(s):
    '''
    Parse a list of coordinates given as a string on the command line
    '''

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            raise argparse.ArgumentTypeError('object_coords_3d must be a list of [x, y, z] coordinates, '
                                             'e.g. "[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]"')


def main():
    parser = argparse.ArgumentParser(description='Calibrate cameras from known 3D points and points clicked on one image per camera.')
    parser.add_argument('-i', '--img_path', required=True, nargs='+', help='path(s) of the image(s), one per camera')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c', '--coord_path', help='path to a .toml file with an "object_coords_3d" item, anywhere in its hierarchy')
    group.add_argument('--object_coords_3d', type=parse_coords_string, help='list of 3D coordinates in meters, e.g. "[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]"')
    parser.add_argument('-p', '--calib_path', help='optional: path to a Pose2Sim calibration file to retrieve intrinsics from')
    parser.add_argument('-o', '--out_path', help='optional: path of the output calibration file (.toml)')
    parser.add_argument('--no_display', action='store_true', help='do not display the reprojection after each calibration')
    args = vars(parser.parse_args())
    args['display'] = not args.pop('no_display')

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    calib_from_image_func(**args)


if __name__ == '__main__':
    main()
