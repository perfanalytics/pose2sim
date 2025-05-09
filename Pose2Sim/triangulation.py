#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## ROBUST TRIANGULATION  OF 2D COORDINATES                               ##
###########################################################################

This module triangulates 2D json coordinates and builds a .trc file readable 
by OpenSim.

The triangulation is weighted by the likelihood of each detected 2D keypoint 
(if they meet the likelihood threshold). If the reprojection error is above a
threshold, right and left sides are swapped; if it is still above, a camera 
is removed for this point and this frame, until the threshold is met. If more 
cameras are removed than a predefined minimum, triangulation is skipped for 
the point and this frame. In the end, missing values are interpolated.

In case of multiple subjects detection, make sure you first run the 
personAssociation module. It will then associate people across frames by 
measuring the frame-by-frame distance between them.

INPUTS: 
- a calibration file (.toml extension)
- json files for each camera with only one person of interest
- a Config.toml file
- a skeleton model

OUTPUTS: 
- a .trc file with 3D coordinates in Y-up system coordinates
'''


## INIT
import os
import glob
import fnmatch
import re
import numpy as np
import json
import itertools as it
import pandas as pd
import cv2
import toml
from tqdm import tqdm
from collections import Counter
from anytree import RenderTree
from anytree.importer import DictImporter
import logging

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, sort_people_sports2d, interpolate_zeros_nans, \
    sort_stringlist_by_last_number, zup2yup, convert_to_c3d
from Pose2Sim.skeletons import *


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
def count_persons_in_json(file_path):
    '''
    Count the number of persons in a json file.

    INPUT:
    - file_path: path to the json file

    OUTPUT:
    - int: number of persons in the json file
    '''

    with open(file_path, 'r') as file:
        data = json.load(file)
        return len(data.get('people', []))
    

def indices_of_first_last_non_nan_chunks(series, min_chunk_size=10):
    '''
    Find indices of the first and last chunks of at least min_chunk_size consecutive non-NaN values.

    INPUT:
    - series: pandas Series to trim
    - min_chunk_size: minimum size of consecutive non-NaN values to consider (default: 5)

    OUTPUT:
    - tuple: (start_index, end_index) of the first and last valid chunks
    '''
    
    non_nan_mask = ~np.isnan(series.values)
    
    # Find runs of consecutive non-NaN values (eg [(8, 15), (16, 17), (19, 26)])
    runs = []
    run_start = None
    for i, bool_val in enumerate(non_nan_mask):
        if bool_val and run_start is None:
            run_start = i
        elif not bool_val and run_start is not None:
            run_end = i
            runs.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(non_nan_mask)))
    
    # Find runs that have at least min_chunk_size consecutive non-NaN values
    valid_runs = [(start, end) for start, end in runs if end - start >= min_chunk_size]
    if not valid_runs:
        return(0,0)
    
    # Get the start of the first valid run and the end of the last valid run
    first_run_start = valid_runs[0][0]
    last_run_end = valid_runs[-1][1]
    
    # Return the trimmed series
    return first_run_start, last_run_end


def make_trc(config_dict, Q, keypoints_names, f_range, id_person=-1):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - config_dict: dictionary of configuration parameters
    - Q: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings
    - f_range: list of two numbers. Range of frames

    OUTPUT:
    - trc file
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    multi_person = config_dict.get('project').get('multi_person')
    if multi_person:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}_P{id_person+1}'
    else:
        seq_name = f'{os.path.basename(os.path.realpath(project_dir))}'
    pose3d_dir = os.path.join(project_dir, 'pose-3d')

    # Get frame_rate
    video_dir = os.path.join(project_dir, 'videos')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))
    frame_rate = config_dict.get('project').get('frame_rate')
    if frame_rate == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
        except:
            frame_rate = 60

    trc_f = f'{seq_name}_{f_range[0]}-{f_range[1]}.trc'

    #Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    NumFrames = len(Q)
    NumMarkers = len(keypoints_names)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_f, 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, f_range[0], f_range[1]])),
            'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(len(keypoints_names))]) + '\t']
    
    # Zup to Yup coordinate system
    Q = zup2yup(Q)
    
    #Add Frame# and Time columns
    Q.index = np.array(range(f_range[0], f_range[1]))
    Q.insert(0, 't', Q.index/ frame_rate)
    # Q = Q.fillna(' ')

    #Write file
    if not os.path.exists(pose3d_dir): os.mkdir(pose3d_dir)
    trc_path = os.path.realpath(os.path.join(pose3d_dir, trc_f))
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        Q.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def retrieve_right_trc_order(trc_paths):
    '''
    Lets the user input which static file correspond to each generated trc file.
    
    INPUT:
    - trc_paths: list of strings
    
    OUTPUT:
    - trc_id: list of integers
    '''
    
    logging.info('\n\nReordering trc file IDs:')
    logging.info(f'\nPlease visualize the generated trc files in Blender or OpenSim.\nTrc files are stored in {os.path.dirname(trc_paths[0])}.\n')
    retry = True
    while retry:
        retry = False
        logging.info('List of trc files:')
        [logging.info(f'#{t_list}: {os.path.basename(trc_list)}') for t_list, trc_list in enumerate(trc_paths)]
        trc_id = []
        for t, trc_p in enumerate(trc_paths):
            logging.info(f'\nStatic trial #{t} corresponds to trc number:')
            trc_id += [input('Enter ID:')]
        
        # Check non int and duplicates
        try:
            trc_id = [int(t) for t in trc_id]
            duplicates_in_input = (len(trc_id) != len(set(trc_id)))
            if duplicates_in_input:
                retry = True
                print('\n\nWARNING: Same ID entered twice: please check IDs again.\n')
        except:
            print('\n\nWARNING: The ID must be an integer: please check IDs again.\n')
            retry = True
    
    return trc_id


def recap_triangulate(config_dict, error, nb_cams_excluded, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, f_range_trimmed, trc_path):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold 
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe
    - keypoints_names: list of strings

    OUTPUT:
    - Message in console
    '''

    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    calib_files = glob.glob(os.path.join(calib_dir, '*.toml'))
    calib_file = max(calib_files, key=os.path.getctime) # lastly created calibration file
    calib = toml.load(calib_file)
    cal_keys = [c for c in calib.keys() 
            if c not in ['metadata', 'capture_volume', 'charuco', 'checkerboard'] 
            and isinstance(calib[c],dict)]
    cam_names = np.array([calib[c].get('name') if calib[c].get('name') else c for c in cal_keys])
    cam_names = cam_names[list(cam_excluded_count[0].keys())]
    error_threshold_triangulation = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    
    # Recap
    calib_cam1 = calib[cal_keys[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])

    logging.info('')
    nb_persons_to_detect = len(error)
    for n in range(nb_persons_to_detect):
        if nb_persons_to_detect > 1:
            logging.info(f'\n\nPARTICIPANT {n+1}\n')
        
        for idx, name in enumerate(keypoints_names):
            mean_error_keypoint_px = np.around(error[n].iloc[:,idx].mean(), decimals=1) # RMS à la place?
            mean_error_keypoint_m = np.around(mean_error_keypoint_px * Dm / fm, decimals=3)
            mean_cam_excluded_keypoint = np.around(nb_cams_excluded[n].iloc[:,idx].mean(), decimals=2)
            logging.info(f'Mean reprojection error for {name} is {mean_error_keypoint_px} px (~ {mean_error_keypoint_m} m), reached with {mean_cam_excluded_keypoint} excluded cameras. ')
            if show_interp_indices:
                if interpolation_kind != 'none':
                    if len(list(interp_frames[n][idx])) == 0 and len(list(non_interp_frames[n][idx])) == 0:
                        logging.info(f'  No frames needed to be interpolated.')
                    if len(list(interp_frames[n][idx]))>0: 
                        interp_str = str(interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {interp_str} were interpolated.')
                    if len(list(non_interp_frames[n][idx]))>0:
                        noninterp_str = str(non_interp_frames[n][idx]).replace(":", " to ").replace("'", "").replace("]", "").replace("[", "")
                        logging.info(f'  Frames {noninterp_str} were not interpolated.')
                else:
                    logging.info(f'  No frames were interpolated because \'interpolation_kind\' was set to none. ')
        
        mean_error_px = np.around(error[n]['mean'].mean(), decimals=1)
        mean_error_mm = np.around(mean_error_px * Dm / fm *1000, decimals=1)
        mean_cam_excluded = np.around(nb_cams_excluded[n]['mean'].mean(), decimals=2)

        logging.info(f'\n--> Mean reprojection error for all points on frames {f_range_trimmed[n][0]} to {f_range_trimmed[n][1]} is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
        logging.info(f'Cameras were excluded if likelihood was below {likelihood_threshold} and if the reprojection error was above {error_threshold_triangulation} px.') 
        if interpolation_kind != 'none':
            logging.info(f'Gaps were interpolated with {interpolation_kind} method if smaller than {interp_gap_smaller_than} frames. Larger gaps were filled with {["the last valid value" if fill_large_gaps_with == "last_value" else "zeros" if fill_large_gaps_with == "zeros" else "NaNs"][0]}.') 
        logging.info(f'In average, {mean_cam_excluded} cameras had to be excluded to reach these thresholds.')
        
        cam_excluded_count[n] = {i: v for i, v in zip(cam_names, cam_excluded_count[n].values())}
        cam_excluded_count[n] = {k: v for k, v in sorted(cam_excluded_count[n].items(), key=lambda item: item[1])[::-1]}
        str_cam_excluded_count = ''
        for i, (k, v) in enumerate(cam_excluded_count[n].items()):
            if i ==0:
                 str_cam_excluded_count += f'Camera {k} was excluded {int(np.round(v*100))}% of the time, '
            elif i == len(cam_excluded_count[n])-1:
                str_cam_excluded_count += f'and Camera {k}: {int(np.round(v*100))}%.'
            else:
                str_cam_excluded_count += f'Camera {k}: {int(np.round(v*100))}%, '
        logging.info(str_cam_excluded_count)
        logging.info(f'\n3D coordinates are stored at {trc_path[n]}.')
        
    logging.info('\n\n')
    if make_c3d:
        logging.info('All trc files have been converted to c3d.')
    logging.info(f'Limb swapping was {"handled" if handle_LR_swap else "not handled"}.')
    logging.info(f'Lens distortions were {"taken into account" if undistort_points else "not taken into account"}.')


def triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, projection_matrices, calib_params):
    '''
    Triangulates 2D keypoint coordinates. If reprojection error is above threshold,
    tries swapping left and right sides. If still above, removes a camera until error
    is below threshold unless the number of remaining cameras is below a predefined number.

    1. Creates subset with N cameras excluded 
    2. Tries all possible triangulations
    3. Chooses the one with smallest reprojection error
    If error too big, take off one more camera.
        If then below threshold, retain result.
        If better but still too big, take off one more camera.
    
    INPUTS:
    - a Config.toml file
    - coords_2D_kpt: (x,y,likelihood) * ncams array
    - coords_2D_kpt_swapped: (x,y,likelihood) * ncams array  with left/right swap
    - projection_matrices: list of arrays

    OUTPUTS:
    - Q: array of triangulated point (x,y,z,1.)
    - error_min: float
    - nb_cams_excluded: int
    '''
    
    # Read config_dict
    error_threshold_triangulation = config_dict.get('triangulation').get('reproj_error_threshold_triangulation')
    min_cameras_for_triangulation = config_dict.get('triangulation').get('min_cameras_for_triangulation')
    handle_LR_swap = config_dict.get('triangulation').get('handle_LR_swap')

    undistort_points = config_dict.get('triangulation').get('undistort_points')
    if undistort_points:
        calib_params_K = calib_params['K']
        calib_params_dist = calib_params['dist']
        calib_params_R = calib_params['R']
        calib_params_T = calib_params['T']

    # Initialize
    x_files, y_files, likelihood_files = coords_2D_kpt
    x_files_swapped, y_files_swapped, likelihood_files_swapped = coords_2D_kpt_swapped
    n_cams = len(x_files)
    error_min = np.inf 
    
    nb_cams_off = 0 # cameras will be taken-off until reprojection error is under threshold
    # print('\n')
    while error_min > error_threshold_triangulation and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # print("error min ", error_min, "thresh ", error_threshold_triangulation, 'nb_cams_off ', nb_cams_off)
        # Create subsets with "nb_cams_off" cameras excluded
        id_cams_off = np.array(list(it.combinations(range(n_cams), nb_cams_off)))
        
        if undistort_points:
            calib_params_K_filt = [calib_params_K]*len(id_cams_off)
            calib_params_dist_filt = [calib_params_dist]*len(id_cams_off)
            calib_params_R_filt = [calib_params_R]*len(id_cams_off)
            calib_params_T_filt = [calib_params_T]*len(id_cams_off)
        projection_matrices_filt = [projection_matrices]*len(id_cams_off)

        x_files_filt = np.vstack([x_files.copy()]*len(id_cams_off))
        y_files_filt = np.vstack([y_files.copy()]*len(id_cams_off))
        x_files_swapped_filt = np.vstack([x_files_swapped.copy()]*len(id_cams_off))
        y_files_swapped_filt = np.vstack([y_files_swapped.copy()]*len(id_cams_off))
        likelihood_files_filt = np.vstack([likelihood_files.copy()]*len(id_cams_off))
        
        if nb_cams_off > 0:
            for i in range(len(id_cams_off)):
                x_files_filt[i][id_cams_off[i]] = np.nan
                y_files_filt[i][id_cams_off[i]] = np.nan
                x_files_swapped_filt[i][id_cams_off[i]] = np.nan
                y_files_swapped_filt[i][id_cams_off[i]] = np.nan
                likelihood_files_filt[i][id_cams_off[i]] = np.nan
        
        # Excluded cameras index and count
        id_cams_off_tot_new = [np.argwhere(np.isnan(x)).ravel() for x in likelihood_files_filt]
        nb_cams_excluded_filt = [np.count_nonzero(np.nan_to_num(x)==0) for x in likelihood_files_filt] # count nans and zeros
        nb_cams_off_tot = max(nb_cams_excluded_filt)
        # print('likelihood_files_filt ',likelihood_files_filt)
        # print('nb_cams_excluded_filt ', nb_cams_excluded_filt, 'nb_cams_off_tot ', nb_cams_off_tot)
        if nb_cams_off_tot > n_cams - min_cameras_for_triangulation:
            break
        id_cams_off_tot = id_cams_off_tot_new
        
        # print('still in loop')
        if undistort_points:
            calib_params_K_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_K_filt) ]
            calib_params_dist_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_dist_filt) ]
            calib_params_R_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_R_filt) ]
            calib_params_T_filt = [ [ c[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, c in enumerate(calib_params_T_filt) ]
        projection_matrices_filt = [ [ p[i] for i in range(n_cams) if not np.isnan(likelihood_files_filt[j][i]) and not likelihood_files_filt[j][i]==0. ] for j, p in enumerate(projection_matrices_filt) ]
        
        # print('\nnb_cams_off', repr(nb_cams_off), 'nb_cams_excluded', repr(nb_cams_excluded_filt))
        # print('likelihood_files ', repr(likelihood_files))
        # print('y_files ', repr(y_files))
        # print('x_files ', repr(x_files))
        # print('x_files_swapped ', repr(x_files_swapped))
        # print('likelihood_files_filt ', repr(likelihood_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # print('id_cams_off_tot ', id_cams_off_tot)
        
        x_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_filt) ]
        y_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_filt) ]
        x_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(x_files_swapped_filt) ]
        y_files_swapped_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(likelihood_files_filt[i][ii]) and not likelihood_files_filt[i][ii]==0. ]) for i,x in enumerate(y_files_swapped_filt) ]
        likelihood_files_filt = [ np.array([ xx for ii, xx in enumerate(x) if not np.isnan(xx) and not xx==0. ]) for x in likelihood_files_filt ]
        # print('y_files_filt ', repr(y_files_filt))
        # print('x_files_filt ', repr(x_files_filt))
        # Triangulate 2D points
        Q_filt = [weighted_triangulation(projection_matrices_filt[i], x_files_filt[i], y_files_filt[i], likelihood_files_filt[i]) for i in range(len(id_cams_off))]
        
        # Reprojection
        if undistort_points:
            coords_2D_kpt_calc_filt = [np.array([cv2.projectPoints(np.array(Q_filt[i][:-1]), calib_params_R_filt[i][j], calib_params_T_filt[i][j], calib_params_K_filt[i][j], calib_params_dist_filt[i][j])[0].ravel() 
                                        for j in range(n_cams-nb_cams_excluded_filt[i])]) 
                                        for i in range(len(id_cams_off))]
            coords_2D_kpt_calc_filt = [[coords_2D_kpt_calc_filt[i][:,0], coords_2D_kpt_calc_filt[i][:,1]] for i in range(len(id_cams_off))]
        else:
            coords_2D_kpt_calc_filt = [reprojection(projection_matrices_filt[i], Q_filt[i]) for i in range(len(id_cams_off))]
        coords_2D_kpt_calc_filt = np.array(coords_2D_kpt_calc_filt, dtype=object)
        x_calc_filt = coords_2D_kpt_calc_filt[:,0]
        # print('x_calc_filt ', x_calc_filt)
        y_calc_filt = coords_2D_kpt_calc_filt[:,1]
        
        # Reprojection error
        error = []
        for config_off_id in range(len(x_calc_filt)):
            q_file = [(x_files_filt[config_off_id][i], y_files_filt[config_off_id][i]) for i in range(len(x_files_filt[config_off_id]))]
            q_calc = [(x_calc_filt[config_off_id][i], y_calc_filt[config_off_id][i]) for i in range(len(x_calc_filt[config_off_id]))]
            error.append( np.mean( [euclidean_distance(q_file[i], q_calc[i]) for i in range(len(q_file))] ) )
        # print('error ', error)
            
        # Choosing best triangulation (with min reprojection error)
        # print('\n', error)
        # print('len(error) ', len(error))
        # print('len(x_calc_filt) ', len(x_calc_filt))
        # print('len(likelihood_files_filt) ', len(likelihood_files_filt))
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('min error ', np.nanmin(error))
        # print('argmin error ', np.nanargmin(error))
        error_min = np.nanmin(error)
        # print(error_min)
        best_cams = np.nanargmin(error)
        nb_cams_excluded = nb_cams_excluded_filt[best_cams]
        
        Q = Q_filt[best_cams][:-1]


        # Swap left and right sides if reprojection error still too high
        if handle_LR_swap and error_min > error_threshold_triangulation:
            # print('handle')
            n_cams_swapped = 1
            error_off_swap_min = error_min
            while error_off_swap_min > error_threshold_triangulation and n_cams_swapped < (n_cams - nb_cams_off_tot) / 2: # more than half of the cameras switched: may triangulate twice the same side
                # print('SWAP: nb_cams_off ', nb_cams_off, 'n_cams_swapped ', n_cams_swapped, 'nb_cams_off_tot ', nb_cams_off_tot)
                # Create subsets 
                id_cams_swapped = np.array(list(it.combinations(range(n_cams-nb_cams_off_tot), n_cams_swapped)))
                # print('id_cams_swapped ', id_cams_swapped)
                x_files_filt_off_swap = [[x] * len(id_cams_swapped) for x in x_files_filt]
                y_files_filt_off_swap = [[y] * len(id_cams_swapped) for y in y_files_filt]
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('y_files_filt_off_swap ', y_files_filt_off_swap)
                for id_off in range(len(id_cams_off)): # for each configuration with nb_cams_off_tot removed 
                    for id_swapped, config_swapped in enumerate(id_cams_swapped): # for each of these configurations, test all subconfigurations with with n_cams_swapped swapped
                        # print('id_off ', id_off, 'id_swapped ', id_swapped, 'config_swapped ',  config_swapped)
                        x_files_filt_off_swap[id_off][id_swapped][config_swapped] = x_files_swapped_filt[id_off][config_swapped] 
                        y_files_filt_off_swap[id_off][id_swapped][config_swapped] = y_files_swapped_filt[id_off][config_swapped]
                                
                # Triangulate 2D points
                Q_filt_off_swap = np.array([[weighted_triangulation(projection_matrices_filt[id_off], x_files_filt_off_swap[id_off][id_swapped], y_files_filt_off_swap[id_off][id_swapped], likelihood_files_filt[id_off]) 
                                                for id_swapped in range(len(id_cams_swapped))]
                                                for id_off in range(len(id_cams_off))] )
                
                # Reprojection
                if undistort_points:
                    coords_2D_kpt_calc_off_swap = [np.array([[cv2.projectPoints(np.array(Q_filt_off_swap[id_off][id_swapped][:-1]), calib_params_R_filt[id_off][j], calib_params_T_filt[id_off][j], calib_params_K_filt[id_off][j], calib_params_dist_filt[id_off][j])[0].ravel() 
                                                    for j in range(n_cams-nb_cams_off_tot)] 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                    coords_2D_kpt_calc_off_swap = np.array([[[coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,0], coords_2D_kpt_calc_off_swap[id_off][id_swapped,:,1]] 
                                                    for id_swapped in range(len(id_cams_swapped))] 
                                                    for id_off in range(len(id_cams_off))])
                else:
                    coords_2D_kpt_calc_off_swap = [np.array([reprojection(projection_matrices_filt[id_off], Q_filt_off_swap[id_off][id_swapped]) 
                                                    for id_swapped in range(len(id_cams_swapped))])
                                                    for id_off in range(len(id_cams_off))]
                # print(repr(coords_2D_kpt_calc_off_swap))
                x_calc_off_swap = [c[:,0] for c in coords_2D_kpt_calc_off_swap]
                y_calc_off_swap = [c[:,1] for c in coords_2D_kpt_calc_off_swap]
                
                # Reprojection error
                # print('x_files_filt_off_swap ', x_files_filt_off_swap)
                # print('x_calc_off_swap ', x_calc_off_swap)
                error_off_swap = []
                for id_off in range(len(id_cams_off)):
                    error_percam = []
                    for id_swapped, config_swapped in enumerate(id_cams_swapped):
                        # print(id_off,id_swapped,n_cams,nb_cams_off)
                        # print(repr(x_files_filt_off_swap))
                        q_file_off_swap = [(x_files_filt_off_swap[id_off][id_swapped][i], y_files_filt_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        q_calc_off_swap = [(x_calc_off_swap[id_off][id_swapped][i], y_calc_off_swap[id_off][id_swapped][i]) for i in range(n_cams - nb_cams_off_tot)]
                        error_percam.append( np.mean( [euclidean_distance(q_file_off_swap[i], q_calc_off_swap[i]) for i in range(len(q_file_off_swap))] ) )
                    error_off_swap.append(error_percam)
                error_off_swap = np.array(error_off_swap)
                # print('error_off_swap ', error_off_swap)
                
                # Choosing best triangulation (with min reprojection error)
                error_off_swap_min = np.min(error_off_swap)
                best_off_swap_config = np.unravel_index(error_off_swap.argmin(), error_off_swap.shape)
                
                id_off_cams = best_off_swap_config[0]
                id_swapped_cams = id_cams_swapped[best_off_swap_config[1]]
                Q_best = Q_filt_off_swap[best_off_swap_config][:-1]

                n_cams_swapped += 1

            if error_off_swap_min < error_min:
                error_min = error_off_swap_min
                best_cams = id_off_cams
                Q = Q_best
        
        # print(error_min)
        
        nb_cams_off += 1
    
    # Index of excluded cams for this keypoint
    # print('Loop ended')
    
    if 'best_cams' in locals():
        # print(id_cams_off_tot)
        # print('len(id_cams_off_tot) ', len(id_cams_off_tot))
        # print('id_cams_off_tot ', id_cams_off_tot)
        id_excluded_cams = id_cams_off_tot[best_cams]
        # print('id_excluded_cams ', id_excluded_cams)
    else:
        id_excluded_cams = list(range(n_cams))
        nb_cams_excluded = n_cams
    # print('id_excluded_cams ', id_excluded_cams)
    
    # If triangulation not successful, error = nan,  and 3D coordinates as missing values
    if error_min > error_threshold_triangulation:
        error_min = np.nan
        Q = np.array([np.nan, np.nan, np.nan])
        
    return Q, error_min, nb_cams_excluded, id_excluded_cams


def extract_files_frame_f(json_tracked_files_f, keypoints_ids, nb_persons_to_detect):
    '''
    Extract data from json files for frame f, 
    in the order of the body model hierarchy.

    INPUTS:
    - json_tracked_files_f: list of str. Paths of json_files for frame f.
    - keypoints_ids: list of int. Keypoints IDs in the order of the hierarchy.
    - nb_persons_to_detect: int

    OUTPUTS:
    - x_files, y_files, likelihood_files: [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
    '''

    n_cams = len(json_tracked_files_f)
    
    x_files = [[] for n in range(nb_persons_to_detect)]
    y_files = [[] for n in range(nb_persons_to_detect)]
    likelihood_files = [[] for n in range(nb_persons_to_detect)]
    for n in range(nb_persons_to_detect):
        for cam_nb in range(n_cams):
            x_files_cam, y_files_cam, likelihood_files_cam = [], [], []
            try:
                with open(json_tracked_files_f[cam_nb], 'r') as json_f:
                    js = json.load(json_f)
                    for keypoint_id in keypoints_ids:
                        try:
                            x_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3] )
                            y_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+1] )
                            likelihood_files_cam.append( js['people'][n]['pose_keypoints_2d'][keypoint_id*3+2] )
                        except:
                            x_files_cam.append( np.nan )
                            y_files_cam.append( np.nan )
                            likelihood_files_cam.append( np.nan )
            except:
                x_files_cam = [np.nan] * len(keypoints_ids)
                y_files_cam = [np.nan] * len(keypoints_ids)
                likelihood_files_cam = [np.nan] * len(keypoints_ids)
            x_files[n].append(x_files_cam)
            y_files[n].append(y_files_cam)
            likelihood_files[n].append(likelihood_files_cam)
        
    x_files = np.array(x_files)
    y_files = np.array(y_files)
    likelihood_files = np.array(likelihood_files)

    return x_files, y_files, likelihood_files


def triangulate_all(config_dict):
    '''
    For each frame
    For each keypoint
    - Triangulate keypoint
    - Reproject it on all cameras
    - Take off cameras until requirements are met
    Interpolate missing values
    Create trc file
    Print recap message
    
     INPUTS: 
    - a calibration file (.toml extension)
    - json files for each camera with indices matching the detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - a .trc file with 3D coordinates in Y-up system coordinates 
    '''
    
    # Read config_dict
    project_dir = config_dict.get('project').get('project_dir')
    # if batch
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    # if single trial
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    multi_person = config_dict.get('project').get('multi_person')
    pose_model = config_dict.get('pose').get('pose_model')
    frame_range = config_dict.get('project').get('frame_range')
    likelihood_threshold = config_dict.get('triangulation').get('likelihood_threshold_triangulation')
    interpolation_kind = config_dict.get('triangulation').get('interpolation')
    interp_gap_smaller_than = config_dict.get('triangulation').get('interp_if_gap_smaller_than')
    fill_large_gaps_with = config_dict.get('triangulation').get('fill_large_gaps_with')
    show_interp_indices = config_dict.get('triangulation').get('show_interp_indices')
    undistort_points = config_dict.get('triangulation').get('undistort_points')
    make_c3d = config_dict.get('triangulation').get('make_c3d')
    
    try:
        calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    except:
        raise Exception(f'No .toml calibration direcctory found.')
    try:
        calib_files = glob.glob(os.path.join(calib_dir, '*.toml'))
        calib_file = max(calib_files, key=os.path.getctime) # lastly created calibration file
    except:
        raise Exception(f'No .toml calibration file found in the {calib_dir}.')
    pose_dir = os.path.join(project_dir, 'pose')
    poseSync_dir = os.path.join(project_dir, 'pose-sync')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')
    
    # Projection matrix from toml calibration file
    P = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
        
    # Retrieve keypoints from model
    try: # from skeletons.py
        if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
        elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
        elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
        elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
        elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
        elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
        elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
        else: pass
        model = eval(pose_model)
    except:
        try: # from Config.toml
            model = DictImporter().import_(config_dict.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')
            
    keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    keypoints_idx = list(range(len(keypoints_ids)))
    keypoints_nb = len(keypoints_ids)
    # for pre, _, node in RenderTree(model): 
    #     print(f'{pre}{node.name} id={node.id}')
    
    # left/right swapped keypoints
    keypoints_names_swapped = ['L'+keypoint_name[1:] if keypoint_name.startswith('R') else 'R'+keypoint_name[1:] if keypoint_name.startswith('L') else keypoint_name for keypoint_name in keypoints_names]
    keypoints_names_swapped = [keypoint_name_swapped.replace('right', 'left') if keypoint_name_swapped.startswith('right') else keypoint_name_swapped.replace('left', 'right') if keypoint_name_swapped.startswith('left') else keypoint_name_swapped for keypoint_name_swapped in keypoints_names_swapped]
    keypoints_idx_swapped = [keypoints_names.index(keypoint_name_swapped) for keypoint_name_swapped in keypoints_names_swapped] # find index of new keypoint_name
    
    # 2d-pose files selection
    try:
        pose_listdirs_names = next(os.walk(pose_dir))[1]
        os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
    except:
        raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    n_cams = len(json_dirs_names)
    try: 
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseTracked_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        pose_dir = poseTracked_dir
    except:
        try: 
            json_files_names = [fnmatch.filter(os.listdir(os.path.join(poseSync_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            pose_dir = poseSync_dir
        except:
            try:
                json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
            except:
                raise Exception(f'No json files found in {pose_dir}, {poseSync_dir}, nor {poseTracked_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
    json_files_names = [sort_stringlist_by_last_number(js) for js in json_files_names]    

    # frame range selection
    f_range = [[0,min([len(j) for j in json_files_names])] if frame_range in ('all', 'auto', []) else frame_range][0]
    frame_nb = f_range[1] - f_range[0]
    
    # Check that camera number is consistent between calibration file and pose folders
    if n_cams != len(P):
        raise Exception(f'Error: The number of cameras is not consistent: Found {len(P)} cameras in the calibration file, and {n_cams} cameras based on the number of pose folders.')
    
    # Triangulation
    if multi_person:
        nb_persons_to_detect = max(max(count_persons_in_json(os.path.join(pose_dir, json_dirs_names[c], json_fname)) for json_fname in json_files_names[c]) for c in range(n_cams))
    else:
        nb_persons_to_detect = 1

    Q = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    Q_old = [[[np.nan]*3]*keypoints_nb for n in range(nb_persons_to_detect)]
    error = [[] for n in range(nb_persons_to_detect)]
    nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
    id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
    Q_tot, error_tot, nb_cams_excluded_tot, id_excluded_cams_tot, f_range_trimmed = [], [], [], [], []
    for f in tqdm(range(*f_range)):
        # print(f'\nFrame {f}:')        
        # Get x,y,likelihood values from files
        json_files_names_f = [[j for j in json_files_names[c] if int(re.split(r'(\d+)',j)[-2])==f] for c in range(n_cams)]
        json_files_names_f = [j for j_list in json_files_names_f for j in (j_list or ['none'])]
        json_files_f = [os.path.join(pose_dir, json_dirs_names[c], json_files_names_f[c]) for c in range(n_cams)]

        x_files, y_files, likelihood_files = extract_files_frame_f(json_files_f, keypoints_ids, nb_persons_to_detect)
        # [[[list of coordinates] * n_cams ] * nb_persons_to_detect]
        # vs. [[list of coordinates] * n_cams ] 
        
        # undistort points
        if undistort_points:
            for n in range(nb_persons_to_detect):
                points = [np.array(tuple(zip(x_files[n][i],y_files[n][i]))).reshape(-1, 1, 2).astype('float32') for i in range(n_cams)]
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                x_files[n] =  np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points])
                y_files[n] =  np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points])
                # This is good for slight distortion. For fisheye camera, the model does not work anymore. See there for an example https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L301

        # Replace likelihood by 0 if under likelihood_threshold
        with np.errstate(invalid='ignore'):
            for n in range(nb_persons_to_detect):
                x_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                y_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
                likelihood_files[n][likelihood_files[n] < likelihood_threshold] = np.nan
        
        # Q_old = Q except when it has nan, otherwise it takes the Q_old value
        nan_mask = np.isnan(Q)
        Q_old = np.where(nan_mask, Q_old, Q)
        Q = [[] for n in range(nb_persons_to_detect)]
        error = [[] for n in range(nb_persons_to_detect)]
        nb_cams_excluded = [[] for n in range(nb_persons_to_detect)]
        id_excluded_cams = [[] for n in range(nb_persons_to_detect)]
        
        for n in range(nb_persons_to_detect):
            for keypoint_idx in keypoints_idx:
            # keypoints_nb = 2
            # for keypoint_idx in range(2):
            # Triangulate cameras with min reprojection error
                # print('\n', keypoints_names[keypoint_idx])
                coords_2D_kpt = np.array( (x_files[n][:, keypoint_idx], y_files[n][:, keypoint_idx], likelihood_files[n][:, keypoint_idx]) )
                coords_2D_kpt_swapped = np.array(( x_files[n][:, keypoints_idx_swapped[keypoint_idx]], y_files[n][:, keypoints_idx_swapped[keypoint_idx]], likelihood_files[n][:, keypoints_idx_swapped[keypoint_idx]] ))

                Q_kpt, error_kpt, nb_cams_excluded_kpt, id_excluded_cams_kpt = triangulation_from_best_cameras(config_dict, coords_2D_kpt, coords_2D_kpt_swapped, P, calib_params) # P has been modified if undistort_points=True

                Q[n].append(Q_kpt)
                error[n].append(error_kpt)
                nb_cams_excluded[n].append(nb_cams_excluded_kpt)
                id_excluded_cams[n].append(id_excluded_cams_kpt)
        
        if multi_person:
            # reID persons across frames by checking the distance from one frame to another
            # print('Q before ordering ', np.array(Q)[:,:2])
            if f !=0:
                Q, associated_tuples = sort_people_sports2d(Q_old, Q)
                # Q, personsIDs_sorted, associated_tuples = sort_people(Q_old, Q)
                # print('Q after ordering ', personsIDs_sorted, associated_tuples, np.array(Q)[:,:2])
                
                error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted = [], [], []
                for i in range(len(Q)):
                    id_in_old =  associated_tuples[:,1][associated_tuples[:,0] == i].tolist()
                    if len(id_in_old) > 0:
                        # personsIDs_sorted += id_in_old
                        error_sorted += [error[id_in_old[0]]]
                        nb_cams_excluded_sorted += [nb_cams_excluded[id_in_old[0]]]
                        id_excluded_cams_sorted += [id_excluded_cams[id_in_old[0]]]
                    else:
                        # personsIDs_sorted += [-1]
                        error_sorted += [error[i]]
                        nb_cams_excluded_sorted += [nb_cams_excluded[i]]
                        id_excluded_cams_sorted += [id_excluded_cams[i]]
                error, nb_cams_excluded, id_excluded_cams = error_sorted, nb_cams_excluded_sorted, id_excluded_cams_sorted
        
        # TODO: if distance > threshold, new person
        
        # Add triangulated points, errors and excluded cameras to pandas dataframes
        Q_tot.append([np.concatenate(Q[n]) for n in range(nb_persons_to_detect)])
        error_tot.append([error[n] for n in range(nb_persons_to_detect)])
        nb_cams_excluded_tot.append([nb_cams_excluded[n] for n in range(nb_persons_to_detect)])
        id_excluded_cams = [[id_excluded_cams[n][k] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        id_excluded_cams_tot.append(id_excluded_cams)
            
    # fill values for if a person that was not initially detected has entered the frame 
    Q_tot = [list(tpl) for tpl in zip(*it.zip_longest(*Q_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    error_tot = [list(tpl) for tpl in zip(*it.zip_longest(*error_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    nb_cams_excluded_tot = [list(tpl) for tpl in zip(*it.zip_longest(*nb_cams_excluded_tot, fillvalue=[np.nan]*keypoints_nb*3))]
    id_excluded_cams_tot = [list(tpl) for tpl in zip(*it.zip_longest(*id_excluded_cams_tot, fillvalue=[np.nan]*keypoints_nb*3))]

    # dataframes for each person
    Q_tot = [pd.DataFrame([Q_tot_f[n] for Q_tot_f in Q_tot]) for n in range(nb_persons_to_detect)]
    error_tot = [pd.DataFrame([error_tot_f[n] for error_tot_f in error_tot]) for n in range(nb_persons_to_detect)]
    nb_cams_excluded_tot = [pd.DataFrame([nb_cams_excluded_tot_f[n] for nb_cams_excluded_tot_f in nb_cams_excluded_tot]) for n in range(nb_persons_to_detect)]
    id_excluded_cams_tot = [pd.DataFrame([id_excluded_cams_tot_f[n] for id_excluded_cams_tot_f in id_excluded_cams_tot]) for n in range(nb_persons_to_detect)]
    
    for n in range(nb_persons_to_detect):
        error_tot[n]['mean'] = error_tot[n].mean(axis=1,skipna=False)
        nb_cams_excluded_tot[n]['mean'] = nb_cams_excluded_tot[n].mean(axis=1)
    
    # Delete participants with less than 4 valid triangulated frames
    # for each person, for each keypoint, frames to interpolate
    zero_nan_frames = [np.where( Q_tot[n].iloc[:,::3].T.eq(0) | ~np.isfinite(Q_tot[n].iloc[:,::3].T) ) for n in range(nb_persons_to_detect)]
    zero_nan_frames_per_kpt = [[zero_nan_frames[n][1][np.where(zero_nan_frames[n][0]==k)[0]] for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
    non_nan_nb_first_kpt = [frame_nb - len(zero_nan_frames_per_kpt[n][0]) for n in range(nb_persons_to_detect)]
    deleted_person_id = [n for n in range(len(non_nan_nb_first_kpt)) if non_nan_nb_first_kpt[n]<4]

    Q_tot = [Q_tot[n] for n in range(len(Q_tot)) if n not in deleted_person_id]
    error_tot = [error_tot[n] for n in range(len(error_tot)) if n not in deleted_person_id]
    nb_cams_excluded_tot = [nb_cams_excluded_tot[n] for n in range(len(nb_cams_excluded_tot)) if n not in deleted_person_id]
    id_excluded_cams_tot = [id_excluded_cams_tot[n] for n in range(len(id_excluded_cams_tot)) if n not in deleted_person_id]
    nb_persons_to_detect = len(Q_tot)

    if nb_persons_to_detect == 0:
        raise Exception('No persons have been triangulated. Please check your calibration and your synchronization, or the triangulation parameters in Config.toml.')

    # import pickle
    # with open(os.path.join(session_dir, 'all.pkl'), 'wb') as f:
    #     pickle.dump([Q_tot, error_tot, nb_cams_excluded_tot, id_excluded_cams_tot, zero_nan_frames_per_kpt], f)
    ## with open(os.path.join(session_dir, 'all.pkl'), 'rb') as f:
    ##     Q_tot, error_tot, nb_cams_excluded_tot, id_excluded_cams_tot, zero_nan_frames_per_kpt = pickle.load(f)
    # Q_tot[0].to_csv(os.path.join(session_dir, 'Q_tot.csv'), index=False, sep='\t')
    # error_tot[0].to_csv(os.path.join(session_dir, 'error_tot.csv'), index=False, sep='\t')

    # Trim around good frames
    f_range_trimmed = [indices_of_first_last_non_nan_chunks(err['mean'], interp_gap_smaller_than) for err in error_tot]
    Q_tot = [Q_tot[n].iloc[f_range_trimmed[n][0]:f_range_trimmed[n][1]] for n in range(nb_persons_to_detect)]
    error_tot = [error_tot[n].iloc[f_range_trimmed[n][0]:f_range_trimmed[n][1]] for n in range(nb_persons_to_detect)]
    nb_cams_excluded_tot = [nb_cams_excluded_tot[n].iloc[f_range_trimmed[n][0]:f_range_trimmed[n][1]] for n in range(nb_persons_to_detect)]
    id_excluded_cams_tot = [id_excluded_cams_tot[n].iloc[f_range_trimmed[n][0]:f_range_trimmed[n][1]] for n in range(nb_persons_to_detect)]
    zero_nan_frames_per_kpt = [[z[(f_range_trimmed[n][0] < z) & (f_range_trimmed[n][1] > z)] for z in zero_nan_frames_per_kpt[n]] for n in range(nb_persons_to_detect)]

    # Interpolate missing values
    if interpolation_kind != 'none':
        for n in range(nb_persons_to_detect):
            try:
                Q_tot[n] = Q_tot[n].apply(interpolate_zeros_nans, axis=0, args=[interp_gap_smaller_than, interpolation_kind])
            except:
                logging.info(f'Interpolation was not possible for person {n}. This means that not enough points are available, which is often due to a bad calibration.')

    # Fill non-interpolated values with last valid one
    if fill_large_gaps_with == 'last_value':
        for n in range(nb_persons_to_detect): 
            Q_tot[n] = Q_tot[n].ffill(axis=0).bfill(axis=0)
    elif fill_large_gaps_with == 'zeros':
        for n in range(nb_persons_to_detect): 
            Q_tot[n].replace(np.nan, 0, inplace=True)

    # Create TRC file
    trc_paths = [make_trc(config_dict, Q_tot[n], keypoints_names, f_range_trimmed[n], id_person=n) for n in range(len(Q_tot))]
    if make_c3d:
        c3d_paths = [convert_to_c3d(t) for t in trc_paths]
        
    # # Reorder TRC files
    # if multi_person and reorder_trc and len(trc_paths)>1:
    #     trc_id = retrieve_right_trc_order(trc_paths)
    #     [os.rename(t, t+'.old') for t in trc_paths]
    #     [os.rename(t+'.old', trc_paths[i]) for i, t in zip(trc_id,trc_paths)]
    #     if make_c3d:
    #         [os.rename(c, c+'.old') for c in c3d_paths]
    #         [os.rename(c+'.old', c3d_paths[i]) for i, c in zip(trc_id,c3d_paths)]
    #     error_tot = [error_tot[i] for i in trc_id]
    #     nb_cams_excluded_tot = [nb_cams_excluded_tot[i] for i in trc_id]
    #     cam_excluded_count = [cam_excluded_count[i] for i in trc_id]
    #     interp_frames = [interp_frames[i] for i in trc_id]
    #     non_interp_frames = [non_interp_frames[i] for i in trc_id]
    #     logging.info('\nThe trc and c3d files have been renamed to match the order of the static sequences.')

    # IDs of excluded cameras
    # id_excluded_cams_tot = [np.concatenate([id_excluded_cams_tot[f][k] for f in range(frames_nb)]) for k in range(keypoints_nb)]
    id_excluded_cams_tot = [np.hstack(np.hstack(np.array(id_excluded_cams_tot[n]))) for n in range(nb_persons_to_detect)]
    cam_excluded_count = [dict(Counter(k)) for k in id_excluded_cams_tot]
    [cam_excluded_count[n].update((x, y/frame_nb/keypoints_nb) for x, y in cam_excluded_count[n].items()) for n in range(nb_persons_to_detect)]

    # Optionally, for each person, for each keypoint, show indices of frames that should be interpolated
    if show_interp_indices:
        gaps = [[np.where(np.diff(zero_nan_frames_per_kpt[n][k]) > 1)[0] + 1 for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        sequences = [[np.split(zero_nan_frames_per_kpt[n][k], gaps[n][k]) for k in range(keypoints_nb)] for n in range(nb_persons_to_detect)]
        interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)<=interp_gap_smaller_than and len(seq)>0] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
        non_interp_frames = [[[f'{seq[0]}:{seq[-1]}' for seq in seq_kpt if len(seq)>interp_gap_smaller_than] for seq_kpt in sequences[n]] for n in range(nb_persons_to_detect)]
    else:
        interp_frames = None
        non_interp_frames = []

    # Recap message
    recap_triangulate(config_dict, error_tot, nb_cams_excluded_tot, keypoints_names, cam_excluded_count, interp_frames, non_interp_frames, f_range_trimmed, trc_paths)
