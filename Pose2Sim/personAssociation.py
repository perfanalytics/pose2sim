#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## TRACKING OF PERSON OF INTEREST                                        ##
    ###########################################################################
    
    Openpose detects all people in the field of view. 
    Which is the one of interest?
    
    This module tries all possible triangulations of a chosen anatomical 
    point. If "multi_person" mode is not used, it chooses the person for
    whom the reprojection error is smallest. Otherwise, it selects all 
    persons with a reprojection error smaller than a threshold, and then 
    associates them across time frames by minimizing the displacement speed.
    
    INPUTS: 
    - a calibration file (.toml extension)
    - json files from each camera folders with several detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - json files for each camera with only one person of interest
    
'''


## INIT
import os
import glob
import fnmatch
import numpy as np
import json
import itertools as it
import toml
from tqdm import tqdm
import cv2
from anytree import RenderTree
from anytree.importer import DictImporter
import logging

from Pose2Sim.common import retrieve_calib_params, computeP, weighted_triangulation, \
    reprojection, euclidean_distance, natural_sort
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.6'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def common_items_in_list(list1, list2):
    '''
    Do two lists have any items in common at the same index?
    Returns True or False
    '''
    
    for i, j in enumerate(list1):
        if j == list2[i]:
            return True
    return False
    

def min_with_single_indices(L, T):
    '''
    Let L be a list (size s) with T associated tuple indices (size s).
    Select the smallest values of L, considering that 
    the next smallest value cannot have the same numbers 
    in the associated tuple as any of the previous ones.

    Example:
    L = [  20,   27,  51,    33,   43,   23,   37,   24,   4,   68,   84,    3  ]
    T = list(it.product(range(2),range(3)))
      = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(2,0),(2,1),(2,2),(2,3)]

    - 1st smallest value: 3 with tuple (2,3), index 11
    - 2nd smallest value when excluding indices (2,.) and (.,3), i.e. [(0,0),(0,1),(0,2),X,(1,0),(1,1),(1,2),X,X,X,X,X]:
    20 with tuple (0,0), index 0
    - 3rd smallest value when excluding [X,X,X,X,X,(1,1),(1,2),X,X,X,X,X]:
    23 with tuple (1,1), index 5
    
    INPUTS:
    - L: list (size s)
    - T: T associated tuple indices (size s)

    OUTPUTS: 
    - minL: list of smallest values of L, considering constraints on tuple indices
    - argminL: list of indices of smallest values of L
    - T_minL: list of tuples associated with smallest values of L
    '''

    minL = [np.min(L)]
    argminL = [np.argmin(L)]
    T_minL = [T[argminL[0]]]
    
    mask_tokeep = np.array([True for t in T])
    i=0
    while mask_tokeep.any()==True:
        mask_tokeep = mask_tokeep & np.array([t[0]!=T_minL[i][0] and t[1]!=T_minL[i][1] for t in T])
        if mask_tokeep.any()==True:
            indicesL_tokeep = np.where(mask_tokeep)[0]
            minL += [np.min(np.array(L)[indicesL_tokeep])]
            argminL += [indicesL_tokeep[np.argmin(np.array(L)[indicesL_tokeep])]]
            T_minL += (T[argminL[i+1]],)
            i+=1
    
    return minL, argminL, T_minL
    
    
def sort_people(Q_kpt_old, Q_kpt):
    '''
    Associate persons across frames
    Persons' indices are sometimes swapped when changing frame
    A person is associated to another in the next frame when they are at a small distance
    
    INPUTS:
    - Q_kpt_old: list of arrays of 3D coordinates [X, Y, Z, 1.] for the previous frame
    - Q_kpt: idem Q_kpt_old, for current frame
    
    OUTPUT:
    - Q_kpt: array with reordered persons
    - personsIDs_sorted: index of reordered persons
    '''
    
    # Generate possible person correspondences across frames
    if len(Q_kpt_old) < len(Q_kpt):
        Q_kpt_old = np.concatenate((Q_kpt_old, [[0., 0., 0., 1.]]*(len(Q_kpt)-len(Q_kpt_old))))
    personsIDs_comb = sorted(list(it.product(range(len(Q_kpt_old)),range(len(Q_kpt)))))
    # Compute distance between persons from one frame to another
    frame_by_frame_dist = []
    for comb in personsIDs_comb:
        frame_by_frame_dist += [euclidean_distance(Q_kpt_old[comb[0]][:3],Q_kpt[comb[1]][:3])]
    # sort correspondences by distance
    _, index_best_comb, _ = min_with_single_indices(frame_by_frame_dist, personsIDs_comb)
    index_best_comb.sort()
    personsIDs_sorted = np.array(personsIDs_comb)[index_best_comb][:,1]
    # rearrange persons
    Q_kpt = np.array(Q_kpt)[personsIDs_sorted]
    
    return Q_kpt, personsIDs_sorted


def persons_combinations(json_files_framef):
    '''
    Find all possible combinations of detected persons' ids. 
    Person's id when no person detected is set to -1.
    
    INPUT:
    - json_files_framef: list of strings

    OUTPUT:
    - personsIDs_comb: array, list of lists of int
    '''
    
    n_cams = len(json_files_framef)
    
    # amount of persons detected for each cam
    nb_persons_per_cam = []
    for c in range(n_cams):
        with open(json_files_framef[c], 'r') as js:
            nb_persons_per_cam += [len(json.load(js)['people'])]
    
    # persons_combinations
    id_no_detect = [i for i, x in enumerate(nb_persons_per_cam) if x == 0]  # ids of cameras that have not detected any person
    nb_persons_per_cam = [x if x != 0 else 1 for x in nb_persons_per_cam] # temporarily replace persons count by 1 when no detection
    range_persons_per_cam = [range(nb_persons_per_cam[c]) for c in range(n_cams)] 
    personsIDs_comb = np.array(list(it.product(*range_persons_per_cam)), float) # all possible combinations of persons' ids
    personsIDs_comb[:,id_no_detect] = np.nan # -1 = persons' ids when no person detected
    
    return personsIDs_comb


def best_persons_and_cameras_combination(config, json_files_framef, personsIDs_combinations, projection_matrices, tracked_keypoint_id, calib_params):
    '''
    - if multi_person: Choose all the combination of cameras that give a reprojection error below a threshold
    - else: Chooses the right person among the multiple ones found by
    OpenPose & excludes cameras with wrong 2d-pose estimation.
    
    1. triangulate the tracked keypoint for all possible combinations of people,
    2. compute difference between reprojection & original openpose detection,
    3. take combination with smallest error OR all those below the error threshold
    If error is too big, take off one or several of the cameras until err is 
    lower than "max_err_px".
    
    INPUTS:
    - a Config.toml file
    - json_files_framef: list of strings
    - personsIDs_combinations: array, list of lists of int
    - projection_matrices: list of arrays
    - tracked_keypoint_id: int

    OUTPUTS:
    - errors_below_thresh: list of float
    - comb_errors_below_thresh: list of arrays of ints
    '''
    
    multi_person = config.get('project').get('multi_person')
    nb_persons_to_detect = config.get('project').get('nb_persons_to_detect')
    error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
    likelihood_threshold = config.get('personAssociation').get('likelihood_threshold_association')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    undistort_points = config.get('triangulation').get('undistort_points')

    n_cams = len(json_files_framef)
    error_min = np.inf 
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    errors_below_thresh = []
    comb_errors_below_thresh = []
    Q_kpt = []
    while error_min > error_threshold_tracking and n_cams - nb_cams_off >= min_cameras_for_triangulation:
        # Try all persons combinations
        for combination in personsIDs_combinations:
            # Get x,y,likelihood values from files
            x_files, y_files,likelihood_files = [], [], []
            for index_cam, person_nb in enumerate(combination):
                with open(json_files_framef[index_cam], 'r') as json_f:
                    js = json.load(json_f)
                    try:
                        x_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3] )
                        y_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3+1] )
                        likelihood_files.append( js['people'][int(person_nb)]['pose_keypoints_2d'][tracked_keypoint_id*3+2] )
                    except:
                        x_files.append(np.nan)
                        y_files.append(np.nan)
                        likelihood_files.append(np.nan)
            
            # undistort points
            if undistort_points:
                points = np.array(tuple(zip(x_files,y_files))).reshape(-1, 1, 2).astype('float32')
                undistorted_points = [cv2.undistortPoints(points[i], calib_params['K'][i], calib_params['dist'][i], None, calib_params['optim_K'][i]) for i in range(n_cams)]
                x_files = np.array([[u[i][0][0] for i in range(len(u))] for u in undistorted_points]).squeeze()
                y_files = np.array([[u[i][0][1] for i in range(len(u))] for u in undistorted_points]).squeeze()
            
            # Replace likelihood by 0. if under likelihood_threshold
            likelihood_files = [0. if lik < likelihood_threshold else lik for lik in likelihood_files]
            
            # For each persons combination, create subsets with "nb_cams_off" cameras excluded
            id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off))
            combinations_with_cams_off = np.array([combination.copy()]*len(id_cams_off))
            for i, id in enumerate(id_cams_off):
                combinations_with_cams_off[i,id] = np.nan

            # Try all subsets
            error_comb = []
            Q_comb = []
            for comb in combinations_with_cams_off:
                # Filter x, y, likelihood, projection_matrices, with subset
                x_files_filt = [x_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                y_files_filt = [y_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                likelihood_files_filt = [likelihood_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                projection_matrices_filt = [projection_matrices[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                if undistort_points:
                    calib_params_R_filt = [calib_params['R'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
                    calib_params_T_filt = [calib_params['T'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
                    calib_params_K_filt = [calib_params['K'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
                    calib_params_dist_filt = [calib_params['dist'][i] for i in range(len(comb)) if not np.isnan(comb[i])]
                
                # Triangulate 2D points
                Q_comb.append(weighted_triangulation(projection_matrices_filt, x_files_filt, y_files_filt, likelihood_files_filt))
                
                # Reprojection
                if undistort_points:
                    coords_2D_kpt_calc_filt = [cv2.projectPoints(np.array(Q_comb[-1][:-1]), calib_params_R_filt[i], calib_params_T_filt[i], calib_params_K_filt[i], calib_params_dist_filt[i])[0] for i in range(n_cams-nb_cams_off)]
                    x_calc = [coords_2D_kpt_calc_filt[i][0,0,0] for i in range(n_cams-nb_cams_off)]
                    y_calc = [coords_2D_kpt_calc_filt[i][0,0,1] for i in range(n_cams-nb_cams_off)]
                else:
                    x_calc, y_calc = reprojection(projection_matrices_filt, Q_comb[-1])
                                                
                # Reprojection error
                error_comb_per_cam = []
                for cam in range(len(x_calc)):
                    q_file = (x_files_filt[cam], y_files_filt[cam])
                    q_calc = (x_calc[cam], y_calc[cam])
                    error_comb_per_cam.append( euclidean_distance(q_file, q_calc) )
                error_comb.append( np.mean(error_comb_per_cam) )
            
            if multi_person:
                errors_below_thresh += [e for e in error_comb if e<error_threshold_tracking]
                comb_errors_below_thresh += [combinations_with_cams_off[error_comb.index(e)] for e in error_comb if e<error_threshold_tracking]
                Q_kpt += [Q_comb[error_comb.index(e)] for e in error_comb if e<error_threshold_tracking]
            else:
                error_min = np.nanmin(error_comb)
                errors_below_thresh = [error_min]
                comb_errors_below_thresh = [combinations_with_cams_off[np.argmin(error_comb)]]
                Q_kpt = [Q_comb[np.argmin(error_comb)]]
                if errors_below_thresh[0] < error_threshold_tracking:
                    break 
                
        if multi_person:
            if len(errors_below_thresh)>0:
                # sort combinations by error magnitude
                errors_below_thresh_sorted = sorted(errors_below_thresh)
                sorted_idx = np.array([errors_below_thresh.index(e) for e in errors_below_thresh_sorted])
                comb_errors_below_thresh = np.array(comb_errors_below_thresh)[sorted_idx]
                Q_kpt = np.array(Q_kpt)[sorted_idx]
                # remove combinations with indices used several times for the same person 
                comb_errors_below_thresh = [c.tolist() for c in comb_errors_below_thresh]
                comb = comb_errors_below_thresh.copy()
                comb_ok = np.array([comb[0]])
                for i, c1 in enumerate(comb):
                    idx_ok = np.array([not(common_items_in_list(c1, c2)) for c2 in comb[1:]])
                    try:
                        comb = np.array(comb[1:])[idx_ok]
                        comb_ok = np.concatenate((comb_ok, [comb[0]]))
                    except:
                        break
                sorted_pruned_idx = [i for i, x in enumerate(comb_errors_below_thresh) for c in comb_ok if np.array_equal(x,c,equal_nan=True)]
                errors_below_thresh = np.array(errors_below_thresh_sorted)[sorted_pruned_idx].tolist()
                comb_errors_below_thresh = np.array(comb_errors_below_thresh)[sorted_pruned_idx].tolist()
                Q_kpt = Q_kpt[sorted_pruned_idx].tolist()

            # Remove indices already used for a person
            personsIDs_combinations = np.array([personsIDs_combinations[i] for i in range(len(personsIDs_combinations))
                                           if not np.array( 
                                                [personsIDs_combinations[i,j]==comb[j] for comb in comb_errors_below_thresh for j in range(len(comb))]
                                                ).any()])
            if len(errors_below_thresh) >= len(personsIDs_combinations) or len(errors_below_thresh) >= nb_persons_to_detect: 
                errors_below_thresh = errors_below_thresh[:nb_persons_to_detect]
                comb_errors_below_thresh = comb_errors_below_thresh[:nb_persons_to_detect]
                Q_kpt = Q_kpt[:nb_persons_to_detect]
                break

        nb_cams_off += 1
    
    return errors_below_thresh, comb_errors_below_thresh, Q_kpt


def recap_tracking(config, error, nb_cams_excluded):
    '''
    Print a message giving statistics on reprojection errors (in pixel and in m)
    as well as the number of cameras that had to be excluded to reach threshold
    conditions. Also stored in User/logs.txt.

    INPUT:
    - a Config.toml file
    - error: dataframe 
    - nb_cams_excluded: dataframe

    OUTPUT:
    - Message in console
    '''
    
    # Read config
    project_dir = config.get('project').get('project_dir')
    session_dir = os.path.realpath(os.path.join(project_dir, '..', '..'))
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
    error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if ('Calib' or 'calib') in c][0]
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    
    # Error
    mean_error_px = np.around(np.mean(error), decimals=1)
    
    calib = toml.load(calib_file)
    calib_cam1 = calib[list(calib.keys())[0]]
    fm = calib_cam1['matrix'][0][0]
    Dm = euclidean_distance(calib_cam1['translation'], [0,0,0])
    mean_error_mm = np.around(mean_error_px * Dm / fm * 1000, decimals=1)
    
    # Excluded cameras
    mean_cam_off_count = np.around(np.mean(nb_cams_excluded), decimals=2)

    # Recap
    logging.info(f'\n--> Mean reprojection error for {tracked_keypoint} point on all frames is {mean_error_px} px, which roughly corresponds to {mean_error_mm} mm. ')
    logging.info(f'--> In average, {mean_cam_off_count} cameras had to be excluded to reach the demanded {error_threshold_tracking} px error threshold.')
    logging.info(f'\nTracked json files are stored in {os.path.realpath(poseTracked_dir)}.')
    

def track_2d_all(config):
    '''
    For each frame,
    - Find all possible combinations of detected persons
    - Triangulate 'tracked_keypoint' for all combinations
    - Reproject the point on all cameras
    - Take combination with smallest reprojection error
    - Write json file with only one detected person
    Print recap message
    
    INPUTS: 
    - a calibration file (.toml extension)
    - json files from each camera folders with several detected persons
    - a Config.toml file
    - a skeleton model
    
    OUTPUTS: 
    - json files for each camera with only one person of interest    
    '''
    
    # Read config
    project_dir = config.get('project').get('project_dir')
    session_dir = os.path.realpath(os.path.join(project_dir, '..', '..'))
    multi_person = config.get('project').get('multi_person')
    pose_model = config.get('pose').get('pose_model')
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
    frame_range = config.get('project').get('frame_range')
    undistort_points = config.get('triangulation').get('undistort_points')
    
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if ('Calib' or 'calib') in c][0]
    try:
        calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
    except:
        raise Exception(f'No .toml calibration file found in the {calib_dir}.')
    pose_dir = os.path.join(project_dir, 'pose')
    poseTracked_dir = os.path.join(project_dir, 'pose-associated')

    if multi_person:
        logging.info('\nMulti-person analysis selected. Note that you can set this option to false for faster runtime if you only need the main person in the scene.')
    else:
        logging.info('\nSingle-person analysis selected.')

    # projection matrix from toml calibration file
    P = computeP(calib_file, undistort=undistort_points)
    calib_params = retrieve_calib_params(calib_file)
        
    # selection of tracked keypoint id
    try: # from skeletons.py
        model = eval(pose_model)
    except:
        try: # from Config.toml
            model = DictImporter().import_(config.get('pose').get(pose_model))
            if model.id == 'None':
                model.id = None
        except:
            raise NameError('Model not found in skeletons.py nor in Config.toml')
    tracked_keypoint_id = [node.id for _, _, node in RenderTree(model) if node.name==tracked_keypoint][0]
    
    # 2d-pose files selection
    pose_listdirs_names = next(os.walk(pose_dir))[1]
    pose_listdirs_names = natural_sort(pose_listdirs_names)
    json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
    json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
    json_files_names = [natural_sort(j) for j in json_files_names]
    json_files = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    # 2d-pose-associated files creation
    if not os.path.exists(poseTracked_dir): os.mkdir(poseTracked_dir)   
    try: [os.mkdir(os.path.join(poseTracked_dir,k)) for k in json_dirs_names]
    except: pass
    json_tracked_files = [[os.path.join(poseTracked_dir, j_dir, j_file) for j_file in json_files_names[j]] for j, j_dir in enumerate(json_dirs_names)]
    
    # person's tracking
    f_range = [[min([len(j) for j in json_files])] if frame_range==[] else frame_range][0]
    n_cams = len(json_dirs_names)
    error_min_tot, cameras_off_tot = [], []
    
    # Check that camera number is consistent between calibration file and pose folders
    if n_cams != len(P):
        raise Exception(f'Error: The number of cameras is not consistent:\
                    Found {len(P)} cameras in the calibration file,\
                    and {n_cams} cameras based on the number of pose folders.')
    
    Q_kpt = [np.array([0., 0., 0., 1.])]
    for f in tqdm(range(*f_range)):
        # print(f'\nFrame {f}:')
        json_files_f = [json_files[c][f] for c in range(n_cams)]
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        
        # all possible combinations of persons
        personsIDs_comb = persons_combinations(json_files_f) 
        
        # choose persons of interest and exclude cameras with bad pose estimation
        Q_kpt_old = Q_kpt
        errors_below_thresh, comb_errors_below_thresh, Q_kpt = best_persons_and_cameras_combination(config, json_files_f, personsIDs_comb, P, tracked_keypoint_id, calib_params)
        
        # reID persons across frames by checking the distance from one frame to another
        Q_kpt, personsIDs_sorted = sort_people(Q_kpt_old, Q_kpt)
        errors_below_thresh = np.array(errors_below_thresh)[personsIDs_sorted]
        comb_errors_below_thresh = np.array(comb_errors_below_thresh)[personsIDs_sorted]
        
        # rewrite json files with a single or multiple persons of interest
        error_min_tot.append(np.mean(errors_below_thresh))
        cameras_off_count = np.count_nonzero([np.isnan(comb) for comb in comb_errors_below_thresh]) / len(comb_errors_below_thresh)
        cameras_off_tot.append(cameras_off_count)
        for cam in range(n_cams):
            with open(json_tracked_files_f[cam], 'w') as json_tracked_f:
                with open(json_files_f[cam], 'r') as json_f:
                    js = json.load(json_f)
                    js_new = js.copy()
                    js_new['people'] = []
                    for new_comb in comb_errors_below_thresh:
                        if not np.isnan(new_comb[cam]):
                            js_new['people'] += [js['people'][int(new_comb[cam])]]
                        else:
                            js_new['people'] += [{}]
                json_tracked_f.write(json.dumps(js_new))

    # recap message
    recap_tracking(config, error_min_tot, cameras_off_tot)
    
