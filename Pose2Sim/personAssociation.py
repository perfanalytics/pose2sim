#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    ###########################################################################
    ## TRACKING OF PERSON OF INTEREST                                        ##
    ###########################################################################
    
    Openpose detects all people in the field of view. 
    Which is the one of interest?
    
    This module tries all possible triangulations of a chosen anatomical 
    point, and chooses the person for whom the reprojection error is smallest.
    
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
from anytree import RenderTree
from anytree.importer import DictImporter
import logging

from Pose2Sim.common import computeP, weighted_triangulation, reprojection, \
    euclidean_distance, natural_sort
from Pose2Sim.skeletons import *


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = '0.4'
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
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


def best_persons_and_cameras_combination(config, json_files_framef, personsIDs_combinations, projection_matrices, tracked_keypoint_id):
    '''
    At the same time, chooses the right person among the multiple ones found by
    OpenPose & excludes cameras with wrong 2d-pose estimation.
    
    1. triangulate the tracked keypoint for all possible combinations of people,
    2. compute difference between reprojection & original openpose detection,
    3. take combination with smallest difference.
    If error is too big, take off one or several of the cameras until err is 
    lower than "max_err_px".
    
    INPUTS:
    - a Config.toml file
    - json_files_framef: list of strings
    - personsIDs_combinations: array, list of lists of int
    - projection_matrices: list of arrays
    - tracked_keypoint_id: int

    OUTPUTS:
    - error_min: float
    - persons_and_cameras_combination: array of ints
    '''
    
    error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
    min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
    likelihood_threshold = config.get('triangulation').get('likelihood_threshold')

    n_cams = len(json_files_framef)
    error_min = np.inf 
    nb_cams_off = 0 # cameras will be taken-off until the reprojection error is under threshold
    
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
            
            # Replace likelihood by 0. if under likelihood_threshold
            likelihood_files = [0. if lik < likelihood_threshold else lik for lik in likelihood_files]
            
            # For each persons combination, create subsets with "nb_cams_off" cameras excluded
            id_cams_off = list(it.combinations(range(len(combination)), nb_cams_off))
            combinations_with_cams_off = np.array([combination.copy()]*len(id_cams_off))
            for i, id in enumerate(id_cams_off):
                combinations_with_cams_off[i,id] = np.nan

            # Try all subsets
            error_comb = []
            for comb in combinations_with_cams_off:
                # Filter x, y, likelihood, projection_matrices, with subset
                x_files_filt = [x_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                y_files_filt = [y_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                likelihood_files_filt = [likelihood_files[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                projection_matrices_filt = [projection_matrices[i] for i in range(len(comb)) if not np.isnan(comb[i])]
                
                # Triangulate 2D points
                Q_comb = weighted_triangulation(projection_matrices_filt, x_files_filt, y_files_filt, likelihood_files_filt)
                
                # Reprojection
                x_calc, y_calc = reprojection(projection_matrices_filt, Q_comb)
                                
                # Reprojection error
                error_comb_per_cam = []
                for cam in range(len(x_calc)):
                    q_file = (x_files_filt[cam], y_files_filt[cam])
                    q_calc = (x_calc[cam], y_calc[cam])
                    error_comb_per_cam.append( euclidean_distance(q_file, q_calc) )
                error_comb.append( np.mean(error_comb_per_cam) )
            
            error_min = min(error_comb)
            persons_and_cameras_combination = combinations_with_cams_off[np.argmin(error_comb)]
            
            if error_min < error_threshold_tracking:
                break

        nb_cams_off += 1
    
    return error_min, persons_and_cameras_combination


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
    if project_dir == '': project_dir = os.getcwd()
    poseTracked_folder_name = config.get('project').get('poseAssociated_folder_name')
    calib_folder_name = config.get('project').get('calib_folder_name')
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
    error_threshold_tracking = config.get('personAssociation').get('error_threshold_tracking')
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name)
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    
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
    logging.info(f'\nTracked json files are stored in {poseTracked_dir}.')
    

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
    if project_dir == '': project_dir = os.getcwd()
    calib_folder_name = config.get('project').get('calib_folder_name')
    poseTracked_folder_name = config.get('project').get('poseAssociated_folder_name')
    pose_folder_name = config.get('project').get('pose_folder_name')
    pose_model = config.get('pose').get('pose_model')
    tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
    json_folder_extension =  config.get('project').get('pose_json_folder_extension')
    frame_range = config.get('project').get('frame_range')
    
    calib_dir = os.path.join(project_dir, calib_folder_name)
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0]
    pose_dir = os.path.join(project_dir, pose_folder_name)
    poseTracked_dir = os.path.join(project_dir, poseTracked_folder_name)

    # projection matrix from toml calibration file
    P = computeP(calib_file)
    
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
    json_dirs_names = [k for k in pose_listdirs_names if json_folder_extension in k]
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
    
    for f in tqdm(range(*f_range)):
        json_files_f = [json_files[c][f] for c in range(n_cams)]
        json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]
        
        # all possible combinations of persons
        personsIDs_comb = persons_combinations(json_files_f) 
        
        # choose person of interest and exclude cameras with bad pose estimation
        error_min, persons_and_cameras_combination = best_persons_and_cameras_combination(config, json_files_f, personsIDs_comb, P, tracked_keypoint_id)
        error_min_tot.append(error_min)
        cameras_off_count = np.count_nonzero(np.isnan(persons_and_cameras_combination))
        cameras_off_tot.append(cameras_off_count)
        
        # rewrite json files with only one person of interest
        for cam_nb, person_id in enumerate(persons_and_cameras_combination):
            with open(json_tracked_files_f[cam_nb], 'w') as json_tracked_f:
                with open(json_files_f[cam_nb], 'r') as json_f:
                    js = json.load(json_f)
                    if not np.isnan(person_id):
                        js['people'] = [js['people'][int(person_id)]]
                    else: 
                        js['people'] = []
                json_tracked_f.write(json.dumps(js))

    # recap message
    recap_tracking(config, error_min_tot, cameras_off_tot)
    
    
    
    
