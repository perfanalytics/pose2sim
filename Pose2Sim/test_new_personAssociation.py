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

config = toml.load(r'D:\data\forceteck\Rugby_Lab_2022_02\Config.toml')
project_dir = r'D:\data\forceteck\Rugby_Lab_2022_02\tackle_zak_01\tackle_zak_01_tackle'

session_dir = os.path.realpath(os.path.join(project_dir, '..', '..'))
multi_person = config.get('project').get('multi_person')
pose_model = config.get('pose').get('pose_model')
tracked_keypoint = config.get('personAssociation').get('tracked_keypoint')
frame_range = config.get('project').get('frame_range')
undistort_points = config.get('triangulation').get('undistort_points')

calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if 'calib' in c.lower() ][0]
try:
    calib_file = glob.glob(os.path.join(calib_dir, '*.toml'))[0] # lastly created calibration file
except:
    raise Exception(f'No .toml calibration file found in the {calib_dir}.')
pose_dir = os.path.join(project_dir, 'pose')
poseTracked_dir = os.path.join(project_dir, 'pose-associated')

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
f=3
n_cams_off = 0

json_files_f = [json_files[c][f] for c in range(n_cams)]
json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]


Q_kpt_old = Q_kpt
error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
likelihood_threshold = config.get('personAssociation').get('likelihood_threshold_association')
min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
n_cams = len(json_files_f)


P_all = computeP(calib_file)

reconstruction_error_threshold = 0.1 # 0.1 = 10 cm
min_affinity = 0.2

'''
The new (March 2024) multi-person algorithm is largely inspired from Dong et al (2022) 
"Fast and Robust Multi-Person 3D Pose Estimation and Tracking From Multiple Views"
https://ieeexplore.ieee.org/document/9492024
https://github.com/zju3dv/mvpose https://github.com/zju3dv/EasyMocap

'''

def read_json(js_file):
    '''
    Read OpenPose json file
    '''
    with open(js_file, 'r') as json_f:
        js = json.load(json_f)
        json_data = []
        for people in range(len(js['people'])):
            if len(js['people'][people]['pose_keypoints_2d']) < 3: continue
            else:
                json_data.append(js['people'][people]['pose_keypoints_2d'])
    return json_data


def compute_rays(json_coord, calib_params, cam_id):
    '''
    Plucker coordinates of rays from camera to each joint of a person
    Plucker coordinates: camera to keypoint line direction (size 3) 
                         moment: origin ^ line (size 3)
                         additionally, confidence

    INPUTS:
    - json_coord: x, y, likelihood for a person seen from a camera (list of 3*joint_nb)
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cam_id: camera id (int)

    OUTPUT:
    - plucker: array. nb joints * (6 plucker coordinates + 1 likelihood)
    '''

    x = json_coord[0::3]
    y = json_coord[1::3]
    likelihood = json_coord[2::3]
    
    inv_K = calib_params['inv_K'][cam_id]
    R_mat = calib_params['R_mat'][cam_id]
    T = calib_params['T'][cam_id]

    cam_center = -R_mat.T @ T
    plucker = []
    for i in range(len(x)):
        q = np.array([x[i], y[i], 1])
        norm_Q = R_mat.T @ (inv_K @ q -T)
        
        line = norm_Q - cam_center
        norm_line = line/np.linalg.norm(line)
        moment = np.cross(cam_center, norm_line)
        plucker.append(np.concatenate([norm_line, moment, [likelihood[i]]]))

    return np.array(plucker)


def broadcast_line_to_line_distance(p0, p1):
    '''
    Compute the distance between two lines in 3D space.

    see: https://faculty.sites.iastate.edu/jia/files/inline-files/plucker-coordinates.pdf
    p0 = (l0,m0), p1 = (l1,m1)
    dist = | (l0,m0) * (l1,m1) | / || l0 x l1 ||
    (l0,m0) * (l1,m1) = l0 @ m1 + m0 @ l1 (reciprocal product)
    
    No need to divide by the norm of the cross product of the directions, since we
    don't need the actual distance but whether the lines are close to intersecting or not
    => dist = | (l0,m0) * (l1,m1) |

    INPUTS:
    - p0: array(nb_persons_detected * 1 * nb_joints * 7 coordinates)
    - p1: array(1 * nb_persons_detected * nb_joints * 7 coordinates)

    OUTPUT:
    - dist: distances between the two lines (not normalized). 
            array(nb_persons_0 * nb_persons_1 * nb_joints)
    '''

    product = np.sum(p0[..., :3] * p1[..., 3:6], axis=-1) + np.sum(p1[..., :3] * p0[..., 3:6], axis=-1)
    dist = np.abs(product)

    return dist


def compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=0.1):
    '''
    Compute the affinity between all the people in the different views.

    The affinity is defined as 1 - distance/max_distance, with distance the
    distance between epipolar lines in each view (reciprocal product of Plucker 
    coordinates).

    Another approach would be to project one epipolar line onto the other camera
    plane and compute the line to point distance, but it is more computationally 
    intensive (simple dot product vs. projection and distance calculation). 
    
    INPUTS:
    - all_json_data_f: list of json data. For frame f, nb_views*nb_persons*(x,y,likelihood)*nb_joints
    - calib_params: calibration parameters from retrieve_calib_params('calib.toml')
    - cum_persons_per_view: cumulative number of persons per view
    - reconstruction_error_threshold: maximum distance between epipolar lines to consider a match

    OUTPUT:
    - affinity: affinity matrix between all the people in the different views. 
                (nb_views*nb_persons_per_view * nb_views*nb_persons_per_view)
    '''

    # Compute plucker coordinates for all keypoints for each person in each view
    # pluckers_f: dims=(camera, person, joint, 7 coordinates)
    pluckers_f = []
    for cam_id, json_cam  in enumerate(all_json_data_f):
        pluckers = []
        for json_coord in json_cam:
            plucker = compute_rays(json_coord, calib_params, cam_id) # LIMIT TO 15 JOINTS? json_coord[:15*3]
            pluckers.append(plucker)
        pluckers = np.array(pluckers)
        pluckers_f.append(pluckers)

    # Compute affinity matrix
    distance = np.zeros((cum_persons_per_view[-1], cum_persons_per_view[-1])) + 2*reconstruction_error_threshold
    for compared_cam0, compared_cam1 in it.combinations(range(len(all_json_data_f)), 2):
        # skip when no detection for a camera
        if cum_persons_per_view[compared_cam0] == cum_persons_per_view[compared_cam0+1] \
            or cum_persons_per_view[compared_cam1] == cum_persons_per_view[compared_cam1 +1]:
            continue

        # compute distance
        p0 = pluckers_f[compared_cam0][:,None] # add coordinate on second dimension
        p1 = pluckers_f[compared_cam1][None,:] # add coordinate on first dimension
        dist = broadcast_line_to_line_distance(p0, p1)
        likelihood = np.sqrt(p0[..., -1] * p1[..., -1])
        mean_weighted_dist = np.sum(dist*likelihood, axis=-1)/(1e-5 + likelihood.sum(axis=-1)) # array(nb_persons_0 * nb_persons_1)
        
        # populate distance matrix
        distance[cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1], \
                 cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1]] \
                 = mean_weighted_dist
        distance[cum_persons_per_view[compared_cam1]:cum_persons_per_view[compared_cam1+1], \
                 cum_persons_per_view[compared_cam0]:cum_persons_per_view[compared_cam0+1]] \
                 = mean_weighted_dist.T

    # compute affinity matrix and clamp it to zero when distance > reconstruction_error_threshold
    distance[distance > reconstruction_error_threshold] = reconstruction_error_threshold
    affinity = 1 - distance / reconstruction_error_threshold

    return affinity


def circular_constraint(cum_persons_per_view):
    '''
    A person can be matched only with themselves in the same view, and with any 
    person from other views

    INPUT:
    - cum_persons_per_view: cumulative number of persons per view

    OUTPUT:
    - circ_constraint: circular constraint matrix
    '''

    circ_constraint = np.identity(cum_persons_per_view[-1])
    for i in range(len(cum_persons_per_view)-1):
        circ_constraint[cum_persons_per_view[i]:cum_persons_per_view[i+1], cum_persons_per_view[i+1]:cum_persons_per_view[-1]] = 1
        circ_constraint[cum_persons_per_view[i+1]:cum_persons_per_view[-1], cum_persons_per_view[i]:cum_persons_per_view[i+1]] = 1
    
    return circ_constraint


def SVT(matrix, threshold):
    '''
    Find a low-rank approximation of the matrix using Singular Value Thresholding.

    INPUTS:
    - matrix: matrix to decompose
    - threshold: threshold for singular values

    OUTPUT:
    - matrix_thresh: low-rank approximation of the matrix
    '''
    
    U, s, Vt = np.linalg.svd(matrix) # decompose matrix
    s_thresh = np.maximum(s - threshold, 0) # set smallest singular values to zero
    matrix_thresh = U @ np.diag(s_thresh) @ Vt # recompose matrix

    return matrix_thresh


def matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1):
    '''
    Find low-rank approximation of 'affinity' while satisfying the circular constraint.

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - cum_persons_per_view: cumulative number of persons per view
    - circ_constraint: circular constraint matrix
    - max_iter: maximum number of iterations
    - w_rank: threshold for singular values
    - tol: tolerance for convergence
    - w_sparse: regularization parameter

    OUTPUT:
    - new_aff: low-rank approximation of the affinity matrix
    '''

    new_aff = affinity.copy()
    N = new_aff.shape[0]
    index_diag = np.arange(N)
    new_aff[index_diag, index_diag] = 0.
    # new_aff = (new_aff + new_aff.T)/2 # symmetric by construction

    Y = np.zeros_like(new_aff) # Initial deviation matrix / residual ()
    W = w_sparse - new_aff # Initial sparse matrix / regularization (prevent overfitting)
    mu = 64 # initial step size

    for iter in range(max_iter):
        new_aff0 = new_aff.copy()
        
        Q = new_aff + Y*1.0/mu
        Q = SVT(Q,w_rank/mu)
        new_aff = Q - (W + Y)/mu

        # Project X onto dimGroups
        for i in range(len(cum_persons_per_view) - 1):
            ind1, ind2 = cum_persons_per_view[i], cum_persons_per_view[i + 1]
            new_aff[ind1:ind2, ind1:ind2] = 0
            
        # Reset diagonal elements to one and ensure X is within valid range [0, 1]
        new_aff[index_diag, index_diag] = 1.
        new_aff[new_aff < 0] = 0
        new_aff[new_aff > 1] = 1
        
        # Enforce circular constraint
        new_aff = new_aff * circ_constraint
        new_aff = (new_aff + new_aff.T) / 2 # kept just in case X loses its symmetry during optimization 
        Y = Y + mu * (new_aff - Q)
        
        # Compute convergence criteria: break if new_aff is close enough to Q and no evolution anymore
        pRes = np.linalg.norm(new_aff - Q) / N # primal residual (diff between new_aff and SVT result)
        dRes = mu * np.linalg.norm(new_aff - new_aff0) / N # dual residual (diff between new_aff and previous new_aff)
        if pRes < tol and dRes < tol:
            break
        if pRes > 10 * dRes: mu = 2 * mu
        elif dRes > 10 * pRes: mu = mu / 2

        iter +=1

    return new_aff


def person_index_per_cam(affinity, min_cameras_for_triangulation):
    '''
    For each detected person, gives their index for each camera

    INPUTS:
    - affinity: affinity matrix between all the people in the different views
    - min_cameras_for_triangulation: exclude proposals if less than N cameras see them

    OUTPUT:
    - proposals: 2D array: n_persons * n_cams
    '''

    # index of the max affinity for each group (-1 if no detection)
    proposals = []
    for row in range(affinity.shape[0]):
        proposal_row = []
        for cam in range(n_cams):
            id_persons_per_view = affinity[row, cum_persons_per_view[cam]:cum_persons_per_view[cam+1]]
            proposal_row += [np.argmax(id_persons_per_view) if (len(id_persons_per_view)>0 and max(id_persons_per_view)>0) else -1]
        proposals.append(proposal_row)
    proposals = np.array(proposals, dtype=float)

    # remove duplicates and order
    proposals, nb_detections = np.unique(proposals, axis=0, return_counts=True)
    proposals = proposals[np.argsort(nb_detections)[::-1]]

    # remove row if any value is the same in previous rows at same index (nan!=nan so nan ignored)
    proposals[proposals==-1] = np.nan
    mask = np.ones(proposals.shape[0], dtype=bool)
    for i in range(1, len(proposals)):
        mask[i] = ~np.any(proposals[i] == proposals[:i], axis=0).any()
    proposals = proposals[mask]

    # remove identifications if less than N cameras see them
    nb_cams_per_person = [np.count_nonzero(~np.isnan(p)) for p in proposals]
    proposals = np.array([p for (n,p) in zip(nb_cams_per_person, proposals) if n >= min_cameras_for_triangulation])

    return proposals






config = toml.load(r'D:\data\forceteck\Rugby_Lab_2022_02\Config.toml')
project_dir = r'D:\data\forceteck\Rugby_Lab_2022_02\tackle_zak_01\tackle_zak_01_tackle'
# 
# config = toml.load(r'D:\softs\github_david\pose2sim\Pose2Sim\S00_Demo_Session\Config.toml')
# project_dir = r'D:\softs\github_david\pose2sim\Pose2Sim\S00_Demo_Session\S00_P00_SingleParticipant\S00_P00_T01_BalancingTrial'


all_json_data_f = []
for js_file in json_files_f:
    all_json_data_f.append(read_json(js_file))
#TODO: remove people with average likelihood < 0.3, no full torso, less than 12 joints...
#print('filter2d commented by David in dataset/base.py L498')

persons_per_view = [0] + [len(j) for j in all_json_data_f]
cum_persons_per_view = np.cumsum(persons_per_view)

affinity = compute_affinity(all_json_data_f, calib_params, cum_persons_per_view, reconstruction_error_threshold=reconstruction_error_threshold)
circ_constraint = circular_constraint(cum_persons_per_view)
affinity = affinity * circ_constraint
#TODO: affinity without hand, face, feet in ray.py L31
affinity = matchSVT(affinity, cum_persons_per_view, circ_constraint, max_iter = 20, w_rank = 50, tol = 1e-4, w_sparse=0.1)
affinity[affinity<min_affinity] = 0
proposals = person_index_per_cam(affinity, min_cameras_for_triangulation)
#TODO: SAVE IN JSON
#TODO: INTEGRATE




for cam in range(n_cams):
    with open(json_tracked_files_f[cam], 'w') as json_tracked_f:
        with open(json_files_f[cam], 'r') as json_f:
            js = json.load(json_f)
            js_new = js.copy()
            js_new['people'] = []
            for new_comb in proposals:
                if not np.isnan(new_comb[cam]):
                    js_new['people'] += [js['people'][int(new_comb[cam])]]
                else:
                    js_new['people'] += [{}]
        json_tracked_f.write(json.dumps(js_new))







# not used






# annots: nview * n_persons * {bbox, keypoints, isKeyFrame, id}. keypoints: n_joints*(x,y,conf)


# python scripts/preprocess/extract_video.py D:\softs\github_david\EasyMocap\data\rugby_zak --openpose D:\softs\openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended\openpose
# python apps/demo/mvmp.py D:\softs\github_david\EasyMocap\data\rugby_zak --out D:\softs\github_david\EasyMocap\data\rugby_zak/output --annot annots --cfg config/exp/mvmp1f.yml --undis --vis_det --vis_repro

# extract images
# run openpose
# create annots (image name and size, coords+bbox+ID)


# cameras: calib
# cams: camera basenames
# affinity: from config/exp/mvmp1f.yml

'''
affinity_model(annots, images) -> see ComposedAffinity -> 
-> model -> mvmp1f.yml -> ray.Affinity
-> see getDimGroupd, composeAff, SimpleConstrain, matchSVT

out[key] = model(annots, dimGroups) -> __call__(self, annots, dimGroups)

'''