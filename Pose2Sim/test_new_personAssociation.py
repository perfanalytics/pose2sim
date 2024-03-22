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
f=0
n_cams_off = 0

json_files_f = [json_files[c][f] for c in range(n_cams)]
json_tracked_files_f = [json_tracked_files[c][f] for c in range(n_cams)]


Q_kpt_old = Q_kpt
error_threshold_tracking = config.get('personAssociation').get('reproj_error_threshold_association')
likelihood_threshold = config.get('personAssociation').get('likelihood_threshold_association')
min_cameras_for_triangulation = config.get('triangulation').get('min_cameras_for_triangulation')
n_cams = len(json_files_f)



margin_percent = 0.1
likelihood_affinity_rejection_threshold = 0.1
reconstruction_error_threshold = 0.1 # 0.1 = 10 cm


def retrieve_calib_params(calib_file):
    '''
    Compute projection matrices from toml calibration file.
    
    INPUT:
    - calib_file: calibration .toml file.
    
    OUTPUT:
    - S: (h,w) vectors as list of 2x1 arrays
    - K: intrinsic matrices as list of 3x3 arrays
    - dist: distortion vectors as list of 4x1 arrays
    - inv_K: inverse intrinsic matrices as list of 3x3 arrays
    - optim_K: intrinsic matrices for undistorting points as list of 3x3 arrays
    - R: rotation rodrigue vectors as list of 3x1 arrays
    - T: translation vectors as list of 3x1 arrays
    '''
    
    calib = toml.load(calib_file)

    S, K, dist, optim_K, inv_K, R, R_mat, T = [], [], [], [], [], [], [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            S.append(np.array(calib[cam]['size']))
            K.append(np.array(calib[cam]['matrix']))
            dist.append(np.array(calib[cam]['distortions']))
            optim_K.append(cv2.getOptimalNewCameraMatrix(K[c], dist[c], [int(s) for s in S[c]], 1, [int(s) for s in S[c]])[0])
            inv_K.append(np.linalg.inv(K[c]))
            R.append(np.array(calib[cam]['rotation']))
            R_mat.append(cv2.Rodrigues(R[c])[0])
            T.append(np.array(calib[cam]['translation']))
    calib_params = {'S': S, 'K': K, 'dist': dist, 'inv_K': inv_K, 'optim_K': optim_K, 'R': R, 'R_mat': R_mat, 'T': T}
            
    return calib_params


def bounding_boxes(js_file, margin_percent=0.1, around='extremities'):
    '''
    Compute the bounding boxes of the people in the json file.
    Either around the extremities (with a margin)
    or around the center of the person (with a margin).
    '''

    bounding_boxes = []
    with open(js_file, 'r') as json_f:
        js = json.load(json_f)
        for people in range(len(js['people'])):
            if len(js['people'][people]['pose_keypoints_2d']) < 3: continue
            else:
                x = js['people'][people]['pose_keypoints_2d'][0::3]
                y = js['people'][people]['pose_keypoints_2d'][1::3]
                x_min, x_max = min(x), max(x)
                y_min, y_max = min(y), max(y)

                if around == 'extremities':
                    dx = (x_max - x_min) * margin_percent
                    dy = (y_max - y_min) * margin_percent
                    bounding_boxes.append([x_min-dx, y_min-dy, x_max+dx, y_max+dy])
                
                elif around == 'center':
                    x_mean, y_mean = np.mean(x), np.mean(y)
                    x_size = (x_max - x_min) * (1 + margin_percent)
                    y_size = (y_max - y_min) * (1 + margin_percent)
                    bounding_boxes.append([x_mean - x_size/2, y_mean - y_size/2, x_mean + x_size/2, y_mean + y_size/2])

    return bounding_boxes   


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


def compute_affinity(all_json_data_f, calib_params, reconstruction_error_threshold=0.1):
    '''
    Compute the affinity between all the people in the different views.

    The affinity is defined as the distance between epipolar lines in each view 
    (reciprocal product of Plucker coordinates).
    Another approach would be to project one epipolar line onto the other camera
    plane and compute the line to point distance, but it is more computationally 
    intensive (simple dot product vs projection and distance calculation). 
    
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
    persons_per_view = [0] + [len(j) for j in all_json_data_f]
    cum_persons_per_view = np.cumsum(persons_per_view)
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


# how many persons per view: [2, 3, 4, 3, 4, 0, 4, 5]
# dimGroups: [0, 2, 5, 9, 12, 16, 16, 20, 25]
# maptoview :  [0 0 1 1 1 2 2 2 2 3 3 3 4 4 4 4 6 6 6 6 7 7 7 7 7]

    return affinity


all_json_data_f = []
for js_file in json_files_f:
    all_json_data_f.append(read_json(js_file))
#TODO: remove people with average likelihood < 0.3, no full torso, less than 12 joints...
#print('filter2d commented by David in dataset/base.py L498')
affinity = compute_affinity(all_json_data_f)
#TODO: affinity without hand, face, feet
# A person cannot be matched with 


# python scripts/preprocess/extract_video.py D:\softs\github_david\EasyMocap\data\rugby_zak --openpose D:\softs\openpose-1.6.0-binaries-win64-gpu-flir-3d_recommended\openpose
# python apps/demo/mvmp.py D:\softs\github_david\EasyMocap\data\rugby_zak --out D:\softs\github_david\EasyMocap\data\rugby_zak/output --annot annots --cfg config/exp/mvmp1f.yml --undis --vis_det --vis_repro

# extract images
# run openpose
# create annots (image name and size, coords+bbox+ID)


# cameras: calib
# cams: camera basenames
# affinity: from config/exp/mvmp1f.yml

affinity_model(annots, images) -> see ComposedAffinity -> 
-> model -> mvmp1f.yml -> ray.Affinity
-> see getDimGroupd, composeAff, SimpleConstrain, matchSVT

out[key] = model(annots, dimGroups) -> __call__(self, annots, dimGroups)



 













