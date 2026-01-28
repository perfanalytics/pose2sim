#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
###########################################################################
## BUNDLE ADJUSTMENT                                                     ##
###########################################################################

GPU-accelerated Bundle Adjustment for camera extrinsic parameter refinement.
Uses ba-cuda library for Levenberg-Marquardt optimization with analytical Jacobians.

This module provides bundle adjustment functionality for:
1. Calibration refinement using scene points (e.g., acrylic box corners)
2. Triangulation refinement using human keypoints

INPUTS:
- 2D observations (keypoints or scene points)
- 3D points (triangulated or known)
- Camera parameters (intrinsics fixed, extrinsics to be optimized)

OUTPUTS:
- Refined camera extrinsic parameters (R, T)
'''

## INIT
import numpy as np
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ba_cuda import LevenbergMarquardtOptimizer
    BA_CUDA_AVAILABLE = True
except ImportError:
    BA_CUDA_AVAILABLE = False


## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim"
__copyright__ = "Copyright 2026, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__maintainer__ = "HunMin Kim"
__email__ = ""
__status__ = "Development"


## FUNCTIONS
def check_ba_availability():
    '''
    Check if bundle adjustment dependencies are available.
    
    RETURNS:
    - available: bool, True if ba-cuda and torch are available
    - message: str, status message
    '''
    if not TORCH_AVAILABLE:
        return False, "PyTorch is not installed. Install with: pip install torch"
    if not BA_CUDA_AVAILABLE:
        return False, "ba-cuda is not installed. Install with: pip install ba-cuda"
    return True, "Bundle adjustment is available"


def get_device():
    '''
    Get the best available device for computation.
    
    RETURNS:
    - device: torch.device
    '''
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        logging.warning("CUDA not available, using CPU for bundle adjustment (slower)")
        return torch.device('cpu')


def rodrigues_to_rotation_matrix(rvec):
    '''
    Convert Rodrigues vector to rotation matrix.
    
    INPUTS:
    - rvec: numpy array (3,) - Rodrigues rotation vector
    
    RETURNS:
    - R: numpy array (3, 3) - Rotation matrix
    '''
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotation_matrix_to_rodrigues(R):
    '''
    Convert rotation matrix to Rodrigues vector.
    
    INPUTS:
    - R: numpy array (3, 3) - Rotation matrix
    
    RETURNS:
    - rvec: numpy array (3,) - Rodrigues rotation vector
    '''
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def prepare_ba_inputs(
    observations_2d,
    points_3d,
    camera_indices,
    intrinsics_list,
    distortions_list,
    extrinsics_list,
    confidence_weights=None
):
    '''
    Prepare inputs for bundle adjustment optimizer.
    
    INPUTS:
    - observations_2d: list of numpy arrays, 2D observations per camera [(N_i, 2), ...]
    - points_3d: numpy array (N_points, 3), 3D points in world coordinates
    - camera_indices: list of numpy arrays, point indices per camera
    - intrinsics_list: list of numpy arrays (3, 3), camera intrinsic matrices
    - distortions_list: list of numpy arrays (5,), distortion coefficients
    - extrinsics_list: list of tuples (R, T), rotation matrices and translation vectors
    - confidence_weights: optional, list of numpy arrays with confidence scores
    
    RETURNS:
    - dict with tensors ready for optimization
    '''
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    
    device = get_device()
    dtype = torch.float64
    
    n_cameras = len(intrinsics_list)
    
    # Flatten observations and create camera indices
    all_observations = []
    all_points_3d = []
    all_camera_indices = []
    all_weights = []
    
    for cam_idx in range(n_cameras):
        if len(observations_2d[cam_idx]) == 0:
            continue
            
        obs = observations_2d[cam_idx]
        pts_indices = camera_indices[cam_idx]
        
        for i, pt_idx in enumerate(pts_indices):
            if not np.isnan(obs[i]).any() and not np.isnan(points_3d[pt_idx]).any():
                all_observations.append(obs[i])
                all_points_3d.append(points_3d[pt_idx])
                all_camera_indices.append(cam_idx)
                
                if confidence_weights is not None and confidence_weights[cam_idx] is not None:
                    all_weights.append(confidence_weights[cam_idx][i])
                else:
                    all_weights.append(1.0)
    
    if len(all_observations) == 0:
        raise ValueError("No valid observations for bundle adjustment")
    
    # Convert to tensors
    observations_tensor = torch.tensor(np.array(all_observations), dtype=dtype, device=device)
    points_3d_tensor = torch.tensor(np.array(all_points_3d), dtype=dtype, device=device)
    camera_indices_tensor = torch.tensor(all_camera_indices, dtype=torch.long, device=device)
    weights_tensor = torch.tensor(all_weights, dtype=dtype, device=device)
    
    # Prepare intrinsics tensor (N_cams, 3, 3)
    intrinsics_tensor = torch.tensor(
        np.array([K for K in intrinsics_list]), 
        dtype=dtype, 
        device=device
    )
    
    # Prepare distortions tensor (N_cams, 5)
    distortions_tensor = torch.tensor(
        np.array([D.flatten()[:5] if len(D.flatten()) >= 5 else np.pad(D.flatten(), (0, 5-len(D.flatten()))) 
                  for D in distortions_list]),
        dtype=dtype,
        device=device
    )
    
    # Prepare initial poses tensor (N_cams, 3, 4) as [R|t]
    initial_poses = []
    for R, T in extrinsics_list:
        if R.shape == (3,):  # Rodrigues vector
            R_mat = rodrigues_to_rotation_matrix(R)
        else:
            R_mat = R
        T_vec = T.flatten().reshape(3, 1)
        pose = np.hstack([R_mat, T_vec])  # (3, 4)
        initial_poses.append(pose)
    
    initial_poses_tensor = torch.tensor(
        np.array(initial_poses),
        dtype=dtype,
        device=device
    )
    
    return {
        'observations': observations_tensor,
        'points_3d': points_3d_tensor,
        'camera_indices': camera_indices_tensor,
        'intrinsics': intrinsics_tensor,
        'distortions': distortions_tensor,
        'initial_poses': initial_poses_tensor,
        'weights': weights_tensor,
        'device': device,
        'dtype': dtype
    }


def run_bundle_adjustment(
    observations_2d,
    points_3d,
    camera_indices,
    intrinsics_list,
    distortions_list,
    extrinsics_list,
    ref_cam_idx=0,
    max_iterations=50,
    abs_tolerance=1e-6,
    rel_tolerance=1e-6,
    huber_delta=1.0,
    confidence_weights=None
):
    '''
    Run bundle adjustment to refine camera extrinsic parameters.
    
    INPUTS:
    - observations_2d: list of numpy arrays, 2D observations per camera
    - points_3d: numpy array (N_points, 3), 3D points
    - camera_indices: list of numpy arrays, point indices per camera
    - intrinsics_list: list of numpy arrays (3, 3), camera intrinsic matrices
    - distortions_list: list of numpy arrays, distortion coefficients
    - extrinsics_list: list of tuples (R, T), initial extrinsic parameters
    - ref_cam_idx: int, reference camera index (fixed during optimization)
    - max_iterations: int, maximum LM iterations
    - abs_tolerance: float, absolute convergence tolerance
    - rel_tolerance: float, relative convergence tolerance
    - huber_delta: float, Huber loss delta for outlier robustness
    - confidence_weights: optional, confidence scores for observations
    
    RETURNS:
    - optimized_extrinsics: list of tuples (R, T), refined extrinsic parameters
    - rmse: float, final reprojection RMSE
    - success: bool, whether optimization converged
    '''
    # Check availability
    available, msg = check_ba_availability()
    if not available:
        logging.error(f"Bundle adjustment not available: {msg}")
        return extrinsics_list, float('inf'), False
    
    logging.info("Preparing bundle adjustment inputs...")
    
    try:
        # Prepare inputs
        ba_inputs = prepare_ba_inputs(
            observations_2d,
            points_3d,
            camera_indices,
            intrinsics_list,
            distortions_list,
            extrinsics_list,
            confidence_weights
        )
        
        logging.info(f"Bundle adjustment: {len(ba_inputs['observations'])} observations, "
                    f"{len(intrinsics_list)} cameras")
        
        # Create optimizer
        optimizer = LevenbergMarquardtOptimizer(
            max_iterations=max_iterations,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
            initial_damping=1e-3,
            huber_delta=huber_delta,
            device=ba_inputs['device'],
            dtype=ba_inputs['dtype']
        )
        
        logging.info("Running Levenberg-Marquardt optimization...")
        
        # Run optimization
        optimized_poses, rmse = optimizer.optimize(
            observations=ba_inputs['observations'],
            points_3d=ba_inputs['points_3d'],
            camera_indices=ba_inputs['camera_indices'],
            intrinsics=ba_inputs['intrinsics'],
            distortions=ba_inputs['distortions'],
            initial_poses=ba_inputs['initial_poses'],
            ref_cam_idx=ref_cam_idx
        )
        
        logging.info(f"Bundle adjustment completed. RMSE: {rmse:.4f} pixels")
        
        # Convert back to numpy
        optimized_poses_np = optimized_poses.cpu().numpy()
        
        # Extract R and T from optimized poses
        optimized_extrinsics = []
        for i in range(len(extrinsics_list)):
            R_mat = optimized_poses_np[i, :, :3]  # (3, 3)
            T_vec = optimized_poses_np[i, :, 3]   # (3,)
            
            # Convert R to Rodrigues if original was Rodrigues
            original_R = extrinsics_list[i][0]
            if original_R.shape == (3,):
                R_out = rotation_matrix_to_rodrigues(R_mat)
            else:
                R_out = R_mat
            
            optimized_extrinsics.append((R_out, T_vec))
        
        return optimized_extrinsics, float(rmse), True
        
    except Exception as e:
        logging.error(f"Bundle adjustment failed: {str(e)}")
        return extrinsics_list, float('inf'), False


def bundle_adjustment_calibration(
    calib_params,
    scene_points_2d,
    scene_points_3d,
    config_dict
):
    '''
    Bundle adjustment for calibration refinement using scene points.
    
    INPUTS:
    - calib_params: dict with camera parameters (C, S, D, K, R, T)
    - scene_points_2d: list of numpy arrays, 2D scene points per camera
    - scene_points_3d: numpy array (N, 3), known 3D scene points
    - config_dict: configuration dictionary with BA settings
    
    RETURNS:
    - refined_calib_params: dict with refined camera parameters
    - rmse: float, final reprojection RMSE
    - success: bool
    '''
    ba_config = config_dict.get('calibration', {}).get('bundle_adjustment', {})
    
    if not ba_config.get('enabled', False):
        logging.info("Bundle adjustment disabled in calibration config")
        return calib_params, 0.0, True
    
    logging.info("Running bundle adjustment for calibration refinement...")
    
    n_cameras = len(calib_params['R'])
    n_points = len(scene_points_3d)
    
    # Prepare camera indices (all cameras see all points for scene calibration)
    camera_indices = [np.arange(n_points) for _ in range(n_cameras)]
    
    # Prepare extrinsics list
    extrinsics_list = list(zip(calib_params['R'], calib_params['T']))
    
    # Run BA
    optimized_extrinsics, rmse, success = run_bundle_adjustment(
        observations_2d=scene_points_2d,
        points_3d=scene_points_3d,
        camera_indices=camera_indices,
        intrinsics_list=calib_params['K'],
        distortions_list=calib_params['D'],
        extrinsics_list=extrinsics_list,
        ref_cam_idx=ba_config.get('ref_cam_idx', 0),
        max_iterations=ba_config.get('max_iterations', 50),
        abs_tolerance=ba_config.get('tolerance', 1e-6),
        rel_tolerance=ba_config.get('tolerance', 1e-6),
        huber_delta=ba_config.get('huber_delta', 1.0)
    )
    
    if success:
        refined_calib_params = calib_params.copy()
        refined_calib_params['R'] = [ext[0] for ext in optimized_extrinsics]
        refined_calib_params['T'] = [ext[1] for ext in optimized_extrinsics]
        logging.info(f"Calibration BA successful. RMSE: {rmse:.4f} pixels")
    else:
        refined_calib_params = calib_params
        logging.warning("Calibration BA failed, using original parameters")
    
    return refined_calib_params, rmse, success


def bundle_adjustment_triangulation(
    calib_params,
    keypoints_2d,
    keypoints_3d,
    keypoints_confidence,
    camera_visibility,
    config_dict
):
    '''
    Bundle adjustment for triangulation refinement using human keypoints.
    
    INPUTS:
    - calib_params: dict with camera parameters (C, S, D, K, R, T)
    - keypoints_2d: numpy array (N_cams, N_frames, N_keypoints, 2), 2D keypoints
    - keypoints_3d: numpy array (N_frames, N_keypoints, 3), triangulated 3D keypoints
    - keypoints_confidence: numpy array (N_cams, N_frames, N_keypoints), confidence scores
    - camera_visibility: numpy array (N_cams, N_frames, N_keypoints), visibility mask
    - config_dict: configuration dictionary with BA settings
    
    RETURNS:
    - refined_calib_params: dict with refined camera parameters
    - rmse: float, final reprojection RMSE
    - success: bool
    '''
    ba_config = config_dict.get('triangulation', {}).get('bundle_adjustment', {})
    
    if not ba_config.get('enabled', False):
        logging.info("Bundle adjustment disabled in triangulation config")
        return calib_params, 0.0, True
    
    logging.info("Running bundle adjustment for triangulation refinement...")
    
    n_cameras = len(calib_params['R'])
    n_frames, n_keypoints, _ = keypoints_3d.shape
    
    # Flatten keypoints across frames
    points_3d_flat = keypoints_3d.reshape(-1, 3)  # (N_frames * N_keypoints, 3)
    
    # Prepare observations and indices per camera
    observations_2d = []
    camera_indices = []
    confidence_weights = []
    
    use_confidence = ba_config.get('use_confidence_weights', True)
    
    for cam_idx in range(n_cameras):
        cam_obs = []
        cam_indices = []
        cam_conf = []
        
        for frame_idx in range(n_frames):
            for kpt_idx in range(n_keypoints):
                flat_idx = frame_idx * n_keypoints + kpt_idx
                
                # Check visibility
                if camera_visibility[cam_idx, frame_idx, kpt_idx]:
                    obs = keypoints_2d[cam_idx, frame_idx, kpt_idx]
                    if not np.isnan(obs).any():
                        cam_obs.append(obs)
                        cam_indices.append(flat_idx)
                        if use_confidence:
                            cam_conf.append(keypoints_confidence[cam_idx, frame_idx, kpt_idx])
                        else:
                            cam_conf.append(1.0)
        
        observations_2d.append(np.array(cam_obs) if cam_obs else np.array([]).reshape(0, 2))
        camera_indices.append(np.array(cam_indices, dtype=np.int64))
        confidence_weights.append(np.array(cam_conf) if cam_conf else None)
    
    # Prepare extrinsics list
    extrinsics_list = list(zip(calib_params['R'], calib_params['T']))
    
    # Run BA
    optimized_extrinsics, rmse, success = run_bundle_adjustment(
        observations_2d=observations_2d,
        points_3d=points_3d_flat,
        camera_indices=camera_indices,
        intrinsics_list=calib_params['K'],
        distortions_list=calib_params['D'],
        extrinsics_list=extrinsics_list,
        ref_cam_idx=ba_config.get('ref_cam_idx', 0),
        max_iterations=ba_config.get('max_iterations', 50),
        abs_tolerance=ba_config.get('tolerance', 1e-6),
        rel_tolerance=ba_config.get('tolerance', 1e-6),
        huber_delta=ba_config.get('huber_delta', 1.0),
        confidence_weights=confidence_weights if use_confidence else None
    )
    
    if success:
        refined_calib_params = calib_params.copy()
        refined_calib_params['R'] = [ext[0] for ext in optimized_extrinsics]
        refined_calib_params['T'] = [ext[1] for ext in optimized_extrinsics]
        logging.info(f"Triangulation BA successful. RMSE: {rmse:.4f} pixels")
    else:
        refined_calib_params = calib_params
        logging.warning("Triangulation BA failed, using original parameters")
    
    return refined_calib_params, rmse, success


def update_calibration_file(calib_path, refined_params):
    '''
    Update calibration TOML file with refined parameters.
    
    INPUTS:
    - calib_path: str, path to calibration file
    - refined_params: dict with refined camera parameters
    
    RETURNS:
    - success: bool
    '''
    import toml
    
    try:
        # Read existing calibration
        with open(calib_path, 'r') as f:
            calib_data = toml.load(f)
        
        # Update extrinsic parameters
        for i, cam_name in enumerate(refined_params.get('C', [])):
            if cam_name in calib_data:
                R = refined_params['R'][i]
                T = refined_params['T'][i]
                
                # Convert to list for TOML
                if isinstance(R, np.ndarray):
                    R = R.flatten().tolist()
                if isinstance(T, np.ndarray):
                    T = T.flatten().tolist()
                
                calib_data[cam_name]['rotation'] = R
                calib_data[cam_name]['translation'] = T
        
        # Write back
        with open(calib_path, 'w') as f:
            toml.dump(calib_data, f)
        
        logging.info(f"Updated calibration file: {calib_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to update calibration file: {str(e)}")
        return False
