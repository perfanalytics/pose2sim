from typing import Tuple

import numpy as np


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # get maximum value locations
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # get maximum value across x and y axis
    # mask = max_val_x > max_val_y
    # max_val_x[mask] = max_val_y[mask]
    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1

    # reshape
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def convert_coco_to_openpose(keypoints, scores):
    keypoints_info = np.concatenate((keypoints, scores[..., None]), axis=-1)

    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)

    # neck score when visualizing pred
    neck[:,
         2:3] = np.where(keypoints_info[:, 5, 2:3] > keypoints_info[:, 6, 2:3],
                         keypoints_info[:, 6, 2:3], keypoints_info[:, 5, 2:3])
    new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)

    mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    keypoints, scores = keypoints_info[..., :2], keypoints_info[..., 2]
    return keypoints, scores
