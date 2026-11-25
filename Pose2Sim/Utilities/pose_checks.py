#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ###########################################################
    ## POSE CHECKS                                           ##
    ###########################################################

    Checks for the pose correction tool: point jumps between frames, and
    disagreement with the other cameras (triangulated, then projected back).
'''

import re
import warnings
from pathlib import Path

import numpy as np
import rtoml


## AUTHORSHIP INFORMATION
__author__ = "AY LARDJANE"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["AY LARDJANE"]
__license__ = "BSD 3-Clause License"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


def _read_toml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return rtoml.loads(fh.read())


NOT_CAMERAS = ("metadata", "capture_volume", "charuco", "checkerboard")


# ===========================================================================
# Calibration
# ===========================================================================
def find_calibration(pose_dir):
    """The calibration file Pose2Sim would use for this keypoint folder."""
    trial = Path(pose_dir).parent
    for folder in (trial / "calibration", trial.parent / "calibration"):
        if folder.is_dir():
            found = sorted(folder.glob("Calib*.toml")) or sorted(folder.glob("*.toml"))
            if found:
                return found[0]
    return None


def rodrigues(vector):
    """Rotation matrix from a rotation vector (same result as cv2.Rodrigues)."""
    r = np.asarray(vector, float).reshape(3)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


def _number(text):
    found = re.findall(r"\d+", str(text))
    return int(found[-1]) if found else None


class Camera:
    """One camera of the calibration: K, distortion, R, T, and P = K [R|T]."""

    def __init__(self, key, section):
        self.key = key
        self.name = str(section.get("name") or key)
        self.K = np.asarray(section["matrix"], float)
        dist = np.asarray(section.get("distortions") or [0, 0, 0, 0], float).ravel()
        self.dist = np.zeros(5)
        self.dist[:min(5, dist.size)] = dist[:5]          # k1, k2, p1, p2, k3
        self.R = rodrigues(section["rotation"])
        self.T = np.asarray(section["translation"], float).reshape(3)
        self.P = self.K @ np.hstack([self.R, self.T.reshape(3, 1)])
        self.size = tuple(section.get("size") or ())
        self.fisheye = bool(section.get("fisheye", False))

    # pixels <-> normalised image coordinates -------------------------------
    def _distort(self, xn, yn):
        if self.fisheye:                     # cv2.fisheye model, k1..k4
            k1, k2, k3, k4 = self.dist[:4]
            r = np.sqrt(xn * xn + yn * yn)
            th = np.arctan(r)
            th2 = th * th
            thd = th * (1 + k1 * th2 + k2 * th2 ** 2 + k3 * th2 ** 3 + k4 * th2 ** 4)
            with np.errstate(all="ignore"):
                scale = np.where(r > 1e-12, thd / r, 1.0)
            return xn * scale, yn * scale
        k1, k2, p1, p2, k3 = self.dist
        r2 = xn * xn + yn * yn
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return xd, yd

    def undistort(self, xy):
        """Distorted pixels -> ideal pixels, like cv2.undistortPoints with P =
        K.
        """
        xy = np.asarray(xy, float)
        if not self.dist.any():
            return xy.copy()
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        xd, yd = (xy[..., 0] - cx) / fx, (xy[..., 1] - cy) / fy
        if self.fisheye:                     # invert theta_d(theta) by Newton
            k1, k2, k3, k4 = self.dist[:4]
            thd = np.sqrt(xd * xd + yd * yd)
            th = thd.copy()
            with np.errstate(all="ignore"):
                for _ in range(20):
                    th2 = th * th
                    f = th * (1 + k1 * th2 + k2 * th2 ** 2 + k3 * th2 ** 3 + k4 * th2 ** 4) - thd
                    df = 1 + 3 * k1 * th2 + 5 * k2 * th2 ** 2 + 7 * k3 * th2 ** 3 + 9 * k4 * th2 ** 4
                    th = th - f / df
                scale = np.where(thd > 1e-12, np.tan(th) / thd, 1.0)
            return np.stack([xd * scale * fx + cx, yd * scale * fy + cy], axis=-1)
        xn, yn = xd.copy(), yd.copy()
        k1, k2, p1, p2, k3 = self.dist
        with np.errstate(all="ignore"):
            for _ in range(20):
                r2 = xn * xn + yn * yn
                radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
                dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
                xn = (xd - dx) / radial
                yn = (yd - dy) / radial
        return np.stack([xn * fx + cx, yn * fy + cy], axis=-1)

    def project(self, Q):
        """Homogeneous 3D points (..., 4) -> pixels (..., 2), with distortion."""
        Q = np.asarray(Q, float)
        with np.errstate(all="ignore"):
            X = Q[..., :3] / Q[..., 3:4]
            C = X @ self.R.T + self.T
            xn, yn = C[..., 0] / C[..., 2], C[..., 1] / C[..., 2]
            xd, yd = self._distort(xn, yn)
            out = np.stack([self.K[0, 0] * xd + self.K[0, 2],
                            self.K[1, 1] * yd + self.K[1, 2]], axis=-1)
            out[C[..., 2] <= 0] = np.nan                  # behind the camera
        return out


def load_cameras(calib_file):
    """Every camera section of a Pose2Sim calibration, in file order."""
    calib = _read_toml(calib_file)
    keys = [k for k in calib if k not in NOT_CAMERAS and isinstance(calib[k], dict)
            and "matrix" in calib[k]]
    return [Camera(k, calib[k]) for k in keys]


def pair_cameras(calib_file, cam_names):
    """Pair the calibration cameras with the camera folders, in order. Returns
    (cameras, warning).
    """
    cams = load_cameras(calib_file)
    if len(cams) < len(cam_names):
        return None, (f"{Path(calib_file).name} describes {len(cams)} camera(s), "
                      f"the folder has {len(cam_names)}.")
    cams = cams[:len(cam_names)]
    note = ""
    folder_numbers = [_number(n) for n in cam_names]
    calib_numbers = [_number(c.name) for c in cams]
    if None not in folder_numbers and None not in calib_numbers \
            and folder_numbers != calib_numbers:
        pairs = ", ".join(f"{a} with {c.name}" for a, c in zip(cam_names, cams))
        note = (f"Pose2Sim pairs the cameras in order: {pairs}. The numbers do "
                f"not match: check the calibration belongs to these videos.")
    return cams, note


def load_projections(calib_file, cam_names):
    """One 3x4 projection matrix per camera folder (Pose2Sim's pairing)."""
    cams, _ = pair_cameras(calib_file, cam_names)
    return None if cams is None else [c.P for c in cams]


def _as_cameras(cameras):
    """Accept Camera objects, or bare 3x4 matrices (no distortion)."""
    out = []
    for c in cameras:
        if isinstance(c, Camera):
            out.append(c)
        else:
            P = np.asarray(c, float)
            cam = Camera.__new__(Camera)
            cam.key = cam.name = "P"
            cam.P = P
            cam.dist = np.zeros(5)
            cam.K = np.eye(3)
            cam.size, cam.fisheye = (), False
            cam.project = (lambda Q, P=P: _project(P, Q))
            cam.undistort = (lambda xy: np.asarray(xy, float).copy())
            out.append(cam)
    return out


# ===========================================================================
# Measures
# ===========================================================================
def present(block):
    """block: (frames, keypoints, 3). True where a point was detected."""
    return (block[..., 2] > 0) & ~((block[..., 0] == 0) & (block[..., 1] == 0))


HEIGHT_WINDOW = 121      # frames around each frame used for the height (about 4 s at 30 fps)
HEIGHT_PERCENTILE = 50   # of the vertical extent over that window: the median
# "vertical": extent from highest to lowest point (a body); "largest": the
# larger extent (a hand or a face).
EXTENT = "vertical"
# Keypoint names of the model, set by the window: the height then comes from
# the segment lengths (see segment_height).
HEIGHT_NAMES = None


# Segments added up for the height, as compute_height in Pose2Sim/common.py.
HEIGHT_SEGMENTS = [("RHeel", "RAnkle", "LHeel", "LAnkle"),
                   ("RAnkle", "RKnee", "LAnkle", "LKnee"),
                   ("RKnee", "RHip", "LKnee", "LHip"),
                   ("RHip", "RShoulder", "LHip", "LShoulder")]


def segment_height(block, ok, names):
    """Height in pixels per frame from the segment lengths (heel to ankle,
    shank, thigh, hip to shoulder, then the head), as compute_height in
    Pose2Sim/common.py, so that it does not shrink when the person crouches.
    None when the model lacks these points.
    """
    names = list(names or [])
    needed = {n for seg in HEIGHT_SEGMENTS for n in seg}
    if not needed <= set(names):
        return None
    xy = np.where(ok[..., None], block[..., :2], np.nan)
    pt = lambda n: xy[:, names.index(n)]
    length = lambda a, b: np.linalg.norm(pt(a) - pt(b), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        parts = [np.nanmean([length(r0, r1), length(l0, l1)], axis=0)
                 for r0, r1, l0, l1 in HEIGHT_SEGMENTS]
        legs = parts[0] + parts[1] + parts[2]
        mid_shoulder = np.nanmean([pt("RShoulder"), pt("LShoulder")], axis=0)
        head = np.full(len(block), np.nan)
        if "Head" in names:
            head = np.linalg.norm(pt("Head") - mid_shoulder, axis=1) * 1.008
        if "Nose" in names:
            nose = np.linalg.norm(pt("Nose") - mid_shoulder, axis=1) * 1.5
            head = np.where(np.isfinite(head), head, nose)
        # without the head: leg length / 0.485 (Winter, 2009), as Pose2Sim
        return np.where(np.isfinite(head), legs + parts[3] + head, legs / 0.485)


def body_height(block, threshold=0.0, window=HEIGHT_WINDOW, names=None):
    """Height of the person in pixels, per frame, median over `window` frames.
    With the keypoint names of a body model: from the segment lengths (see
    segment_height), the vertical extent filling the frames where a segment
    is missing. Otherwise: the vertical extent.
    """
    names = HEIGHT_NAMES if names is None else names
    ok = present(block) & (block[..., 2] >= threshold)
    y = np.where(ok, block[..., 1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
        if EXTENT == "largest":
            x = np.where(ok, block[..., 0], np.nan)
            h = np.fmax(h, np.nanmax(x, axis=1) - np.nanmin(x, axis=1))
        h[ok.sum(axis=1) < 4] = np.nan
        seg = segment_height(block, ok, names) if EXTENT == "vertical" else None
        if seg is not None and np.isfinite(seg).any():
            both = np.isfinite(seg) & np.isfinite(h) & (h > 0)
            # the extent, brought to the scale of the segments, where they are missing
            ratio = np.nanmedian(seg[both] / h[both]) if both.any() else 1.0
            h = np.where(np.isfinite(seg), seg, h * ratio)
        median = np.nanmedian(h) if np.isfinite(h).any() else np.nan
        h = np.where(np.isfinite(h) & (h > 0.3 * median), h, np.nan)
        n = len(h)
        if window and window > 1 and n > 1 and np.isfinite(h).any():
            half = int(window) // 2
            padded = np.concatenate([np.full(half, np.nan), h, np.full(half, np.nan)])
            windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
            h = np.nanpercentile(windows, HEIGHT_PERCENTILE, axis=1)
    h = np.where(np.isfinite(h), h, median)
    return h


def jumps(block, threshold=0.0):
    """Displacement since the previous frame, in % of the body height."""
    ok = present(block)
    step = np.full(block.shape[:2], np.nan)
    both = ok[1:] & ok[:-1]
    dist = np.linalg.norm(np.diff(block[..., :2], axis=0), axis=2)
    step[1:][both] = dist[both]
    h = body_height(block, threshold)
    with np.errstate(all="ignore"):
        return 100.0 * step / h[:, None]


def _dlt(rows, good):
    """Confidence weighted DLT on the rows where `good` is True."""
    out = np.full((*good.shape, 4), np.nan)
    if good.any():
        _, _, vt = np.linalg.svd(rows[good])
        Q = vt[:, -1, :]
        with np.errstate(all="ignore"):
            out[good] = Q / Q[:, 3:4]
    return out


def _project(P, Q):
    with np.errstate(all="ignore"):
        w = Q @ P[2]
        return np.stack([(Q @ P[0]) / w, (Q @ P[1]) / w], axis=-1)


OUTLIER = 5.0     # % of body height: an other camera further than this is left out


def triangulate_others(data, cameras, target, threshold, frames=None, usable=None):
    """Where the other cameras put each keypoint, seen from camera `target`.

    The cameras vote: the largest group that agrees within OUTLIER % wins.
    Returns (proposed xy, number of cameras).
    """
    import itertools
    cameras = _as_cameras(cameras)
    n_cams = len(data)
    n_frames = min(len(d) for d in data)
    idx = np.arange(n_frames) if frames is None else np.asarray(frames)
    n_kp = data[0].shape[1]

    rows, weights, blocks, heights = [], [], [], []
    for c in range(n_cams):
        block = data[c][idx]
        cam = cameras[c]
        P = cam.P
        ok = present(block) & (block[..., 2] >= threshold)
        if usable is not None:
            ok &= np.asarray(usable[c], bool)[idx][:, None]
        w = np.where(ok, block[..., 2], 0.0)
        ideal = np.nan_to_num(cam.undistort(block[..., :2]))
        x, y = ideal[..., 0:1], ideal[..., 1:2]
        rows.append(np.stack([(P[0] - x * P[2]) * w[..., None],
                              (P[1] - y * P[2]) * w[..., None]], axis=2))
        weights.append(w)
        blocks.append(block)
        heights.append(body_height(data[c], threshold)[idx])
    rows = np.stack(rows, axis=2)                     # (F, K, n, 2, 4)
    seen = np.stack(weights, axis=2) > 0              # (F, K, n)

    def solve(member):
        keep = seen & member
        r = (rows * keep[..., None, None]).reshape(len(idx), n_kp, -1, 4)
        return _dlt(r, keep.sum(axis=2) >= 2)

    def residual(c, Q):
        xy = cameras[c].project(Q)
        with np.errstate(all="ignore"):
            return 100.0 * np.linalg.norm(xy - blocks[c][..., :2], axis=-1) / heights[c][:, None]

    if n_cams <= 7:
        groups = [g for size in range(n_cams, 1, -1)
                  for g in itertools.combinations(range(n_cams), size)]
    else:                       # many cameras: all, then all but one
        groups = [tuple(range(n_cams))] + list(itertools.combinations(range(n_cams), n_cams - 1))

    best_size = np.zeros((len(idx), n_kp), int)
    best_target = np.zeros((len(idx), n_kp), bool)
    best_err = np.full((len(idx), n_kp), np.inf)
    best_group = np.full((len(idx), n_kp), -1)
    for gi, group in enumerate(groups):
        member = np.zeros(n_cams, bool)
        member[list(group)] = True
        valid = seen[..., member].all(axis=-1)
        if not valid.any():
            continue
        Q = solve(member)
        worst = np.zeros((len(idx), n_kp))
        for c in group:
            worst = np.maximum(worst, np.nan_to_num(residual(c, Q), nan=np.inf))
        ok = valid & (worst <= OUTLIER)
        has_target = target in group
        size = len(group)
        better = ok & ((size > best_size)
                       | ((size == best_size) & has_target & ~best_target)
                       | ((size == best_size) & (has_target == best_target) & (worst < best_err)))
        best_size[better] = size
        best_target[better] = has_target
        best_err[better] = worst[better]
        best_group[better] = gi

    # The proposal needs two other cameras: one alone always agrees.
    proposed = np.full((len(idx), n_kp, 2), np.nan)
    used = np.zeros((len(idx), n_kp), int)
    for gi in np.unique(best_group[best_group >= 0]):
        group = groups[gi]
        sel = best_group == gi
        others = [c for c in group if c != target]
        if len(others) < 2:
            continue
        member = np.zeros(n_cams, bool)
        member[others] = True
        Q = solve(member)
        xy = cameras[target].project(Q)
        proposed[sel] = xy[sel]
        used[sel] = len(others)
    used[best_group < 0] = 0
    return proposed, used


def disagreement(data, cameras, target, threshold, frames=None, usable=None):
    """Distance between the detected point and where the other cameras put it
    (px, %, proposed xy).
    """
    n_frames = min(len(d) for d in data)
    idx = np.arange(n_frames) if frames is None else np.asarray(frames)
    proposed, _ = triangulate_others(data, cameras, target, threshold, idx, usable)
    block = data[target][idx]
    px = np.linalg.norm(block[..., :2] - proposed, axis=2)
    ok = present(block)
    if usable is not None:
        ok &= np.asarray(usable[target], bool)[idx][:, None]
    px[~ok] = np.nan
    h = body_height(data[target], threshold)[idx]
    with np.errstate(all="ignore"):
        pct = 100.0 * px / h[:, None]
    return px, pct, proposed


def _plain_disagreement(data, cameras, target, threshold, idx, usable=None):
    """Camera `target` against all the other cameras together (px, %)."""
    cameras = _as_cameras(cameras)
    rows, good = [], None
    for c in range(len(data)):
        if c == target:
            continue
        block = data[c][idx]
        ok = present(block) & (block[..., 2] >= threshold)
        if usable is not None:
            ok &= np.asarray(usable[c], bool)[idx][:, None]
        w = np.where(ok, block[..., 2], 0.0)
        ideal = np.nan_to_num(cameras[c].undistort(block[..., :2]))
        P = cameras[c].P
        rows.append(np.stack([(P[0] - ideal[..., 0:1] * P[2]) * w[..., None],
                              (P[1] - ideal[..., 1:2] * P[2]) * w[..., None]], axis=2))
        good = ok.astype(int) if good is None else good + ok
    rows = np.stack(rows, axis=2).reshape(len(idx), data[0].shape[1], -1, 4)
    Q = _dlt(rows, good >= 2)
    block = data[target][idx]
    px = np.linalg.norm(cameras[target].project(Q) - block[..., :2], axis=-1)
    ok = present(block)
    if usable is not None:
        ok &= np.asarray(usable[target], bool)[idx][:, None]
    px[~ok] = np.nan
    h = body_height(data[target], threshold)[idx]
    with np.errstate(all="ignore"):
        return px, 100.0 * px / h[:, None]


# ===========================================================================
# How well do the cameras agree, overall?
# ===========================================================================
GOOD_AGREEMENT = 4.0      # % of height: typical median for a sound setup
BAD_AGREEMENT = 10.0      # % of height: calibration, order or sync is wrong


def calibration_report(data, cameras, usable=None, threshold=0.6, max_frames=240):
    """Median disagreement of each camera with the others, on confident
    points.
    """
    cameras = _as_cameras(cameras)
    n_frames = min(len(d) for d in data)
    usable_any = np.ones(n_frames, bool)
    seen = np.zeros(n_frames, int)
    for c, block in enumerate(data):
        ok = present(block[:n_frames]) & (block[:n_frames, :, 2] >= threshold)
        if usable is not None:
            ok &= np.asarray(usable[c], bool)[:n_frames][:, None]
        seen += ok.any(axis=1)
    candidates = np.flatnonzero(usable_any & (seen >= 3))
    if len(candidates) > max_frames:
        candidates = candidates[np.linspace(0, len(candidates) - 1, max_frames).astype(int)]
    report = []
    for c in range(len(data)):
        if not len(candidates):
            report.append({"camera": c, "pct": np.nan, "px": np.nan, "points": 0})
            continue
        # raw agreement, no vote: a vote would hide a broken setup
        px, pct = _plain_disagreement(data, cameras, c, threshold, candidates, usable)
        confident = data[c][candidates][..., 2] >= threshold
        values = pct[confident & np.isfinite(pct)]
        pixels = px[confident & np.isfinite(px)]
        report.append({"camera": c,
                       "pct": float(np.median(values)) if values.size else np.nan,
                       "px": float(np.median(pixels)) if pixels.size else np.nan,
                       "points": int(values.size)})
    return report


# ===========================================================================
# Matching the person across cameras (folders before person association)
# ===========================================================================
class _Seen:
    """One person in one camera, prepared once for every combination."""

    def __init__(self, block, cam, threshold):
        self.block, self.cam = block, cam
        self.ok = present(block[None])[0] & (block[:, 2] >= threshold)
        w = np.where(self.ok, block[:, 2], 0.0)
        ideal = np.nan_to_num(cam.undistort(block[:, :2]))
        P = cam.P
        self.rows = np.stack([(P[0] - ideal[:, 0:1] * P[2]) * w[:, None],
                              (P[1] - ideal[:, 1:2] * P[2]) * w[:, None]], axis=1)
        self.h = body_height(block[None], threshold)[0]
        self.usable = self.ok.sum() >= 4 and np.isfinite(self.h) and self.h > 0


def _consistency(seen):
    """How well one person agrees in 3D across cameras: the worst camera error
    (%), and the cameras used.
    """
    used = [s for s in seen if s is not None]
    if len(used) < 2:
        return np.inf, 0
    rows = np.concatenate([s.rows for s in used], axis=1)
    good = np.sum([s.ok for s in used], axis=0) >= 2
    if good.sum() < 3:
        return np.inf, 0
    Q = _dlt(rows, good)
    worst = 0.0
    for s in used:
        sel = s.ok & good
        if sel.sum() < 3:
            continue
        xy = s.cam.project(Q[sel])
        worst = max(worst, float(np.median(np.linalg.norm(xy - s.block[sel, :2], axis=1))
                                 * 100.0 / s.h))
    return worst, len(used)


def associate_person(people, cameras, threshold=0.3, previous=None,
                     missing_penalty=8.0, switch_penalty=1.0, limit=15.0):
    """Which detected person, in each camera, is the same person (-1 if not
    found).
    """
    import itertools
    cameras = _as_cameras(cameras)
    n = len(people)
    prepared, options = [], []
    for c in range(n):
        here = {}
        for i, p in enumerate(people[c]):
            if p is not None and p.size:
                s = _Seen(p, cameras[c], threshold)
                if s.usable:
                    here[i] = s
        prepared.append(here)
        options.append(list(here) + [-1])
    best, best_cost = [-1] * n, np.inf
    for combo in itertools.product(*options):
        if sum(1 for i in combo if i >= 0) < 2:
            continue
        missing = sum(1 for i in combo if i < 0)
        lower = missing_penalty * missing
        if lower >= best_cost:
            continue
        err, used = _consistency([prepared[c][i] if i >= 0 else None
                                  for c, i in enumerate(combo)])
        if not np.isfinite(err) or err > limit:
            continue
        cost = err + lower
        if previous is not None:
            cost += switch_penalty * sum(1 for c in range(n)
                                         if previous[c] >= 0 and combo[c] != previous[c])
        if cost < best_cost:
            best, best_cost = list(combo), cost
    return best
