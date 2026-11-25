#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ###########################################################
    ## POSE CORRECTION                                       ##
    ###########################################################

    Manually correct the 2D keypoints produced by pose estimation, before
    they are triangulated.

    Opens a tkinter window with every camera of a trial: select, move and
    check the keypoints, then save them back into the pose folder (a backup
    is made first).

    Usage:
        from Pose2Sim.Utilities import pose_correction; pose_correction.pose_correction_func(r'<pose_folder>')
        OR pose_correction
        OR pose_correction -i <pose_folder>
'''


## INIT
import argparse
import json
import re
import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Video frames will not be shown. "
          "Install with: pip install opencv-python")

try:
    from Pose2Sim.Utilities import pose_checks
except ImportError:                      # run from its own folder
    import pose_checks

try:                                     # (i) tooltips, optional
    from Pose2Sim.Utilities import correction_tips as tips
except ImportError:
    try:
        import correction_tips as tips
    except ImportError:
        tips = None

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available. Video frames will not be shown. "
          "Install with: pip install Pillow")


## AUTHORSHIP INFORMATION
__author__ = "AY LARDJANE"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["AY LARDJANE"]
__license__ = "BSD 3-Clause License"
from importlib.metadata import version
__version__ = version('pose2sim')
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def main():
    parser = argparse.ArgumentParser(
        description='Manually correct 2D keypoints before triangulation.')
    parser.add_argument('-i', '--pose_folder', required=False,
                        help='pose, pose-sync, or pose-associated folder to open. '
                             'If not specified, a folder chooser is displayed')
    args = vars(parser.parse_args())

    pose_correction_func(args)


# ===========================================================================
# Skeletons: names, lines to draw, left and right
# ===========================================================================
# Pose2Sim's pose models (pose_model) and their skeleton in
# Pose2Sim/skeletons.py.
MODEL_ALIASES = {
    "BODY_WITH_FEET": "HALPE_26", "LOWER_BODY": "HALPE_26",
    "HALPE_26_LOWER": "HALPE_26",
    "WHOLE_BODY": "COCO_133", "WHOLE_BODY_WRIST": "COCO_133",
    "COCO_133_WRIST": "COCO_133",
    "BODY": "COCO_17", "HAND": "HAND_21", "FACE": "FACE_106",
    "ANIMAL": "ANIMAL2D_17",
}
# When nothing tells the model, the number of points written per person.
MODEL_BY_COUNT = {26: "HALPE_26", 25: "BODY_25", 17: "COCO_17", 133: "COCO_133",
                  21: "HAND_21", 106: "FACE_106", 136: "HALPE_136", 68: "HALPE_68",
                  39: "SUPERANIMAL_QUADRUPED_39", 27: "SUPERANIMAL_TOPVIEWMOUSE_27"}

HALPE_26_NAMES = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist",
    "LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Head", "Neck", "Hip",
    "LBigToe", "RBigToe", "LSmallToe", "RSmallToe", "LHeel", "RHeel",
)
HALPE_26_LINKS = (
    (19, 12), (12, 14), (14, 16), (16, 21), (21, 23), (16, 25),
    (19, 11), (11, 13), (13, 15), (15, 20), (20, 22), (15, 24),
    (19, 18), (18, 17), (17, 0),
    (18, 6), (6, 8), (8, 10), (18, 5), (5, 7), (7, 9),
    (0, 1), (0, 2), (1, 3), (2, 4),
)
HALPE_26_LEFT = (1, 3, 5, 7, 9, 11, 13, 15, 20, 22, 24)
HALPE_26_RIGHT = (2, 4, 6, 8, 10, 12, 14, 16, 21, 23, 25)

COCO_17_NAMES = HALPE_26_NAMES[:17]
COCO_17_LINKS = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
                 (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
                 (13, 15), (12, 14), (14, 16))

BODY_25_NAMES = (
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow",
    "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
)
BODY_25_LINKS = ((1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                 (1, 8), (8, 9), (9, 10), (10, 11), (11, 22), (22, 23),
                 (11, 24), (8, 12), (12, 13), (13, 14), (14, 19), (19, 20),
                 (14, 21), (0, 15), (0, 16), (15, 17), (16, 18))
BODY_25_LEFT = (5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21)
BODY_25_RIGHT = (2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24)

# Complete tables: every point of the file gets its name and line, even those
# Pose2Sim does not triangulate.
DRAWN = {
    "HALPE_26": (HALPE_26_NAMES, HALPE_26_LINKS),
    "COCO_17": (COCO_17_NAMES, COCO_17_LINKS),
    "BODY_25": (BODY_25_NAMES, BODY_25_LINKS),
}

_SIDE_PREFIX = re.compile(r"^([LR])(?=[A-Z])")               # LShoulder
_SIDE_INFIX = re.compile(r"(?<=[a-z])([LR])(?=\d*$)")        # EyebrowL3
_SIDE_WORD = re.compile(r"(?i)^(left|right)(?=[_ A-Z])")      # left_eye


def side_of(name):
    """'L', 'R' or '' from a keypoint name."""
    for rx in (_SIDE_PREFIX, _SIDE_INFIX):
        m = rx.search(name or "")
        if m:
            return m.group(1)
    m = _SIDE_WORD.search(name or "")
    return m.group(1)[0].upper() if m else ""


def mirror_name(name):
    """The name of the same point on the other side (LHeel -> RHeel)."""
    swap = {"L": "R", "R": "L", "left": "right", "right": "left",
            "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    for rx in (_SIDE_PREFIX, _SIDE_INFIX, _SIDE_WORD):
        m = rx.search(name or "")
        if m:
            return name[:m.start(1)] + swap[m.group(1)] + name[m.end(1):]
    return None


def lr_partners(names):
    """{index: index of the same point on the other side}."""
    where = {n: i for i, n in enumerate(names)}
    out = {}
    for i, n in enumerate(names):
        j = where.get(mirror_name(n))
        if j is not None and j != i:
            out[i] = j
    return out


def normalise_model(name):
    key = str(name or "").strip().upper()
    return MODEL_ALIASES.get(key, key)


class _TreeNode:
    """A skeleton node read from a Config.toml (custom models)."""
    def __init__(self, entry):
        self.name = str(entry.get("name", ""))
        raw = entry.get("id")
        self.id = None if raw in (None, "None", "") else int(raw)
        self.children = [_TreeNode(c) for c in entry.get("children", []) or []]


def pose2sim_tree(model, config=None):
    """The skeleton tree of a model (skeletons.py, or the custom one of
    Config.toml), or None.
    """
    try:
        from Pose2Sim import skeletons
        root = getattr(skeletons, model, None)
        if root is not None and hasattr(root, "children"):
            return root
    except Exception:
        pass
    pose = (config or {}).get("pose") or {}
    for key, value in pose.items():
        if key.upper() == model and isinstance(value, list) and value \
                and isinstance(value[0], dict):
            return _TreeNode(value[0])
    return None


def _walk(node):
    yield node
    for child in getattr(node, "children", ()) or ():
        yield from _walk(child)


def tree_names(root):
    """{id: name} of the points of a skeleton tree."""
    out = {}
    for node in _walk(root):
        if getattr(node, "id", None) is not None:
            out[int(node.id)] = node.name
    return out


def tree_links(root, names):
    """Lines between the points of `names`, following the skeleton tree; nodes
    that are not points are bridged.
    """
    where = {n: i for i, n in enumerate(names)}
    links = set()

    def reps(node):
        if node.name in where:
            return [node.name]
        out = []
        for child in getattr(node, "children", ()) or ():
            out += reps(child)
        return out

    def add(a, b):
        if a in where and b in where and a != b:
            links.add(tuple(sorted((where[a], where[b]))))

    for node in _walk(root):
        children = list(getattr(node, "children", ()) or ())
        if node.name in where:
            for child in children:
                for r in reps(child):
                    add(node.name, r)
            continue
        tops = [r for child in children for r in reps(child)]
        for r in tops:                                   # left with right
            add(r, mirror_name(r) or "")
        for child in children:
            if child.name in where:
                continue
            below = reps(child)
            for r in [c.name for c in children if c.name in where]:
                for b in below:
                    if side_of(r) and side_of(r) == side_of(b):
                        add(r, b)
    return tuple(sorted(links))


def skeleton_for(n_keypoints, model=None, names=None, config=None):
    """Names, lines, left points and right points of the points in a file."""
    model = normalise_model(model) if model else MODEL_BY_COUNT.get(int(n_keypoints), "")
    tree = pose2sim_tree(model, config) if model else None
    table_names, table_links = DRAWN.get(model, ((), ()))
    if names is None:
        known = tree_names(tree) if tree is not None else {}
        names = []
        for i in range(int(n_keypoints)):
            if i < len(table_names):
                names.append(table_names[i])
            else:
                names.append(known.get(i, f"kp{i}"))
        names = tuple(names)
    names = tuple(names)
    if table_links and tuple(names[:len(table_names)]) == tuple(table_names):
        links = table_links
    elif table_links:                       # the same model, other order (TRC)
        where = {n: i for i, n in enumerate(names)}
        links = tuple((where[table_names[a]], where[table_names[b]])
                      for a, b in table_links
                      if table_names[a] in where and table_names[b] in where)
    elif tree is not None:
        links = tree_links(tree, names)
    else:
        links = ()
    left = tuple(i for i, n in enumerate(names) if side_of(n) == "L")
    right = tuple(i for i, n in enumerate(names) if side_of(n) == "R")
    return names, links, left, right


def project_config(folder):
    """(path, dict) of the configuration of a keypoint folder, or (None, {})."""
    folder = Path(folder)
    for base in (folder, folder.parent, folder.parent.parent):
        for name in ("Config.toml", "Config_demo.toml"):
            path = base / name
            if path.is_file():
                try:
                    return path, pose_checks._read_toml(path)
                except Exception:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue
                    m = re.search(r"(?m)^\s*pose_model\s*=\s*['\"]([^'\"]+)['\"]", text)
                    return path, {"pose": {"pose_model": m.group(1)}} if m else {}
    return None, {}


# ===========================================================================
# Reading and writing the keypoint files
# ===========================================================================
FRAME_RE = re.compile(r"_(\d{4,})(?:_keypoints)?\.json$", re.IGNORECASE)
POSE_ORDER = ("pose-associated", "pose-sync", "pose")


def frame_number(path):
    m = FRAME_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def frame_files(directory):
    """Numbered keypoint files of a camera folder, in frame order."""
    directory = Path(directory)
    seen = {}
    for pattern in ("*_keypoints.json", "*.json"):
        for f in directory.glob(pattern):
            if not f.name.startswith(".") and FRAME_RE.search(f.name):
                seen.setdefault(f.name, f)

    def key(path):
        m = FRAME_RE.search(path.name)
        return (0, int(m.group(1)), path.name) if m else (1, 0, path.name)

    return sorted(seen.values(), key=key)


def camera_groups(files, fallback):
    """Files of one folder, grouped by camera name: [(name, files), ...]."""
    groups = {}
    for f in files:
        m = FRAME_RE.search(f.name)
        name = f.name[:m.start()] if m else ""
        groups.setdefault(name or fallback, []).append(f)
    if len(groups) <= 1:
        name = next(iter(groups), fallback)
        name = re.sub(r"[_-]?json$", "", name, flags=re.I) or fallback
        return [(name, files)]
    return sorted(groups.items())


def has_keypoints(directory):
    return bool(frame_files(directory))


def find_pose_folders(root):
    """Keypoint folders under a project, best one per trial."""
    root = Path(root)
    found = []
    for name in POSE_ORDER:
        for d in root.rglob(name):
            if d.is_dir() and "-backup" not in str(d):
                if any(has_keypoints(s) for s in d.iterdir() if s.is_dir()):
                    found.append(d)
    best = {}
    for d in found:
        key = str(d.parent)
        if key not in best or POSE_ORDER.index(d.name) < POSE_ORDER.index(best[key].name):
            best[key] = d
    # Sports2D results: the folders holding its pixel TRC files
    sports2d = sorted({f.parent for f in root.rglob("*.trc")
                       if TRC_PERSON_RE.search(f.name) and "-backup" not in str(f.parent)})
    return list(best.values()) + sports2d


def read_people(path):
    """The people of one keypoint file, as written (list of dicts)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return []
    people = payload.get("people") or []
    return [p if isinstance(p, dict) else {} for p in people]


def person_points(people, person):
    """(n, 3) keypoints of one person, empty when that person is not in the
    file.
    """
    if 0 <= person < len(people):
        flat = people[person].get("pose_keypoints_2d") or []
        if len(flat) >= 3:
            return np.asarray(flat, np.float32).reshape(-1, 3)
    return np.zeros((0, 3), np.float32)


def video_offset(cam_dir, original_dir):
    """Frames to add to a file number to get its video frame (0, the offset,
    or None if unknown).
    """
    cam_dir, original_dir = Path(cam_dir), Path(original_dir)
    if not original_dir.is_dir():
        return None
    try:
        if cam_dir.resolve() == original_dir.resolve():
            return 0
    except OSError:
        pass
    current = {frame_number(f): f for f in frame_files(cam_dir)}
    original = {frame_number(f): f for f in frame_files(original_dir)}
    if not current or not original:
        return None

    def signatures(path):
        out = set()
        for p in read_people(path):
            pts = person_points([p], 0)
            if pts.size and np.abs(pts[:, :2]).sum() > 0:
                out.add(tuple(np.round(pts[:, :2].ravel()[:12], 2)))
        return out

    numbers = sorted(n for n in current if n is not None)
    picks = [numbers[int(i)] for i in np.linspace(0, len(numbers) - 1, min(8, len(numbers)))]
    samples = [(n, signatures(current[n])) for n in picks]
    samples = [(n, s) for n, s in samples if s]
    if not samples:
        return None
    need = max(1, int(np.ceil(0.75 * len(samples))))

    def score(d):
        return sum(1 for n, s in samples
                   if (n + d) in original and s & signatures(original[n + d]))

    first_guess = max(original) - max(current)
    for d in dict.fromkeys((first_guess, 0)):
        if score(d) >= need:
            return int(d)

    # full search: every original file, indexed by its content
    index = {}
    for n, f in original.items():
        for sig in signatures(f):
            index.setdefault(sig, []).append(n)
    votes = {}
    for n, s in samples:
        for sig in s:
            for m in index.get(sig, ()):
                votes[m - n] = votes.get(m - n, 0) + 1
    if votes:
        d, count = max(votes.items(), key=lambda kv: kv[1])
        if count >= need:
            return int(d)
    return None


class Sequence:
    """Every camera of one recording, for one person, in memory, on ONE
    timeline of frame numbers.
    """

    kind = "json"

    def __init__(self, pose_dir, person=0):
        self.root = Path(pose_dir)
        self.person = person
        self.cam_dirs = []
        self.cam_names = []
        self.numbers = []        # the timeline: frame number of each row
        self.files = []          # per camera, one Path or None per row
        self.data = []           # per camera, (rows, keypoints, 3)
        self.original = []       # the same, before any correction
        self.n_people = []       # per camera, people in the file, per row
        self.n_entries = []      # per camera, entries of the file (empty ones too), per row
        self.dirty = []          # rows needing a write, per camera
        self.undo_stack = []
        self.carried = {}        # (cam, row, kp) -> row it was carried from
        self.video_offsets = []
        self.slots = None        # per camera and row: entry of the file, when matched
        self.matched = False
        self._load()

    # ------------------------------------------------------------------
    @property
    def associated(self):
        """True when people are matched across cameras."""
        return self.root.name.startswith("pose-associated")

    def _load(self):
        subdirs = [d for d in sorted(self.root.iterdir())
                   if d.is_dir() and has_keypoints(d)]
        if not subdirs and has_keypoints(self.root):
            subdirs = [self.root]
        if not subdirs:
            raise FileNotFoundError(
                f"No keypoint JSON under {self.root}.\n\n"
                f"Expected camera subfolders (cam01_json, cam01, ...) holding "
                f"files named cam01_000000.json or "
                f"cam01_000000_keypoints.json.")

        per_cam = []
        for d in subdirs:
            if d == self.root:
                # one folder for one camera, or for every camera: split by the
                # name before the number
                for name, files in camera_groups(frame_files(d), d.name):
                    per_cam.append({frame_number(f): f for f in files})
                    self.cam_dirs.append(d)
                    self.cam_names.append(name)
                continue
            per_cam.append({frame_number(f): f for f in frame_files(d)})
            self.cam_dirs.append(d)
            self.cam_names.append(re.sub(r"[_-]?json$", "", d.name, flags=re.I) or d.name)
        self.numbers = sorted(set().union(*[set(m) for m in per_cam]))
        row = {n: i for i, n in enumerate(self.numbers)}

        width, people_max = 0, 1
        raw = []
        for files in per_cam:
            rows = [None] * len(self.numbers)
            count = np.zeros(len(self.numbers), int)
            entries = np.zeros(len(self.numbers), int)
            paths = [None] * len(self.numbers)
            for n, f in files.items():
                people = read_people(f)
                pts = person_points(people, self.person)
                width = max(width, pts.shape[0], *(len(p.get("pose_keypoints_2d") or []) // 3
                                                   for p in people))
                count[row[n]] = sum(1 for p in people if p.get("pose_keypoints_2d"))
                entries[row[n]] = len(people)
                people_max = max(people_max, len(people))
                rows[row[n]], paths[row[n]] = pts, f
            raw.append(rows)
            self.files.append(paths)
            self.n_people.append(count)
            self.n_entries.append(entries)
            self.dirty.append(set())

        self.n_keypoints = width
        for rows in raw:
            block = np.zeros((len(self.numbers), width, 3), np.float32)
            for i, r in enumerate(rows):
                if r is not None and r.size:
                    block[i, : r.shape[0]] = r
            self.data.append(block)
        self.n_frames = len(self.numbers)
        self.max_people = people_max
        self._load_original()

        # video frame of each file, compared with the original `pose` folder
        # (or the backup)
        original_root = self.root.parent / "pose"
        self.video_offsets = []
        for d in self.cam_dirs:
            if self.root.name == "pose" or d == self.root:
                self.video_offsets.append(0)
                continue
            source = self.backup_dir / d.name
            if not source.is_dir():
                source = d
            self.video_offsets.append(video_offset(source, original_root / d.name))

    def _load_original(self):
        """Points before any correction: the backup if there is one, else the
        files now.
        """
        self.original = [b.copy() for b in self.data]
        if not self.has_backup:
            return
        for cam in range(self.n_cameras):
            src = self.cam_backup(cam)
            for t, f in enumerate(self.files[cam]):
                if f is None or not (src / f.name).exists():
                    continue
                pts = person_points(read_people(src / f.name), self.slot(cam, t))
                self.original[cam][t] = 0
                k = min(pts.shape[0], self.n_keypoints)
                self.original[cam][t, :k] = pts[:k]

    # ------------------------------------------------------------------
    @property
    def n_cameras(self):
        return len(self.cam_names)

    @property
    def backup_dir(self):
        return self.root.with_name(self.root.name + "-backup")

    @property
    def has_backup(self):
        d = self.backup_dir
        return d.is_dir() and (has_keypoints(d) or
                               any(has_keypoints(s) for s in d.iterdir() if s.is_dir()))

    def cam_backup(self, cam):
        """Where the backup keeps the files of one camera."""
        d = self.cam_dirs[cam]
        return self.backup_dir if d == self.root else self.backup_dir / d.name

    @property
    def unsaved(self):
        return sum(len(d) for d in self.dirty)

    def has_file(self, cam, t):
        return 0 <= t < self.n_frames and self.files[cam][t] is not None

    def slot(self, cam, t):
        """Entry of the file holding this person (-1: not in this file)."""
        if self.slots is not None:
            return int(self.slots[cam][t])
        return self.person

    @property
    def needs_matching(self):
        """Several people in some camera, and not matched across cameras."""
        return not self.associated and any((n > 1).any() for n in self.n_people)

    def match_people(self, cameras, threshold=0.3, progress=None):
        """Find person 1 in every camera, frame by frame, with the
        calibration.
        """
        if self.person != 0 or not self.needs_matching:
            return 0
        slots = np.zeros((self.n_cameras, self.n_frames), int)
        previous, changed = None, 0
        for t in range(self.n_frames):
            if progress is not None and t % 25 == 0:
                progress(t, self.n_frames)
            people = []
            for cam in range(self.n_cameras):
                f = self.files[cam][t]
                entries = read_people(f) if f is not None else []
                people.append([person_points(entries, i) for i in range(len(entries))])
            if all(len(p) <= 1 for p in people):
                choice = [0 if len(p) else -1 for p in people]
            else:
                choice = pose_checks.associate_person(people, cameras, threshold, previous)
                if max(choice) < 0:                      # nothing consistent
                    choice = [0 if len(p) else -1 for p in people]
            previous = choice
            slots[:, t] = choice
        changed = int((slots > 0).sum())
        self.slots = slots
        self.matched = True
        self.reload_rows()
        self._load_original()
        return changed

    def follow(self, first):
        """Number the other people around person 1, once person 1 was found in
        every camera.
        """
        if self.person == 0 or first is None or first.slots is None:
            return False
        slots = np.full((self.n_cameras, self.n_frames), -1, int)
        for cam in range(self.n_cameras):
            for t in range(self.n_frames):
                others = [e for e in range(int(self.n_entries[cam][t]))
                          if e != int(first.slots[cam][t])]
                if self.person - 1 < len(others):
                    slots[cam, t] = others[self.person - 1]
        self.slots = slots
        self.reload_rows()
        self._load_original()
        return True

    def usable(self):
        """Per camera, the rows the camera check may use."""
        out = []
        for cam in range(self.n_cameras):
            ok = np.array([f is not None for f in self.files[cam]], bool)
            if self.matched:
                ok &= self.slots[cam] >= 0
            elif not self.associated:
                ok &= self.n_people[cam] <= 1
            out.append(ok)
        return out

    def point(self, cam, frame, kp):
        if not 0 <= frame < self.n_frames:
            return np.zeros(3, np.float32)
        return self.data[cam][frame, kp]

    def frame(self, cam, frame):
        if not 0 <= frame < self.n_frames:
            return np.zeros((self.n_keypoints, 3), np.float32)
        return self.data[cam][frame]

    def changed(self, cam=None):
        """(rows, keypoints) True where the point differs from the original."""
        cams = range(self.n_cameras) if cam is None else [cam]
        out = []
        for c in cams:
            a, b = self.data[c], self.original[c]
            moved = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1) > 0.5
            presence = pose_checks.present(a) != pose_checks.present(b)
            out.append(moved | presence)
        return out if cam is None else out[0]

    # ------------------------------------------------------------------
    def set_point(self, cam, frame, kp, x, y, conf=1.0, record=True):
        if not self.has_file(cam, frame):
            return False
        if record:
            self.undo_stack.append((cam, frame, kp, self.data[cam][frame, kp].copy()))
        self.data[cam][frame, kp] = (x, y, conf)
        self.dirty[cam].add(frame)
        return True

    def clear_point(self, cam, frame, kp):
        return self.set_point(cam, frame, kp, 0.0, 0.0, 0.0)

    def swap_lr(self, cam, frame, a, b):
        if not self.has_file(cam, frame):
            return False
        self.undo_stack.append((cam, frame, a, self.data[cam][frame, a].copy()))
        self.undo_stack.append((cam, frame, b, self.data[cam][frame, b].copy()))
        block = self.data[cam][frame]
        block[[a, b]] = block[[b, a]]
        self.dirty[cam].add(frame)
        return True

    def swap_person(self, other, cam, frame, kp):
        """Exchange one point with another person of the same file. Returns
        "done", "same" or "nothing".
        """
        if not self.has_file(cam, frame):
            return "nothing"
        mine_slot, their_slot = self.slot(cam, frame), other.slot(cam, frame)
        if mine_slot >= 0 and mine_slot == their_slot:
            return "same"
        mine = self.data[cam][frame, kp].copy()
        theirs = other.data[cam][frame, kp].copy()
        if not (pose_checks.present(mine[None, None])[0, 0]
                or pose_checks.present(theirs[None, None])[0, 0]):
            return "nothing"
        # one Undo puts both people back
        self.undo_stack.append((cam, frame, kp, mine, (other, theirs)))
        self.data[cam][frame, kp] = theirs
        other.data[cam][frame, kp] = mine
        self.dirty[cam].add(frame)
        other.dirty[cam].add(frame)
        self.carried.pop((cam, frame, kp), None)
        other.carried.pop((cam, frame, kp), None)
        return "done"

    def group(self, start):
        """Group the undo entries recorded since `start`, so one Undo reverts
        a whole edit.
        """
        entries = self.undo_stack[start:]
        del self.undo_stack[start:]
        if len(entries) == 1:
            self.undo_stack.append(entries[0])
        elif entries:
            self.undo_stack.append(("group", entries))
        return len(entries)

    def _undo_entry(self, entry):
        if entry[0] == "group":
            for sub in reversed(entry[1]):
                self._undo_entry(sub)
            return
        cam, frame, kp, old = entry[:4]
        self.data[cam][frame, kp] = old
        self.dirty[cam].add(frame)
        self.carried.pop((cam, frame, kp), None)
        if len(entry) > 4:                       # a swap with another person
            other, theirs = entry[4]
            other.data[cam][frame, kp] = theirs
            other.dirty[cam].add(frame)

    def undo(self):
        if not self.undo_stack:
            return False
        self._undo_entry(self.undo_stack.pop())
        return True

    # ------------------------------------------------------------------
    def create_backup(self):
        dest = self.backup_dir
        if dest.exists():
            return dest, False
        shutil.copytree(self.root, dest)
        return dest, True

    def backup_count(self):
        if not self.has_backup:
            return 0
        return len(frame_files(self.backup_dir)) + sum(
            len(frame_files(s)) for s in self.backup_dir.iterdir() if s.is_dir())

    def save(self):
        """Write every changed frame of every camera (backup first, once);
        other people are left as they are.
        """
        if not self.unsaved:
            return {"written": 0, "created_backup": False}
        _, created = self.create_backup()
        written = 0
        for cam in range(self.n_cameras):
            for t in sorted(self.dirty[cam]):
                target = self.files[cam][t]
                if target is None:
                    continue
                with open(target, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                people = [p if isinstance(p, dict) else {}
                          for p in (payload.get("people") or [])]
                where = self.slot(cam, t)
                if where < 0:                       # not in this file yet: add
                    where = len(people)
                    if self.slots is not None:
                        self.slots[cam][t] = where
                while len(people) <= where:
                    people.append({})
                entry = dict(people[where])
                entry.setdefault("person_id", [-1])
                entry["pose_keypoints_2d"] = [
                    round(float(v), 3) for v in self.data[cam][t].reshape(-1)]
                people[where] = entry
                payload["people"] = people
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                written += 1
            self.dirty[cam].clear()
        return {"written": written, "created_backup": created}

    def reload_rows(self, rows=None):
        """Read rows back from disk (all rows when None), for this person."""
        rows = range(self.n_frames) if rows is None else rows
        for cam in range(self.n_cameras):
            for t in rows:
                f = self.files[cam][t]
                if f is None:
                    continue
                pts = person_points(read_people(f), self.slot(cam, t))
                self.data[cam][t] = 0
                k = min(pts.shape[0], self.n_keypoints)
                self.data[cam][t, :k] = pts[:k]
                self.dirty[cam].discard(t)
                for key in [k_ for k_ in self.carried if k_[0] == cam and k_[1] == t]:
                    self.carried.pop(key, None)

    def restore_frame(self, frame):
        """Put one frame back from the backup, in every camera."""
        if not self.has_backup:
            return 0
        restored = 0
        for cam in range(self.n_cameras):
            f = self.files[cam][frame]
            if f is None:
                continue
            src = self.cam_backup(cam) / f.name
            if src.exists():
                shutil.copy2(src, f)
                restored += 1
        self.reload_rows([frame])
        return restored

    def restore_all(self):
        if not self.has_backup:
            return 0
        restored = 0
        for cam in range(self.n_cameras):
            src_dir = self.cam_backup(cam)
            if not src_dir.is_dir():
                continue
            for f in self.files[cam]:
                if f is not None and (src_dir / f.name).exists():
                    shutil.copy2(src_dir / f.name, f)
                    restored += 1
        self.undo_stack.clear()
        self.carried.clear()
        self.reload_rows()
        return restored


def find_video(pose_dir, cam_name):
    """The video behind a camera, if one is sitting where Pose2Sim puts it."""
    trial = Path(pose_dir).parent
    for base in (trial / "videos", trial, Path(pose_dir)):
        if not base.is_dir():
            continue
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI"):
            for cand in base.glob("*" + ext):
                if cam_name.lower() in cand.stem.lower():
                    return cand
    return None


# ===========================================================================
# Sports2D: the pixel coordinates of each person, in a TRC file
# ===========================================================================
TRC_PERSON_RE = re.compile(r"_px_person(\d+)\.trc$", re.IGNORECASE)
TRC_PRESENT = 0.99      # TRC files hold no confidence: a detected point gets this


def trc_files(folder):
    """Sports2D's pixel TRC files, one per person, in person order."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    found = [f for f in folder.glob("*.trc") if TRC_PERSON_RE.search(f.name)]
    if not found:                       # a single file, as load_trc_px expects
        found = [f for f in folder.glob("*_px.trc")]

    def key(f):
        m = TRC_PERSON_RE.search(f.name)
        return (int(m.group(1)) if m else 0, f.name)

    return sorted(found, key=key)


def _number_or_nan(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def read_trc_px(path):
    """Read a Sports2D pixel TRC file: header, data cells, marker names, frame
    rate and (x, y).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    if len(lines) < 5:
        raise ValueError(f"{Path(path).name} is not a TRC file.")
    header = lines[:5]
    names = [m.strip() for m in header[3].split("\t")[2::3] if m.strip()]
    try:
        fps = float(header[2].split("\t")[0])
    except (ValueError, IndexError):
        fps = 30.0
    rows = [line.split("\t") for line in lines[5:] if line.strip()]
    xy = np.full((len(rows), len(names), 2), np.nan)
    times = np.full(len(rows), np.nan)
    for r, cells in enumerate(rows):
        times[r] = _number_or_nan(cells[1]) if len(cells) > 1 else np.nan
        for m in range(len(names)):
            c = 2 + 3 * m
            if c + 1 < len(cells):
                xy[r, m] = (_number_or_nan(cells[c]), _number_or_nan(cells[c + 1]))
    return header, rows, names, fps, times, xy


def _cell(value):
    return f"{float(value):.6f}"


class TrcSequence(Sequence):
    """The keypoints of a Sports2D result, as a Sequence of one camera (one
    person per TRC file).
    """

    kind = "trc"

    def _load(self):
        if self.root.is_file():
            self.root = self.root.parent
        files = trc_files(self.root)
        if not files:
            raise FileNotFoundError(
                f"No Sports2D pixel TRC file in {self.root}.\n\n"
                f"Expected files named <video>_Sports2D_px_person00.trc.")
        self.trc_paths = files
        if self.person >= len(files):
            raise FileNotFoundError(f"There is no person {self.person + 1} in "
                                    f"{self.root.name}.")
        self.file = files[self.person]
        header, rows, names, fps, times, xy = read_trc_px(self.file)
        self.header, self.rows, self.fps = header, rows, fps
        self.marker_names = tuple(names)
        n = len(rows)
        numbers = np.rint(times * fps)
        if not np.all(np.isfinite(numbers)) or len(set(numbers.tolist())) != n:
            numbers = np.arange(n)
        self.numbers = [int(v) for v in numbers]
        stem = re.sub(r"_Sports2D.*$", "", self.file.stem, flags=re.I)
        stem = re.sub(r"_px(_person\d+)?$", "", stem, flags=re.I) or self.root.name
        self.cam_dirs, self.cam_names = [self.root], [stem]
        self.files = [[self.file] * n]
        self.n_people = [np.ones(n, int)]
        self.n_entries = [np.ones(n, int)]
        self.dirty = [set()]
        block = np.zeros((n, len(names), 3), np.float32)
        ok = np.isfinite(xy).all(axis=2) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        block[..., :2] = np.where(ok[..., None], xy, 0)
        block[..., 2] = np.where(ok, TRC_PRESENT, 0)
        self.loaded = block.copy()       # as read, to rewrite only what changed
        self.data = [block]
        self.n_keypoints = len(names)
        self.n_frames = n
        self.max_people = len(files)
        self.video_offsets = [0]
        self._load_original()

    @property
    def associated(self):
        return False

    @property
    def backup_dir(self):
        return self.root.with_name(self.root.name + "-backup")

    @property
    def has_backup(self):
        return (self.backup_dir / self.file.name).is_file()

    def cam_backup(self, cam):
        return self.backup_dir

    def _load_original(self):
        self.original = [b.copy() for b in self.data]
        if not self.has_backup:
            return
        try:
            _h, rows, names, _fps, _t, xy = read_trc_px(self.backup_dir / self.file.name)
        except Exception:
            return
        if len(rows) != self.n_frames or tuple(names) != self.marker_names:
            return
        ok = np.isfinite(xy).all(axis=2) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        block = np.zeros_like(self.data[0])
        block[..., :2] = np.where(ok[..., None], xy, 0)
        block[..., 2] = np.where(ok, TRC_PRESENT, 0)
        self.original = [block]

    def usable(self):
        return [np.ones(self.n_frames, bool)]

    def create_backup(self):
        dest = self.backup_dir
        created = False
        dest.mkdir(exist_ok=True)
        for f in self.trc_paths:
            if not (dest / f.name).exists():
                shutil.copy2(f, dest / f.name)
                created = True
        return dest, created

    def backup_count(self):
        return len(list(self.backup_dir.glob("*.trc"))) if self.backup_dir.is_dir() else 0

    def save(self):
        """Write the corrected x and y into the TRC file of this person."""
        if not self.unsaved:
            return {"written": 0, "created_backup": False}
        _, created = self.create_backup()
        block, loaded = self.data[0], self.loaded
        for t in sorted(self.dirty[0]):
            cells = self.rows[t]
            for m in range(self.n_keypoints):
                if np.array_equal(block[t, m], loaded[t, m]):
                    continue
                c = 2 + 3 * m
                while len(cells) <= c + 2:
                    cells.append("")
                if block[t, m, 2] > 0 or block[t, m, 0] != 0 or block[t, m, 1] != 0:
                    cells[c], cells[c + 1] = _cell(block[t, m, 0]), _cell(block[t, m, 1])
                else:
                    cells[c], cells[c + 1] = "", ""
            loaded[t] = block[t]
        with open(self.file, "w", encoding="utf-8", newline="\n") as fh:
            for line in self.header:
                fh.write(line + "\n")
            for cells in self.rows:
                fh.write("\t".join(cells) + "\n")
        written = len(self.dirty[0])
        self.dirty[0].clear()
        return {"written": written, "created_backup": created}

    def reload_rows(self, rows=None):
        """Read the file again (every row), for this person."""
        header, rows_, names, fps, times, xy = read_trc_px(self.file)
        self.header, self.rows = header, rows_
        ok = np.isfinite(xy).all(axis=2) & ~((xy[..., 0] == 0) & (xy[..., 1] == 0))
        block = self.data[0]
        block[..., :2] = np.where(ok[..., None], xy, 0)
        block[..., 2] = np.where(ok, TRC_PRESENT, 0)
        self.loaded = block.copy()
        self.dirty[0].clear()
        self.carried.clear()

    def restore_frame(self, frame):
        return 0

    def restore_all(self):
        """Copy every TRC file of the backup back (every person)."""
        if not self.has_backup:
            return 0
        restored = 0
        for f in self.trc_paths:
            if (self.backup_dir / f.name).is_file():
                shutil.copy2(self.backup_dir / f.name, f)
                restored += 1
        self.undo_stack.clear()
        self.carried.clear()
        self.reload_rows()
        return restored


def is_trc_folder(path):
    path = Path(path)
    if path.is_file():
        return path.suffix.lower() == ".trc"
    return bool(trc_files(path)) and not has_keypoints(path)


def open_sequence(path, person=0):
    """The keypoints of one person: Pose2Sim JSON folders, or Sports2D pixel
    TRC files.
    """
    path = Path(path)
    if is_trc_folder(path):
        return TrcSequence(path.parent if path.is_file() else path, person=person)
    return Sequence(path, person=person)


# ===========================================================================
# The application
# ===========================================================================
class PoseCorrectionApp:
    # ===== APPLICATION CONSTANTS =====

    CONTROL_PANEL_WIDTH = 300
    HIT_RADIUS = 14              # px within which a click grabs a point
    FRAME_CACHE = 40             # decoded video frames kept per camera

    DEFAULT_CONF_THRESHOLD = 0.3
    LOW_CONF_RING = 6            # extra radius of the red ring on bad points

    COLOUR_LEFT = "#3b82f6"
    COLOUR_RIGHT = "#22c55e"
    COLOUR_CENTRE = "#f97316"
    COLOUR_SELECTED = "#facc15"
    COLOUR_EDITED = "#22c55e"
    COLOUR_BAD = "#ef4444"
    COLOUR_JUMP = "#f59e0b"
    COLOUR_CAMERAS = "#c084fc"
    COLOUR_CARRIED = "#38bdf8"
    COLOUR_BG = "#14171a"
    COLOUR_WARN = "#b45309"
    COLOUR_ORIGINAL = "#e5e7eb"

    MODE_CORRECTED = "Corrected"
    MODE_BOTH = "Corrected + original"
    MODE_ORIGINAL = "Original"

    # Distances are shown in cm: % of the person's height times
    # ASSUMED_HEIGHT_CM (REFERENCE_CM for hands and faces).
    ASSUMED_HEIGHT_CM = 170.0
    REFERENCE_CM = {"HAND_21": 18.0, "FACE_106": 18.0}
    # Default limits, editable in the panel.
    DEFAULT_JUMP_LIMIT = 15.0    # cm, in one frame
    DEFAULT_CAMERA_LIMIT = 10.0  # cm between the point and where the other cameras see it
    DEFAULT_FEET_LIMIT = 6.0     # the same, for ankles, heels and toes
    FEET_WORDS = ("Ankle", "Heel", "BigToe", "SmallToe")
    # Weight: frames on each side moved with an edit (0: this frame only).
    DEFAULT_WEIGHT = 0
    MAX_WEIGHT = 300
    SPREAD_CONFIDENCE = 0.9      # at least, for points moved by the weight
    PLOT_HEIGHT = 170            # px, trajectory plot under the cameras
    PLOT_SPAN = 120              # frames shown on each side of the current one

    # how the suspect points are listed; each key gives the row order
    SORT_ORDERS = {
        "Most suspect": lambda p: (-p[4], p[2]),
        "Frame": lambda p: (p[2], p[0], p[1]),
        "Camera": lambda p: (p[0], p[2], p[1]),
        "Keypoint": lambda p: (p[1], p[2], p[0]),
    }
    DEFAULT_SORT = "Most suspect"
    MAX_LISTED = 2000            # rows shown in the suspect list

    def __init__(self, root, pose_dir=None):
        self.root = root
        self.root.title("Pose2Sim Keypoint Correction Tool")

        self.root.grid_columnconfigure(0, weight=0)   # control panel
        self.root.grid_columnconfigure(1, weight=1)   # camera views
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)      # status bar

        self.seq = None
        self.pose_dir = None
        self.frame_index = 0
        self.keypoint = 0
        self.active_cam = 0
        self.names, self.links, self.left, self.right = skeleton_for(26)
        self.partners = lr_partners(self.names)
        self.model = "HALPE_26"
        self.reference_cm = self.ASSUMED_HEIGHT_CM

        self.views = []          # one dict per camera: canvas, transform, video
        self.status_text = tk.StringVar(value="Open a keypoint folder to begin.")
        self.conf_threshold = tk.DoubleVar(value=self.DEFAULT_CONF_THRESHOLD)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_all_points = tk.BooleanVar(value=True)
        self.flag_low_conf = tk.BooleanVar(value=True)
        self.weight = tk.IntVar(value=self.DEFAULT_WEIGHT)
        self.follow_point = tk.BooleanVar(value=True)  # zoomed views follow the selected point
        self.person_var = tk.StringVar(value="1")
        self.flag_jumps = tk.BooleanVar(value=True)
        self.jump_limit = tk.DoubleVar(value=self.DEFAULT_JUMP_LIMIT)
        self.flag_cameras = tk.BooleanVar(value=True)
        self.camera_limit = tk.DoubleVar(value=self.DEFAULT_CAMERA_LIMIT)
        self.feet_limit = tk.DoubleVar(value=self.DEFAULT_FEET_LIMIT)
        self.cameras = None      # calibration cameras, paired as Pose2Sim pairs them
        self.calib_file = None
        self.heights = []        # person height in px, per camera and frame
        self.sequences = {}      # person index -> Sequence, edits kept per person
        self.view_mode = tk.StringVar(value=self.MODE_BOTH)
        self.show_original = tk.BooleanVar(value=True)
        self.calib_details = ""  # how well the cameras agree, shown on hover
        self.sort_by = tk.StringVar(value=self.DEFAULT_SORT)
        self.problems = []
        self.shown = []          # rows of the suspect list: (index in problems, frame)
        self.swap_target = tk.StringVar(value="2")
        self.edit = None         # the edit in progress (see begin_edit)
        self.plot_extra = []     # other points drawn on the trajectory plot
        self.plot_span = self.PLOT_SPAN
        self.plot_drag = None
        self.plot_map = None     # how the plot maps frames and values to pixels

        self.build_ui()
        if pose_dir:
            self.load_folder(pose_dir)

    # ==================================================================
    # UI
    # ==================================================================
    def build_ui(self):
        self.build_control_panel()
        self.build_camera_area()
        self.build_status_bar()
        self.bind_keys()

    def build_control_panel(self):
        # The panel scrolls, so every box stays reachable on small screens.
        outer = ttk.Frame(self.root, width=self.CONTROL_PANEL_WIDTH)
        outer.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        outer.grid_propagate(False)
        outer.pack_propagate(False)
        background = ttk.Style().lookup("TFrame", "background") or None
        canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0,
                           width=self.CONTROL_PANEL_WIDTH - 18, background=background)
        bar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(window, width=e.width))

        def wheel(event):
            # only over the panel, and not over the list, which scrolls itself
            widget = str(event.widget)
            if not widget.startswith(str(outer)) or isinstance(event.widget, tk.Listbox):
                return
            if canvas.yview() == (0.0, 1.0):
                return
            step = -1 if (getattr(event, "delta", 0) > 0 or event.num == 4) else 1
            canvas.yview_scroll(step, "units")

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(sequence, wheel, add="+")

        ttk.Label(panel, text="Keypoint Correction",
                  font=("Helvetica", 15, "bold")).pack(anchor="w", pady=(0, 8))

        def limit_row(parent, text, var, low, high, step, indent=8, check=None):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=(indent, 8), pady=(0, 4))
            if check is None:
                ttk.Label(row, text=text).pack(side="left")
            else:
                ttk.Checkbutton(row, text=text, variable=check,
                                command=self.redraw_all).pack(side="left")
            ttk.Spinbox(row, from_=low, to=high, increment=step, width=5,
                        textvariable=var, command=self.redraw_all).pack(side="right")
            return row

        # ---- keypoints: what to open, and what to flag
        box = self._section(panel, "Keypoints", "keypoints")
        box.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(8, 6))
        ttk.Label(row, text="Open").pack(side="left")
        self._tip(ttk.Button(row, text="Pose2Sim project…",
                             command=self.choose_project), "open_project").pack(side="right")
        self._tip(ttk.Button(row, text="JSON folder…",
                             command=self.choose_folder), "open_folder").pack(side="right", padx=(6, 4))
        # shown only when the folder holds several people
        self.person_row = ttk.Frame(box)
        ttk.Label(self.person_row, text="Person").pack(side="left")
        self.person_combo = ttk.Combobox(self.person_row, state="readonly", width=5,
                                         textvariable=self.person_var, values=["1"])
        self.person_combo.pack(side="right")
        self.person_combo.bind("<<ComboboxSelected>>", self.on_person_selected)

        self.first_flag = limit_row(box, "Flag low confidence below",
                                    self.conf_threshold, 0.0, 1.0, 0.05,
                                    check=self.flag_low_conf)
        self._info(self.first_flag, "low_conf")
        self._info(limit_row(box, "Flag jumps above (approx., cm)", self.jump_limit,
                             1, 300, 1, check=self.flag_jumps), "jumps")
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Checkbutton(row, text="Flag camera disagreement", variable=self.flag_cameras,
                        command=self.redraw_all).pack(side="left")
        self._info(row, "cameras")
        row = ttk.Frame(box)
        row.pack(fill="x", padx=(28, 8), pady=(0, 4))
        ttk.Label(row, text="Calibration").pack(side="left")
        self._tip(ttk.Button(row, text="Browse…", width=8,
                             command=self.choose_calibration), "calibration").pack(side="right")
        self.calib_label = ttk.Label(row, text="none")
        self.calib_label.pack(side="right", padx=(0, 6))
        # hovering the file name tells how well the cameras agree
        self.calib_label.bind("<Enter>", lambda e: self.calib_details
                              and self.status_text.set(self.calib_details))
        limit_row(box, "Feet points above (approx., cm)", self.feet_limit,
                  1, 300, 1, indent=28)
        limit_row(box, "Other points above (approx., cm)", self.camera_limit,
                  1, 300, 1, indent=28)
        # a warning about the setup, shown only when there is one
        self.calib_health = ttk.Label(box, text="", foreground=self.COLOUR_WARN,
                                      wraplength=262, justify="left")
        self.scan_button = self._tip(ttk.Button(box, text="Scan for suspect points",
                                                command=self.scan), "scan")
        self.scan_button.pack(fill="x", padx=8, pady=(6, 8))

        # ---- suspect points
        self.suspect_box = self._section(panel, "Suspect points", "suspect")
        self.suspect_box.pack(fill="x", pady=(0, 10))
        box = self.suspect_box
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(8, 6))
        ttk.Label(row, text="Sort by").pack(side="left")
        order = ttk.Combobox(row, state="readonly", width=14, textvariable=self.sort_by,
                             values=list(self.SORT_ORDERS))
        order.pack(side="right")
        order.bind("<<ComboboxSelected>>", lambda e: self.fill_problem_list())
        legend = ttk.Frame(box)
        legend.pack(fill="x", padx=8)
        for colour, text in ((self.COLOUR_BAD, "low confidence"),
                             (self.COLOUR_JUMP, "jump"),
                             (self.COLOUR_CAMERAS, "cameras disagree"),
                             (self.COLOUR_CARRIED, "carried"),
                             (self.COLOUR_EDITED, "edited")):
            line = ttk.Frame(legend)
            line.pack(anchor="w")
            dot = tk.Canvas(line, width=12, height=12, highlightthickness=0)
            dot.create_oval(2, 2, 10, 10, outline=colour, width=2)
            dot.pack(side="left")
            ttk.Label(line, text=text).pack(side="left", padx=4)
        ttk.Label(box, text="Hover over a point to see why it is ringed.",
                  foreground="gray", wraplength=262).pack(anchor="w", padx=8, pady=(2, 6))
        wrap = ttk.Frame(box)
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.problem_list = tk.Listbox(wrap, height=10, activestyle="none",
                                       exportselection=False)
        bar = ttk.Scrollbar(wrap, command=self.problem_list.yview)
        self.problem_list.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.problem_list.pack(fill="both", expand=True)
        self.problem_list.bind("<<ListboxSelect>>", self.on_problem_selected)

        # ---- edit the selected point
        box = self._section(panel, "Edit selected point", "edit")
        box.pack(fill="x", pady=(0, 10))
        ttk.Label(box, text="Click a point on an image, then drag it or use these. "
                            "A right click clears it.",
                  wraplength=262, justify="left").pack(anchor="w", padx=8, pady=(8, 6))
        self._tip(ttk.Button(box, text="Place where other cameras see it",
                             command=self.place_proposed), "place").pack(fill="x", padx=8, pady=(0, 4))
        self._tip(ttk.Button(box, text="Swap left / right",
                             command=self.swap_lr), "swap_lr").pack(fill="x", padx=8, pady=(0, 4))
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(0, 4))
        self._tip(ttk.Button(row, text="Swap with person",
                             command=self.swap_person), "swap_person").pack(side="left", fill="x", expand=True)
        self.swap_combo = ttk.Combobox(row, state="readonly", width=4,
                                       textvariable=self.swap_target, values=["2"])
        self.swap_combo.pack(side="right", padx=(4, 0))
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(2, 8))
        ttk.Label(row, text="Weight (frames)").pack(side="left")
        self._info(row, "weight")
        ttk.Spinbox(row, from_=0, to=self.MAX_WEIGHT, width=5,
                    textvariable=self.weight,
                    command=self.draw_plot).pack(side="right")

        # ---- history
        self.history_box = self._section(panel, "History", "history")
        self.history_box.pack(fill="x")
        box = self.history_box
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=(8, 4))
        self._tip(ttk.Button(row, text="Undo", command=self.undo), "undo").pack(
            side="left", fill="x", expand=True)
        self._tip(ttk.Button(row, text="Delete selected", command=self.delete_correction),
                  "delete").pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._tip(ttk.Button(box, text="Restore everything", command=self.restore_all),
                  "restore").pack(fill="x", padx=8, pady=(0, 4))
        self._tip(ttk.Button(box, text="Save all corrections  (Ctrl+S)",
                             command=self.save), "save").pack(fill="x", padx=8, pady=(0, 6))
        self._tip(ttk.Checkbutton(box, text="Show original positions  (key O)",
                                  variable=self.show_original,
                                  command=self.on_show_original), "original").pack(anchor="w", padx=8)
        wrap = ttk.Frame(box)
        wrap.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.correction_list = tk.Listbox(wrap, height=6, activestyle="none",
                                          exportselection=False)
        bar = ttk.Scrollbar(wrap, command=self.correction_list.yview)
        self.correction_list.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.correction_list.pack(fill="both", expand=True)
        self.correction_list.bind("<<ListboxSelect>>", self.on_correction_selected)

    # (i) tooltips from correction_tips.py; without that module, no
    # explanations.
    def _tip(self, widget, key):
        if tips is not None:
            tips.tip(widget, key)
        return widget

    def _info(self, row, key):
        if tips is not None:
            tips.info(row, key).pack(side="left", padx=(4, 0))
        return row

    def _section(self, parent, title, key):
        if tips is None:
            return ttk.LabelFrame(parent, text=title)
        return tips.section(parent, title, key)

    def build_camera_area(self):
        area = ttk.Frame(self.root)
        area.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        area.grid_rowconfigure(0, weight=1)
        area.grid_columnconfigure(0, weight=1)

        self.camera_frame = ttk.Frame(area)
        self.camera_frame.grid(row=0, column=0, sticky="nsew")

        self.placeholder = ttk.Label(
            self.camera_frame, anchor="center", justify="center",
            text=("No keypoints loaded.\n\n"
                  "Open the folder holding your camera subfolders.\n"
                  "Files may be named cam01_000000.json or "
                  "cam01_000000_keypoints.json — both are read."))
        self.placeholder.pack(expand=True)

        # ---- trajectory of the selected point
        plot = ttk.Frame(area)
        plot.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        head = ttk.Frame(plot)
        head.pack(fill="x")
        self.plot_title = ttk.Label(head, text="Trajectory")
        self.plot_title.pack(side="left")
        ttk.Button(head, text="Clear", width=6,
                   command=self.clear_plot_points).pack(side="right")
        self.plot_add = ttk.Combobox(head, state="readonly", width=14,
                                     values=list(self.names))
        self.plot_add.set("Add a point")
        self.plot_add.pack(side="right", padx=(0, 4))
        self.plot_add.bind("<<ComboboxSelected>>", self.add_plot_point)
        self.plot = tk.Canvas(plot, height=self.PLOT_HEIGHT, bg=self.COLOUR_BG,
                              highlightthickness=1, highlightbackground="#3a4046")
        self.plot.pack(fill="x", pady=(4, 0))
        self.plot.bind("<Configure>", lambda e: self.draw_plot())
        self.plot.bind("<Button-1>", self.on_plot_press)
        self.plot.bind("<B1-Motion>", self.on_plot_drag)
        self.plot.bind("<ButtonRelease-1>", self.on_plot_release)
        self.plot.bind("<MouseWheel>", self.on_plot_wheel)
        self.plot.bind("<Button-4>", lambda e: self.on_plot_wheel(e, 1))
        self.plot.bind("<Button-5>", lambda e: self.on_plot_wheel(e, -1))

        # ---- transport
        transport = ttk.Frame(area)
        transport.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(transport, text="|◀", width=4,
                   command=lambda: self.seek(0)).pack(side="left")
        ttk.Button(transport, text="◀", width=4,
                   command=lambda: self.seek(self.frame_index - 1)).pack(side="left")
        ttk.Button(transport, text="▶", width=4,
                   command=lambda: self.seek(self.frame_index + 1)).pack(side="left")

        self.frame_label = ttk.Label(transport, text="frame 0 / 0", width=18)
        self.frame_label.pack(side="left", padx=10)

        self.slider = ttk.Scale(transport, from_=0, to=0, orient="horizontal",
                                command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        # navigation in zoomed views: follow the point, show the whole image
        ttk.Button(transport, text="Fit  (F)", width=8,
                   command=self.fit_all).pack(side="right")
        ttk.Checkbutton(transport, text="Follow the point",
                        variable=self.follow_point).pack(side="right", padx=(0, 8))

    def build_status_bar(self):
        bar = ttk.Frame(self.root)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(bar, textvariable=self.status_text, anchor="w",
                  padding=(10, 6)).pack(side="left", fill="x", expand=True)
        self.readout = ttk.Label(bar, text="", anchor="e", padding=(10, 6),
                                 font=("Consolas", 9))
        self.readout.pack(side="right")

    def bind_keys(self):
        self.root.bind("<Left>", lambda e: self.seek(self.frame_index - 1))
        self.root.bind("<Right>", lambda e: self.seek(self.frame_index + 1))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-s>", lambda e: self.save())
        self.root.bind("<Delete>", lambda e: self.clear_point())
        self.root.bind("<KeyPress-o>", lambda e: self.toggle_original())
        self.root.bind("<KeyPress-O>", lambda e: self.toggle_original())
        self.root.bind("<KeyPress-f>", self.on_fit_key)
        self.root.bind("<KeyPress-F>", self.on_fit_key)

    # ==================================================================
    # Loading
    # ==================================================================
    def choose_folder(self):
        path = filedialog.askdirectory(
            title="Select the folder holding the camera subfolders, or a "
                  "Sports2D results folder", initialdir=self.start_dir())
        if not path:
            return
        folder = Path(path)
        if (has_keypoints(folder) or is_trc_folder(folder)
                or any(has_keypoints(d) for d in folder.iterdir() if d.is_dir())):
            self.load_folder(path)
        else:
            self.open_found(path)

    def start_dir(self):
        """Where the folder choosers open."""
        if self.pose_dir is not None:
            return str(Path(self.pose_dir).parent)
        return str(Path.cwd())

    def choose_project(self):
        path = filedialog.askdirectory(title="Select the project folder",
                                       initialdir=self.start_dir())
        if not path:
            return
        self.open_found(path)

    def open_found(self, path):
        found = find_pose_folders(path)
        if not found:
            messagebox.showinfo(
                "Nothing to correct",
                f"No keypoint folders (Pose2Sim) and no pixel TRC files "
                f"(Sports2D) under\n{path}\n\nRun pose estimation first.")
            return
        if len(found) == 1:
            self.load_folder(found[0])
            return
        self.pick_from(found)

    def pick_from(self, folders):
        win = tk.Toplevel(self.root)
        win.title("Which recording?")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text="Several recordings have keypoints:",
                  padding=12).pack(anchor="w")
        listbox = tk.Listbox(win, width=80, height=min(len(folders), 12))
        for f in folders:
            listbox.insert("end", str(f))
        listbox.selection_set(0)
        listbox.pack(padx=12, pady=(0, 12))

        def take():
            sel = listbox.curselection()
            win.destroy()
            if sel:
                self.load_folder(folders[sel[0]])

        ttk.Button(win, text="Open", command=take).pack(pady=(0, 12))

    def load_folder(self, pose_dir):
        pose_dir = Path(pose_dir)
        ahead = self.shadowing(pose_dir)
        if ahead:
            names = ", ".join(d.name for d in ahead)
            if messagebox.askyesno(
                    "This folder would be ignored",
                    f"You picked '{pose_dir.name}', but {names} also exists.\n\n"
                    f"Triangulation reads {ahead[0].name} in preference, so "
                    f"corrections here would have no effect.\n\n"
                    f"Open {ahead[0].name} instead?"):
                pose_dir = ahead[0]

        try:
            seq = open_sequence(pose_dir)
        except Exception as exc:
            messagebox.showerror("Could not read that folder", str(exc))
            return

        self.seq = seq
        self.sequences = {0: seq}
        self.pose_dir = pose_dir
        self.set_model()
        self.frame_index = 0
        self.keypoint = 0
        self.active_cam = 0

        self.root.title(f"Pose2Sim Keypoint Correction  ·  {self.folder_text()}")
        self.person_combo.configure(
            values=[str(i + 1) for i in range(max(seq.max_people, 1))])
        self.person_var.set("1")
        if seq.max_people > 1:
            self.person_row.pack(fill="x", padx=8, pady=(0, 6), before=self.first_flag)
        else:
            self.person_row.pack_forget()
        self.update_swap_targets()
        self.slider.configure(to=max(seq.n_frames - 1, 0))

        self.load_calibration(pose_checks.find_calibration(pose_dir))
        self.measure_heights()
        self.build_camera_views()
        self.show_frame(0)
        model = f" ({self.model})" if self.model else ""
        self.status_text.set(
            f"{seq.n_cameras} camera(s) · {seq.n_frames} frames · "
            f"{seq.n_keypoints} keypoints{model} · drag a point to fix it")
        self.scan()
        self.refresh_corrections()

    def set_model(self):
        """Names, lines and sides of the points, from the pose model, the TRC
        names, or the number of points.
        """
        seq = self.seq
        _path, config = project_config(seq.root)
        configured = normalise_model((config.get("pose") or {}).get("pose_model", "")) \
            if config else ""
        counts = {"HALPE_26": 26, "COCO_17": 17, "BODY_25": 25, "COCO_133": 133,
                  "HAND_21": 21, "FACE_106": 106, "ANIMAL2D_17": 17}
        names = getattr(seq, "marker_names", None)
        model = configured
        if names is None and model in counts and counts[model] != seq.n_keypoints:
            model = ""                  # the files were not written by that model
        if not model:
            model = MODEL_BY_COUNT.get(seq.n_keypoints, "")
        self.names, self.links, self.left, self.right = skeleton_for(
            seq.n_keypoints, model or None, names, config)
        self.partners = lr_partners(self.names)
        self.model = model
        self.reference_cm = self.REFERENCE_CM.get(model, self.ASSUMED_HEIGHT_CM)
        pose_checks.EXTENT = "largest" if model in self.REFERENCE_CM else "vertical"
        pose_checks.HEIGHT_NAMES = list(self.names)

    @staticmethod
    def shadowing(pose_dir):
        pose_dir = Path(pose_dir)
        if pose_dir.name not in POSE_ORDER:
            return []
        trial = pose_dir.parent
        ahead = POSE_ORDER[: POSE_ORDER.index(pose_dir.name)]
        return [trial / n for n in ahead
                if (trial / n).is_dir()
                and any(has_keypoints(s) for s in (trial / n).iterdir() if s.is_dir())]

    def build_camera_views(self):
        for child in self.camera_frame.winfo_children():
            child.destroy()
        self.views = []

        n = self.seq.n_cameras
        cols = 1 if n == 1 else (2 if n <= 4 else 3)
        rows = (n + cols - 1) // cols
        for r in range(rows):
            self.camera_frame.grid_rowconfigure(r, weight=1)
        for c in range(cols):
            self.camera_frame.grid_columnconfigure(c, weight=1)

        for i, name in enumerate(self.seq.cam_names):
            holder = ttk.Frame(self.camera_frame)
            holder.grid(row=i // cols, column=i % cols, sticky="nsew", padx=3, pady=3)
            canvas = tk.Canvas(holder, bg=self.COLOUR_BG, highlightthickness=1,
                               highlightbackground="#3a4046", cursor="crosshair")
            canvas.pack(fill="both", expand=True)

            view = {
                "index": i, "name": name, "canvas": canvas,
                "scale": 1.0, "ox": 0.0, "oy": 0.0,
                "img_w": 1920.0, "img_h": 1080.0, "fitted": False,
                "video": None, "cache": {}, "order": [], "photo": None,
                "drag": None, "pan": None,
            }
            self.attach_video(view)
            self.bind_canvas(view)
            self.views.append(view)

        self.root.after(80, self.fit_all)

    def attach_video(self, view):
        if not (OPENCV_AVAILABLE and PIL_AVAILABLE):
            return
        path = find_video(self.pose_dir, view["name"])
        if path is None:
            return
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            view["video"] = cap
            view["img_w"] = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            view["img_h"] = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    # ==================================================================
    # Canvas interaction
    # ==================================================================
    def bind_canvas(self, view):
        c = view["canvas"]
        c.bind("<Configure>", lambda e, v=view: self.on_resize(v))
        c.bind("<Button-1>", lambda e, v=view: self.on_press(v, e))
        c.bind("<B1-Motion>", lambda e, v=view: self.on_drag(v, e))
        c.bind("<ButtonRelease-1>", lambda e, v=view: self.on_release(v, e))
        c.bind("<Button-3>", lambda e, v=view: self.on_right_click(v, e))
        c.bind("<Button-2>", lambda e, v=view: self.on_pan_start(v, e))
        c.bind("<B2-Motion>", lambda e, v=view: self.on_pan(v, e))
        c.bind("<ButtonRelease-2>", lambda e, v=view: self.on_release(v, e))
        c.bind("<MouseWheel>", lambda e, v=view: self.on_wheel(v, e))
        c.bind("<Button-4>", lambda e, v=view: self.on_wheel(v, e, 1))
        c.bind("<Button-5>", lambda e, v=view: self.on_wheel(v, e, -1))
        c.bind("<Motion>", lambda e, v=view: self.on_hover(v, e))
        c.bind("<Leave>", lambda e, v=view: v["canvas"].delete("tip"))

    def to_screen(self, view, x, y):
        return x * view["scale"] + view["ox"], y * view["scale"] + view["oy"]

    def to_image(self, view, sx, sy):
        return ((sx - view["ox"]) / view["scale"],
                (sy - view["oy"]) / view["scale"])

    def fit(self, view):
        c = view["canvas"]
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4:
            return
        view["scale"] = min(w / view["img_w"], h / view["img_h"])
        view["ox"] = (w - view["img_w"] * view["scale"]) / 2
        view["oy"] = (h - view["img_h"] * view["scale"]) / 2
        view["fitted"] = True
        self.redraw(view)

    def fit_all(self):
        for view in self.views:
            self.fit(view)

    def on_fit_key(self, event):
        """F shows the whole image again, unless you are typing a value."""
        if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        self.fit_all()

    def zoomed(self, view):
        """True when the view shows only part of the image."""
        c = view["canvas"]
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4:
            return False
        return view["scale"] > 1.05 * min(w / view["img_w"], h / view["img_h"])

    def point_at(self, view, t):
        """The selected point in this view at row t, or None when missing."""
        if self.seq is None or not 0 <= t < self.seq.n_frames:
            return None
        p = self.seq.frame(view["index"], t)[self.keypoint]
        if p[2] <= 0 and p[0] == 0 and p[1] == 0:
            return None
        return float(p[0]), float(p[1])

    def follow_views(self, old, new):
        """Follow the point: in each zoomed view, keep the selected point at
        its place on screen.
        """
        if not self.follow_point.get() or old == new:
            return
        for view in self.views:
            if not self.zoomed(view):
                continue
            p = self.point_at(view, new)
            if p is None:
                continue
            c = view["canvas"]
            w, h = c.winfo_width(), c.winfo_height()
            sx, sy = w / 2, h / 2
            q = self.point_at(view, old)
            if q is not None:
                qx, qy = self.to_screen(view, *q)
                if 0.1 * w <= qx <= 0.9 * w and 0.1 * h <= qy <= 0.9 * h:
                    sx, sy = qx, qy
            view["ox"] = sx - p[0] * view["scale"]
            view["oy"] = sy - p[1] * view["scale"]

    def on_resize(self, view):
        if view["fitted"]:
            self.redraw(view)
        else:
            self.fit(view)

    def nearest(self, view, sx, sy):
        if self.seq is None:
            return -1, 1e9
        pts = self.seq.frame(view["index"], self.frame_index)
        best, best_d = -1, 1e9
        for i in range(len(pts)):
            if pts[i, 2] <= 0 and pts[i, 0] == 0 and pts[i, 1] == 0:
                continue
            x, y = self.to_screen(view, pts[i, 0], pts[i, 1])
            d = float(np.hypot(x - sx, y - sy))
            if d < best_d:
                best, best_d = i, d
        return best, best_d

    def editable(self, cam):
        """Edits go on the corrected points, in a frame that has a file."""
        if self.view_mode.get() == self.MODE_ORIGINAL:
            self.status_text.set("You are looking at the original points. Press O "
                                 "to edit.")
            return False
        if not self.seq.has_file(cam, self.frame_index):
            self.status_text.set(
                f"{self.seq.cam_names[cam]} has no keypoint file for frame "
                f"{self.frame_number(self.frame_index)}: nothing to correct there.")
            return False
        return True

    def on_press(self, view, event):
        if self.seq is None:
            return
        if self.active_cam != view["index"]:
            self.active_cam = view["index"]
            self.draw_plot()
        i, d = self.nearest(view, event.x, event.y)
        hit = i >= 0 and d <= self.HIT_RADIUS
        if hit:
            # show the point clicked: selected in every camera, named on the
            # image, described in Editing and highlighted in Suspect points
            self.select_keypoint(i)
            listed = self.show_in_problem_list(view["index"], i, self.frame_index)
            self.status_text.set(
                f"Selected {self.names[i]} in {view['name']}, frame "
                f"{self.frame_number(self.frame_index)}"
                + ("  (in the suspect list)" if listed else ""))
        if not hit:
            # pressed on the image, not on a point: drag to move the image
            view["pan"] = (event.x, event.y)
            view["canvas"].configure(cursor="fleur")
            return
        if not self.editable(view["index"]):
            return
        view["drag"] = {"kp": i, "moved": False}
        self.begin_edit(view["index"], i, self.frame_index)

    def on_drag(self, view, event):
        if view["pan"] and not view["drag"]:
            self.on_pan(view, event)
            return
        if not view["drag"] or self.seq is None or self.edit is None:
            return
        x, y = self.to_image(view, event.x, event.y)
        # a hand-placed point is certain, so it is written with confidence 1.0
        self.apply_edit(float(x), float(y))
        view["drag"]["moved"] = True
        self.redraw(view)
        self.draw_plot()
        self.update_readout()

    def on_release(self, view, event):
        if view["pan"]:
            view["pan"] = None
            view["canvas"].configure(cursor="crosshair")
        if view["drag"] and view["drag"]["moved"]:
            self.finish_edit()
            self.redraw_all()
            self.refresh_corrections()
        else:
            self.edit = None
        view["drag"] = None

    def on_right_click(self, view, event):
        if self.seq is None or not self.editable(view["index"]):
            return
        i, d = self.nearest(view, event.x, event.y)
        if i >= 0 and d <= self.HIT_RADIUS:
            self.seq.clear_point(view["index"], self.frame_index, i)
            self.redraw(view)
            self.refresh_corrections()
            self.status_text.set(f"Cleared {self.names[i]} in {view['name']}")

    def on_pan_start(self, view, event):
        view["pan"] = (event.x, event.y)

    def on_pan(self, view, event):
        if not view["pan"]:
            return
        view["ox"] += event.x - view["pan"][0]
        view["oy"] += event.y - view["pan"][1]
        view["pan"] = (event.x, event.y)
        self.redraw(view)

    def on_wheel(self, view, event, direction=None):
        if direction is None:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1
        factor = 1.15 if direction > 0 else 1 / 1.15
        ix, iy = self.to_image(view, event.x, event.y)
        view["scale"] = float(np.clip(view["scale"] * factor, 0.05, 40.0))
        view["ox"] = event.x - ix * view["scale"]
        view["oy"] = event.y - iy * view["scale"]
        self.redraw(view)

    # ==================================================================
    # Drawing
    # ==================================================================
    def video_frame(self, view, index):
        if view["video"] is None:
            return None
        if index in view["cache"]:
            return view["cache"][index]
        try:
            view["video"].set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = view["video"].read()
            if not ok:
                return None
            image = Image.fromarray(frame[:, :, ::-1])
        except Exception:
            return None
        view["cache"][index] = image
        view["order"].append(index)
        while len(view["order"]) > self.FRAME_CACHE:
            view["cache"].pop(view["order"].pop(0), None)
        return image

    def side_colour(self, i):
        if i in self.left:
            return self.COLOUR_LEFT
        if i in self.right:
            return self.COLOUR_RIGHT
        return self.COLOUR_CENTRE

    def redraw_all(self):
        for view in self.views:
            self.redraw(view)
        self.draw_plot()
        self.update_readout()

    def redraw(self, view):
        c = view["canvas"]
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4 or self.seq is None:
            return

        # background: the video frame if we have one, else a grid so the
        # transform is still readable
        image = self.video_frame(view, self.video_index(view["index"], self.frame_index))
        if image is not None:
            try:
                x0, y0 = self.to_image(view, 0, 0)
                x1, y1 = self.to_image(view, w, h)
                cx0, cy0 = max(0, int(x0)), max(0, int(y0))
                cx1 = min(int(view["img_w"]), int(x1) + 1)
                cy1 = min(int(view["img_h"]), int(y1) + 1)
                if cx1 > cx0 and cy1 > cy0:
                    crop = image.crop((cx0, cy0, cx1, cy1))
                    tw = max(1, int((cx1 - cx0) * view["scale"]))
                    th = max(1, int((cy1 - cy0) * view["scale"]))
                    crop = crop.resize((tw, th), Image.BILINEAR)
                    view["photo"] = ImageTk.PhotoImage(crop)
                    sx, sy = self.to_screen(view, cx0, cy0)
                    c.create_image(sx, sy, image=view["photo"], anchor="nw")
            except Exception:
                pass
        else:
            step = 100 * view["scale"]
            if step > 12:
                x = view["ox"] % step
                while x < w:
                    c.create_line(x, 0, x, h, fill="#23272b")
                    x += step
                y = view["oy"] % step
                while y < h:
                    c.create_line(0, y, w, y, fill="#23272b")
                    y += step

        cam, t = view["index"], self.frame_index
        mode = self.view_mode.get()
        current = self.seq.frame(cam, t)
        before = (self.seq.original[cam][t] if 0 <= t < self.seq.n_frames
                  else np.zeros_like(current))
        pts = before if mode == self.MODE_ORIGINAL else current
        present = (pts[:, 2] > 0) | (pts[:, 0] != 0) | (pts[:, 1] != 0)
        threshold = float(self.conf_threshold.get())
        if mode == self.MODE_ORIGINAL:
            checks = self.checks(cam, -1)            # no rings on the original
        else:
            checks = self.checks(cam, t)

        if self.show_skeleton.get():
            width = max(2, min(4, int(view["scale"] * 2)))
            for a, b in self.links:
                if a >= len(pts) or b >= len(pts):
                    continue
                if pts[a, 2] <= 0 or pts[b, 2] <= 0:
                    continue
                xa, ya = self.to_screen(view, pts[a, 0], pts[a, 1])
                xb, yb = self.to_screen(view, pts[b, 0], pts[b, 1])
                colour = self.side_colour(a if a in self.left or a in self.right else b)
                c.create_line(xa, ya, xb, yb, fill=colour, width=width,
                              capstyle="round")

        for i in range(len(pts)):
            if not present[i]:
                continue
            if not self.show_all_points.get() and i != self.keypoint:
                continue
            x, y = self.to_screen(view, pts[i, 0], pts[i, 1])
            selected = i == self.keypoint
            radius = 7 if selected else 4.5

            if self.flag_low_conf.get() and 0 < pts[i, 2] < threshold:
                r = radius + self.LOW_CONF_RING
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_BAD, width=2)
            # one ring per reason, each a little wider, so several can show
            if checks["jump_flag"][i]:
                r = radius + self.LOW_CONF_RING + 3
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_JUMP, width=2)
            if checks["camera_flag"][i]:
                r = radius + self.LOW_CONF_RING + 6
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_CAMERAS, width=2)
                px, py = checks["proposed"][i]
                if np.isfinite(px) and np.isfinite(py):
                    qx, qy = self.to_screen(view, px, py)
                    c.create_line(x, y, qx, qy, fill=self.COLOUR_CAMERAS, dash=(3, 3))
                    c.create_line(qx - 8, qy, qx + 8, qy, fill=self.COLOUR_CAMERAS, width=3)
                    c.create_line(qx, qy - 8, qx, qy + 8, fill=self.COLOUR_CAMERAS, width=3)
            if (view["index"], self.frame_index, i) in self.seq.carried:
                r = radius + 3
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_CARRIED, width=2, dash=(4, 2))
            if selected:
                r = radius + 5
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_SELECTED, width=2)
                # its name beside it, so the selected point is easy to find
                c.create_text(x + r + 4, y - r, text=self.names[i], anchor="sw",
                              fill=self.COLOUR_SELECTED, font=("Helvetica", 9, "bold"))

            if selected:
                fill = self.COLOUR_SELECTED
            elif pts[i, 2] >= 0.999:
                fill = self.COLOUR_EDITED          # placed by hand
            elif pts[i, 2] < threshold:
                fill = self.COLOUR_BAD
            else:
                fill = self.side_colour(i)
            c.create_oval(x - radius, y - radius, x + radius, y + radius,
                          fill=fill, outline="#000000")

        # the original position of every corrected point, linked to the new one
        if mode == self.MODE_BOTH:
            moved = self.seq.changed(cam)[t] if 0 <= t < self.seq.n_frames else []
            for i in np.flatnonzero(moved):
                if not (before[i, 2] > 0 or before[i, 0] != 0 or before[i, 1] != 0):
                    continue
                ox, oy = self.to_screen(view, before[i, 0], before[i, 1])
                if present[i]:
                    nx, ny = self.to_screen(view, current[i, 0], current[i, 1])
                    c.create_line(ox, oy, nx, ny, fill=self.COLOUR_ORIGINAL, dash=(2, 3))
                r = 6
                c.create_oval(ox - r, oy - r, ox + r, oy + r,
                              outline=self.COLOUR_ORIGINAL, width=2, dash=(3, 2))

        if not self.seq.has_file(cam, t):
            c.create_text(w / 2, h / 2, fill="#9ca3af", font=("Helvetica", 11),
                          text=f"No keypoint file for frame {self.frame_number(t)} "
                               f"in this camera")

        # camera name, and whether this frame has unsaved edits
        label = view["name"]
        if mode == self.MODE_ORIGINAL:
            label += "  ·  ORIGINAL (before correction)"
        elif self.frame_index in self.seq.dirty[view["index"]]:
            label += "  ·  edited"
        c.create_text(10, 12, text=label, anchor="w", fill="#facc15",
                      font=("Helvetica", 9, "bold"))
        c.create_text(w - 10, 12, text=f"{view['scale'] * 100:.0f}%", anchor="e",
                      fill="#8b929e", font=("Helvetica", 9))

    # ==================================================================
    # Navigation and editing
    # ==================================================================
    def select_keypoint(self, index):
        self.keypoint = int(index)
        self.redraw_all()

    def on_person_selected(self, _event=None):
        """Switch person; each person keeps its own corrections."""
        if self.seq is None:
            return
        person = int(self.person_var.get()) - 1
        if person == self.seq.person:
            return
        seq = self.person_sequence(person)
        if seq is None:
            self.person_var.set(str(self.seq.person + 1))
            return
        self.seq = seq
        self.update_swap_targets()
        self.measure_heights()
        self.show_frame(min(self.frame_index, self.seq.n_frames - 1))
        self.scan()
        self.refresh_corrections()

    def person_sequence(self, person):
        """The keypoints of one person, read once and kept with its edits."""
        if person not in self.sequences:
            try:
                seq = open_sequence(self.pose_dir, person=person)
                seq.follow(self.sequences.get(0))
            except Exception as exc:
                messagebox.showerror("Could not read that person", str(exc))
                return None
            self.sequences[person] = seq
        return self.sequences[person]

    def update_swap_targets(self):
        """Every person but the one shown, at least person 2."""
        if self.seq is None:
            return
        count = max(self.seq.max_people, 2)
        values = [str(i + 1) for i in range(count) if i != self.seq.person]
        self.swap_combo.configure(values=values)
        if self.swap_target.get() not in values:
            self.swap_target.set(values[0])

    def seek(self, frame):
        if self.seq is None:
            return
        self.show_frame(int(np.clip(frame, 0, self.seq.n_frames - 1)))

    def on_slider(self, value):
        if self.seq is None:
            return
        frame = int(float(value))
        if frame != self.frame_index:
            self.show_frame(frame)

    def frame_number(self, t):
        """The frame number written in the file names, for row t."""
        if self.seq is None or not 0 <= t < self.seq.n_frames:
            return t
        return self.seq.numbers[t]

    def video_index(self, cam, t):
        """The video frame behind row t of camera cam."""
        offset = self.seq.video_offsets[cam] if cam < len(self.seq.video_offsets) else 0
        return self.frame_number(t) + (offset or 0)

    def show_frame(self, frame):
        self.follow_views(self.frame_index, frame)
        self.frame_index = frame
        self.frame_label.configure(
            text=f"frame {self.frame_number(frame)}  ({frame + 1}/{self.seq.n_frames})")
        try:
            self.slider.set(frame)
        except Exception:
            pass
        self.redraw_all()

    # ------------------------------------------------------------------
    # Editing with a weight: the frames around move with the point
    # ------------------------------------------------------------------
    def weight_frames(self):
        try:
            return int(np.clip(int(self.weight.get()), 0, self.MAX_WEIGHT))
        except (tk.TclError, ValueError):
            return 0

    def spread_weights(self, cam, kp, frame):
        """The frames an edit moves and their share of the move (half a cosine
        down to 0).
        """
        w = self.weight_frames()
        frames, shares = [frame], [1.0]
        if w <= 0:
            return frames, shares
        changed = self.seq.changed(cam)[:, kp]
        for direction in (-1, 1):
            for k in range(1, w + 1):
                f = frame + direction * k
                if not self.seq.has_file(cam, f):
                    break
                if changed[f] and (cam, f, kp) not in self.seq.carried:
                    break               # placed by hand: it stays where it is
                frames.append(f)
                shares.append(0.5 * (1.0 + np.cos(np.pi * k / (w + 1))))
        return frames, shares

    def begin_edit(self, cam, kp, frame):
        """Start moving a point: remember the frames the weight reaches."""
        frames, shares = self.spread_weights(cam, kp, frame)
        before = {f: self.seq.data[cam][f, kp].copy() for f in frames}
        self.edit = {"cam": cam, "kp": kp, "frame": frame, "frames": frames,
                     "shares": shares, "before": before,
                     "undo": len(self.seq.undo_stack)}

    def apply_edit(self, x, y, conf=1.0):
        """Put the point at (x, y) and move the frames around it by their
        share.
        """
        e = self.edit
        if e is None:
            return
        cam, kp, here = e["cam"], e["kp"], e["frame"]
        base = e["before"][here]
        exists = base[2] > 0 or base[0] != 0 or base[1] != 0
        move = np.array([x, y], np.float64) - base[:2] if exists else None
        for f, share in zip(e["frames"], e["shares"]):
            old = e["before"][f]
            self.seq.data[cam][f, kp] = old
            if f == here:
                self.seq.data[cam][f, kp] = (x, y, conf)
            elif move is not None and (old[2] > 0 or old[0] != 0 or old[1] != 0):
                step = share * move
                if np.hypot(*step) >= 0.5:      # less than half a pixel: untouched
                    self.seq.data[cam][f, kp] = (old[0] + step[0], old[1] + step[1],
                                                 max(float(old[2]), self.SPREAD_CONFIDENCE))
            self.seq.dirty[cam].add(f)

    def finish_edit(self):
        """End the edit: one Undo entry, and the moved frames ringed in blue."""
        e, self.edit = self.edit, None
        if e is None:
            return 0
        cam, kp, here = e["cam"], e["kp"], e["frame"]
        moved = 0
        for f in e["frames"]:
            old = e["before"][f]
            if np.array_equal(self.seq.data[cam][f, kp], old):
                continue
            self.seq.undo_stack.append((cam, f, kp, old))
            if f == here:
                self.seq.carried.pop((cam, f, kp), None)
            else:
                self.seq.carried[(cam, f, kp)] = here
                moved += 1
        self.seq.group(e["undo"])
        name, cam_name = self.names[kp], self.seq.cam_names[cam]
        if moved:
            self.status_text.set(
                f"Moved {name} in {cam_name}, and {moved} frame(s) around it with "
                f"the weight (ringed in blue). One Undo reverts them all.")
        else:
            self.status_text.set(f"Moved {name} in {cam_name}: "
                                 f"{self.unsaved_total()} frame(s) unsaved")
        return moved

    def weight_window(self, cam, frame):
        """The frames a swap applies to: this one and those within the weight."""
        w = self.weight_frames()
        frames = [frame]
        for direction in (-1, 1):
            for k in range(1, w + 1):
                f = frame + direction * k
                if not self.seq.has_file(cam, f):
                    break
                frames.append(f)
        return sorted(frames)

    # ------------------------------------------------------------------
    # Trajectory plot: x and y of the selected point over the frames
    # ------------------------------------------------------------------
    def add_plot_point(self, _event=None):
        name = self.plot_add.get()
        self.plot_add.set("Add a point")
        if name in self.names:
            kp = self.names.index(name)
            if kp not in self.plot_extra:
                self.plot_extra.append(kp)
        self.draw_plot()

    def clear_plot_points(self):
        self.plot_extra = []
        self.draw_plot()

    def plot_range(self):
        n = self.seq.n_frames
        span = int(max(10, min(self.plot_span, n)))
        t0 = max(0, self.frame_index - span)
        t1 = min(n - 1, t0 + 2 * span)
        t0 = max(0, t1 - 2 * span)
        return t0, t1

    def draw_plot(self):
        c = getattr(self, "plot", None)
        if c is None:
            return
        c.delete("all")
        self.plot_map = None
        if self.seq is None:
            self.plot_title.configure(text="Trajectory")
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 60 or h < 40:
            return
        cam, kp = self.active_cam, self.keypoint
        if kp >= self.seq.n_keypoints:
            return
        self.plot_title.configure(
            text=f"Trajectory of {self.names[kp]} in {self.seq.cam_names[cam]}")
        self.plot_add.configure(values=list(self.names))
        t0, t1 = self.plot_range()
        left, right, top, bottom, gap = 54, 10, 8, 16, 10
        lane_h = (h - top - bottom - gap) / 2
        span = max(t1 - t0, 1)

        def fx(t):
            return left + (t - t0) * (w - left - right) / span

        block = self.seq.data[cam][t0:t1 + 1]
        original = self.seq.original[cam][t0:t1 + 1]
        shown = [kp] + [k for k in self.plot_extra if k != kp and k < self.seq.n_keypoints]

        def present(b):
            return (b[..., 2] > 0) | (b[..., 0] != 0) | (b[..., 1] != 0)

        changed = self.seq.changed(cam)[t0:t1 + 1, kp]
        lanes = []
        for axis in (0, 1):
            y0 = top + axis * (lane_h + gap)
            values = [block[present(block[:, k]), k, axis] for k in shown]
            if changed.any():
                values.append(original[present(original[:, kp]), kp, axis])
            values = np.concatenate(values) if values else np.zeros(0)
            if values.size:
                lo, hi = float(values.min()), float(values.max())
            else:
                lo, hi = 0.0, 1.0
            pad = max(10.0, 0.08 * (hi - lo))
            lo, hi = lo - pad, hi + pad
            lanes.append((y0, lo, hi))

        def fy(axis, v):
            y0, lo, hi = lanes[axis]
            share = (v - lo) / (hi - lo)
            # x grows upwards; y grows downwards, as in the image
            return y0 + lane_h * (1 - share if axis == 0 else share)

        # the frames the weight reaches, and the current frame
        wt = self.weight_frames()
        if wt:
            a, b = fx(max(t0, self.frame_index - wt)), fx(min(t1, self.frame_index + wt))
            c.create_rectangle(a, top, b, h - bottom, fill="#1f2a37", outline="")
        for axis, label in ((0, "x"), (1, "y")):
            y0, lo, hi = lanes[axis]
            c.create_rectangle(left, y0, w - right, y0 + lane_h, outline="#2b3138")
            c.create_text(6, y0 + lane_h / 2, text=label, anchor="w", fill="#9ca3af",
                          font=("Helvetica", 9, "bold"))
            top_value, bottom_value = (hi, lo) if axis == 0 else (lo, hi)
            c.create_text(left - 4, y0 + 2, text=f"{top_value:.0f}", anchor="ne",
                          fill="#8b929e", font=("Helvetica", 8))
            c.create_text(left - 4, y0 + lane_h - 2, text=f"{bottom_value:.0f}",
                          anchor="se", fill="#8b929e", font=("Helvetica", 8))

        def curve(b, k, axis, **style):
            ok = present(b[:, k])
            run = []
            for i, t in enumerate(range(t0, t1 + 1)):
                if ok[i]:
                    run += [fx(t), fy(axis, float(b[i, k, axis]))]
                elif run:
                    self._polyline(c, run, **style)
                    run = []
            if run:
                self._polyline(c, run, **style)

        # suspect frames of this point, marked under the lanes
        colours = {"low": self.COLOUR_BAD, "jum": self.COLOUR_JUMP,
                   "cam": self.COLOUR_CAMERAS, "mis": "#9ca3af"}
        for p in self.problems:
            if p[0] != cam or p[1] != kp or p[5] < t0 or p[2] > t1:
                continue
            colour = colours.get(p[3][:3], "#9ca3af")
            a, b = fx(max(p[2], t0)), fx(min(p[5], t1))
            c.create_rectangle(a - 1, h - bottom + 4, b + 1, h - bottom + 9,
                               fill=colour, outline="")

        for axis in (0, 1):
            for k in shown[1:]:
                curve(block, k, axis, fill=self.side_colour(k), width=1)
            if changed.any():
                curve(original, kp, axis, fill=self.COLOUR_ORIGINAL, width=1, dash=(3, 3))
            curve(block, kp, axis, fill=self.COLOUR_SELECTED, width=2)
            # frames moved by the weight
            for i, t in enumerate(range(t0, t1 + 1)):
                if (cam, t, kp) in self.seq.carried and present(block[i:i + 1, kp])[0]:
                    x_, y_ = fx(t), fy(axis, float(block[i, kp, axis]))
                    c.create_oval(x_ - 3, y_ - 3, x_ + 3, y_ + 3,
                                  outline=self.COLOUR_CARRIED, width=1)
        # the names of the other points, at the end of their curves
        for k in shown[1:]:
            ok = np.flatnonzero(present(block[:, k]))
            if len(ok):
                i = ok[-1]
                c.create_text(fx(t0 + i) - 2, fy(0, float(block[i, k, 0])) - 2,
                              text=self.names[k], anchor="se",
                              fill=self.side_colour(k), font=("Helvetica", 8))

        x_now = fx(self.frame_index)
        c.create_line(x_now, top, x_now, h - bottom, fill="#facc15", dash=(2, 2))
        here = self.seq.data[cam][self.frame_index, kp]
        if here[2] > 0 or here[0] != 0 or here[1] != 0:
            for axis in (0, 1):
                y_ = fy(axis, float(here[axis]))
                c.create_oval(x_now - 5, y_ - 5, x_now + 5, y_ + 5,
                              fill=self.COLOUR_SELECTED, outline="#000000")
        c.create_text(left, h - 2, text=f"{self.frame_number(t0)}", anchor="sw",
                      fill="#8b929e", font=("Helvetica", 8))
        c.create_text(w - right, h - 2, text=f"{self.frame_number(t1)}", anchor="se",
                      fill="#8b929e", font=("Helvetica", 8))
        self.plot_map = {"t0": t0, "t1": t1, "left": left, "right": right, "w": w,
                         "lanes": lanes, "lane_h": lane_h, "cam": cam, "kp": kp}

    @staticmethod
    def _polyline(c, coords, **style):
        if len(coords) >= 4:
            c.create_line(*coords, **style)
        else:
            x, y = coords
            c.create_oval(x - 1, y - 1, x + 1, y + 1, outline=style.get("fill", "#fff"))

    def plot_frame_at(self, sx):
        m = self.plot_map
        share = (sx - m["left"]) / max(m["w"] - m["left"] - m["right"], 1)
        return int(np.clip(round(m["t0"] + share * (m["t1"] - m["t0"])), 0,
                           self.seq.n_frames - 1))

    def plot_value(self, axis, sy):
        y0, lo, hi = self.plot_map["lanes"][axis]
        share = (sy - y0) / self.plot_map["lane_h"]
        if axis == 0:
            share = 1 - share
        return lo + share * (hi - lo)

    def on_plot_press(self, event):
        """Click: go to that frame; drag on the curve: move the point at that
        frame.
        """
        self.plot_drag = None
        if self.seq is None or self.plot_map is None:
            return
        frame = self.plot_frame_at(event.x)
        if frame != self.frame_index:
            self.show_frame(frame)
        m = self.plot_map
        axis = None
        for a, (y0, _lo, _hi) in enumerate(m["lanes"]):
            if y0 <= event.y <= y0 + m["lane_h"]:
                axis = a
        if axis is None:
            return
        cam, kp = m["cam"], m["kp"]
        point = self.seq.point(cam, frame, kp)
        if not (point[2] > 0 or point[0] != 0 or point[1] != 0):
            return
        lanes = self.plot_map["lanes"]
        y0, lo, hi = lanes[axis]
        share = (float(point[axis]) - lo) / (hi - lo)
        y_point = y0 + m["lane_h"] * (1 - share if axis == 0 else share)
        if abs(event.y - y_point) > 10 or not self.editable(cam):
            return
        self.begin_edit(cam, kp, frame)
        # the scale stays as it was when the drag started
        self.plot_drag = {"axis": axis, "map": dict(self.plot_map), "moved": False}

    def on_plot_drag(self, event):
        if self.seq is None:
            return
        if self.plot_drag is None:
            if self.plot_map is not None:           # scrub through the frames
                frame = self.plot_frame_at(event.x)
                if frame != self.frame_index:
                    self.show_frame(frame)
            return
        drag = self.plot_drag
        saved, self.plot_map = self.plot_map, drag["map"]
        value = self.plot_value(drag["axis"], event.y)
        self.plot_map = saved
        e = self.edit
        point = e["before"][e["frame"]]
        x, y = float(point[0]), float(point[1])
        if drag["axis"] == 0:
            x = value
        else:
            y = value
        self.apply_edit(x, y)
        drag["moved"] = True
        self.redraw(self.views[e["cam"]])
        self.draw_plot()
        self.update_readout()

    def on_plot_release(self, _event):
        drag, self.plot_drag = self.plot_drag, None
        if drag is None:
            return
        if drag["moved"]:
            self.finish_edit()
            self.redraw_all()
            self.refresh_corrections()
        else:
            self.edit = None

    def on_plot_wheel(self, event, direction=None):
        if direction is None:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1
        factor = 1 / 1.25 if direction > 0 else 1.25
        n = self.seq.n_frames if self.seq is not None else 1000
        self.plot_span = int(np.clip(self.plot_span * factor, 10, max(n, 10)))
        self.draw_plot()

    # ==================================================================
    # Checks: jumps and agreement between cameras
    # ==================================================================
    def measure_heights(self):
        """The person's height in each camera and frame."""
        self.heights = []
        if self.seq is None:
            return
        threshold = float(self.conf_threshold.get())
        for cam in range(self.seq.n_cameras):
            self.heights.append(pose_checks.body_height(self.seq.data[cam], threshold))

    def load_calibration(self, path):
        """Read and pair the calibration; switch the camera check off if the
        cameras disagree.
        """
        self.cameras, self.calib_file = None, None
        name, warning, info = "none", "", ""
        if path is not None and self.seq is not None and self.seq.n_cameras >= 3:
            try:
                cameras, note = pose_checks.pair_cameras(path, self.seq.cam_names)
            except Exception as exc:
                cameras, note = None, ""
                warning = f"Calibration unreadable: {exc}"
            if cameras is not None:
                self.cameras, self.calib_file = cameras, Path(path)
                name = Path(path).name
                self.match_people()
                warning, info = self.calibration_health(note)
            elif not warning:
                warning = note or f"{Path(path).name} does not match these cameras."
        elif self.seq is not None and self.seq.kind == "trc":
            info = "Sports2D result: one camera, no comparison between cameras."
        elif self.seq is not None and self.seq.n_cameras < 3:
            warning = (f"This folder holds {self.seq.n_cameras} camera(s): comparing "
                       f"with the other cameras needs 3 or more.")
        self.calib_details = "\n".join(t for t in (warning, info) if t)
        if hasattr(self, "calib_label"):
            self.calib_label.configure(text=name if len(name) <= 22 else name[:20] + "…")
            self.calib_health.configure(text=warning)
            if warning:
                self.calib_health.pack(anchor="w", padx=8, pady=(2, 0),
                                       before=self.scan_button)
            else:
                self.calib_health.pack_forget()

    def match_people(self):
        """Person 1 matched across cameras, before person association."""
        seq = self.sequences.get(0)
        if seq is None or self.cameras is None or not seq.needs_matching:
            return
        if seq.unsaved:
            self.status_text.set("Person 1 not matched again: save your "
                                 "corrections first, then choose the calibration.")
            return
        def progress(t, n):
            self.status_text.set(f"Matching person 1 across cameras with the "
                                 f"calibration: frame {t} of {n}...")
            self.root.update_idletasks()
        try:
            seq.match_people(self.cameras, float(self.conf_threshold.get()), progress)
        except Exception as exc:
            self.status_text.set(f"Could not match the person across cameras: {exc}")
            return
        for other in self.sequences.values():
            if other is not seq and not other.unsaved:
                other.follow(seq)
        if seq is self.seq:
            self.measure_heights()

    def calibration_health(self, note=""):
        """How well the cameras agree, in plain words: (warning, info)."""
        warn = [note] if note else []
        lines = []
        seq0 = self.sequences.get(0)
        if seq0 is not None and seq0.matched:
            others = [f"{self.seq.cam_names[c]} ({int((seq0.slots[c] > 0).sum())} of "
                      f"{seq0.n_frames} frames)" for c in range(seq0.n_cameras)
                      if (seq0.slots[c] > 0).any()]
            text = ("People are not matched across cameras in this folder, so "
                    "person 1 was matched automatically with the calibration.")
            if others:
                text += (f" Person 1 is not the first person of the file in "
                         f"{', '.join(others)}.")
            if self.seq.person != 0:
                text += (" Other persons are shown as numbered in each file and "
                         "are not compared with the other cameras.")
            lines.append(text)
        elif not self.seq.associated and any((n > 1).any() for n in self.seq.n_people):
            lines.append("People are not matched across cameras in this folder: "
                         "frames where a camera sees several people are left out "
                         "of the camera check. After person association "
                         "(pose-associated) every frame can be checked.")
        unknown = [n for n, o in zip(self.seq.cam_names, self.seq.video_offsets)
                   if o is None]
        if unknown:
            warn.append(f"Video frame not found for {', '.join(unknown)}: the image "
                         f"behind the points may be from another moment.")
        try:
            report = pose_checks.calibration_report(
                self.seq.data, self.cameras, self.seq.usable())
        except Exception as exc:
            warn.append(f"Could not measure the agreement: {exc}")
            return "\n".join(warn), "\n".join(lines)
        values = [r["pct"] for r in report if np.isfinite(r["pct"])]
        if not values:
            lines.append("Not enough confident points seen by 3 cameras to judge "
                         "the calibration.")
            return "\n".join(warn), "\n".join(lines)
        per_cam = ", ".join(
            f"{self.seq.cam_names[r['camera']]} {self.to_cm(r['pct']):.0f} cm"
            for r in report if np.isfinite(r["pct"]))
        typical = float(np.median(values))
        if typical > pose_checks.BAD_AGREEMENT:
            self.flag_cameras.set(False)
            warn.append(
                f"The cameras disagree even on confident points (typically about "
                f"{self.to_cm(typical):.0f} cm: {per_cam}). The calibration does "
                f"not match these keypoints: wrong file, cameras in another order, "
                f"or frames out of sync. Camera check switched off.")
        else:
            worst = max(report, key=lambda r: r["pct"] if np.isfinite(r["pct"]) else -1)
            lines.append(f"Cameras agree within about {self.to_cm(typical):.0f} cm on "
                         f"confident points ({per_cam}).")
            if worst["pct"] > max(pose_checks.BAD_AGREEMENT, 3 * typical):
                warn.append(f"{self.seq.cam_names[worst['camera']]} disagrees much "
                            f"more than the others: its calibration may be off.")
        return "\n".join(warn), "\n".join(lines)

    def choose_calibration(self):
        start = None
        if self.pose_dir is not None:
            found = pose_checks.find_calibration(self.pose_dir)
            start = str(found.parent) if found else str(Path(self.pose_dir).parent)
        path = filedialog.askopenfilename(
            title="Select the calibration file (Calib_*.toml)", initialdir=start,
            filetypes=[("Calibration", "*.toml"), ("All files", "*.*")])
        if path:
            self.load_calibration(path)
            self.redraw_all()

    # ------------------------------------------------------------------
    # Units: the checks work in % of the person's height, the panel in cm
    # ------------------------------------------------------------------
    def to_pct(self, cm):
        return 100.0 * float(cm) / self.reference_cm

    def to_cm(self, pct):
        return float(pct) * self.reference_cm / 100.0

    def jump_limit_pct(self):
        return self.to_pct(self.jump_limit.get())

    def is_foot(self, kp):
        return 0 <= kp < len(self.names) and any(w in self.names[kp] for w in self.FEET_WORDS)

    def camera_limits(self):
        """Disagreement limit of each keypoint, in % of the height (feet limit
        for ankles, heels, toes).
        """
        k = self.seq.n_keypoints if self.seq else len(self.names)
        out = np.full(k, self.to_pct(self.camera_limit.get()))
        for kp in range(k):
            if self.is_foot(kp):
                out[kp] = self.to_pct(self.feet_limit.get())
        return out

    def checks(self, cam, frame):
        """Flags and measures for every keypoint of one camera and frame."""
        k = self.seq.n_keypoints if self.seq else 0
        out = {"jump": np.full(k, np.nan), "jump_flag": np.zeros(k, bool),
               "cam_px": np.full(k, np.nan), "cam_pct": np.full(k, np.nan),
               "camera_flag": np.zeros(k, bool), "proposed": np.full((k, 2), np.nan)}
        if self.seq is None or not 0 <= frame < self.seq.n_frames:
            return out
        block = self.seq.data[cam]
        here, ok_here = block[frame], pose_checks.present(block[frame:frame + 1])[0]
        height = self.heights[cam][frame] if cam < len(self.heights) else np.nan

        if frame > 0 and np.isfinite(height) and height > 0:
            before = block[frame - 1]
            ok = ok_here & pose_checks.present(block[frame - 1:frame])[0]
            dist = np.linalg.norm(here[:, :2] - before[:, :2], axis=1)
            out["jump"][ok] = 100.0 * dist[ok] / height
            if self.flag_jumps.get():
                out["jump_flag"] = np.nan_to_num(out["jump"]) > self.jump_limit_pct()

        if self.cameras is not None:
            try:
                px, pct, proposed = pose_checks.disagreement(
                    self.seq.data, self.cameras, cam,
                    float(self.conf_threshold.get()), frames=[frame],
                    usable=self.seq.usable())
                out["cam_px"], out["cam_pct"], out["proposed"] = px[0], pct[0], proposed[0]
                if self.flag_cameras.get():
                    out["camera_flag"] = np.nan_to_num(pct[0]) > self.camera_limits()
            except Exception:
                pass
        return out

    def explain(self, view, i):
        """Plain words for the tooltip of one point."""
        cam, frame = view["index"], self.frame_index
        point = self.seq.point(cam, frame, i)
        checks = self.checks(cam, frame)
        threshold = float(self.conf_threshold.get())
        lines = [f"{self.names[i]}  ·  {view['name']}  ·  frame {self.frame_number(frame)}"]
        if self.seq.kind == "trc":
            lines.append("Placed by hand." if point[2] >= 0.999 else
                         "Sports2D position (TRC files hold no confidence).")
        elif point[2] >= 0.999:
            lines.append("Placed by hand (confidence 1.00).")
        elif point[2] < threshold:
            lines.append(f"Confidence {point[2]:.2f}: below your limit of {threshold:.2f}. "
                         "The detector is not sure of this point.")
        else:
            lines.append(f"Confidence {point[2]:.2f} (limit {threshold:.2f}).")
        jump = checks["jump"][i]
        if np.isfinite(jump):
            limit = self.jump_limit_pct()
            verdict = ("more than your limit: it may have jumped to the wrong place"
                       if jump > limit else "a normal move")
            lines.append(f"Moved about {self.to_cm(jump):.0f} cm since frame "
                         f"{self.frame_number(frame - 1)} (limit "
                         f"{self.to_cm(limit):.0f} cm): {verdict}.")
        if self.seq.kind == "trc":
            pass                        # one camera: nothing to compare with
        elif self.seq.n_cameras < 3:
            lines.append(f"Other cameras: this folder holds {self.seq.n_cameras} "
                         f"camera(s), so not compared.")
        elif self.cameras is None:
            lines.append("Other cameras: no calibration loaded, so not compared.")
        elif np.isfinite(checks["cam_pct"][i]):
            limit = float(self.camera_limits()[i])
            which = "feet limit" if self.is_foot(i) else "limit"
            pct, px = checks["cam_pct"][i], checks["cam_px"][i]
            verdict = ("the other cameras disagree; the cross shows where they see it"
                       if pct > limit else "the cameras agree")
            lines.append(f"Other cameras place it {px:.0f} px away, about "
                         f"{self.to_cm(pct):.1f} cm ({which} "
                         f"{self.to_cm(limit):.0f} cm): {verdict}.")
        elif not self.seq.usable()[cam][frame]:
            lines.append("Other cameras: not compared here, this camera sees "
                         "several people and they are not matched across cameras "
                         "in this folder.")
        else:
            lines.append("Other cameras: fewer than two see this point clearly, "
                         "so not compared.")
        source = self.seq.carried.get((cam, frame, i))
        if source is not None:
            lines.append(f"Carried: moved with your correction at frame "
                         f"{self.frame_number(source)} (weight).")
        if self.seq.changed(cam)[frame, i]:
            o = self.seq.original[cam][frame, i]
            if o[2] > 0 or o[0] != 0 or o[1] != 0:
                d = float(np.hypot(*(self.seq.data[cam][frame, i, :2] - o[:2])))
                was = "" if self.seq.kind == "trc" else f" (confidence was {o[2]:.2f})"
                lines.append(f"Corrected: {d:.0f} px from the original position{was}.")
            else:
                lines.append("Added: this point was not detected originally.")
        return "\n".join(lines)

    def on_hover(self, view, event):
        c = view["canvas"]
        c.delete("tip")
        if self.seq is None or view["drag"] or view["pan"]:
            return
        i, d = self.nearest(view, event.x, event.y)
        if i < 0 or d > self.HIT_RADIUS:
            return
        text = self.explain(view, i)
        x, y = event.x + 16, event.y + 16
        item = c.create_text(x + 8, y + 6, text=text, anchor="nw", fill="#e5e7eb",
                             font=("Helvetica", 9), width=340, tags="tip")
        x0, y0, x1, y1 = c.bbox(item)
        w, h = c.winfo_width(), c.winfo_height()
        dx = min(0, w - 4 - (x1 + 8))       # keep the box inside the canvas
        dy = min(0, h - 4 - (y1 + 6))
        if dx or dy:
            c.move(item, dx, dy)
            x0, y0, x1, y1 = c.bbox(item)
        box = c.create_rectangle(x0 - 8, y0 - 6, x1 + 8, y1 + 6, fill="#1f2937",
                                 outline="#4b5563", tags="tip")
        c.tag_raise(item, box)

    def place_proposed(self):
        """Move the selected point to where the other cameras see it."""
        if self.seq is None:
            return
        if self.seq.kind == "trc":
            messagebox.showinfo(
                "One camera",
                "A Sports2D result comes from one camera: there are no other "
                "cameras to compare with. Drag the point, or use the trajectory "
                "plot under the images.")
            return
        if self.seq.n_cameras < 3:
            messagebox.showinfo(
                "Not enough cameras",
                f"This folder holds {self.seq.n_cameras} camera(s). Placing a "
                f"point where the other cameras see it needs 3 cameras or more: "
                f"open the folder that holds one folder per camera (pose, "
                f"pose-sync or pose-associated).")
            return
        if self.cameras is None:
            messagebox.showinfo(
                "No calibration",
                "This needs the calibration file, to know where the other "
                "cameras see the point. Use Browse… next to Calibration.")
            return
        cam, frame, kp = self.active_cam, self.frame_index, self.keypoint
        if not self.editable(cam):
            return
        proposed = self.checks(cam, frame)["proposed"][kp]
        if not np.all(np.isfinite(proposed)):
            messagebox.showinfo(
                "Not enough cameras",
                f"Fewer than two other cameras see {self.names[kp]} clearly in "
                f"frame {self.frame_number(frame)}, so there is no position to "
                f"propose.")
            return
        self.begin_edit(cam, kp, frame)
        self.apply_edit(float(proposed[0]), float(proposed[1]))
        moved = self.finish_edit()
        self.redraw_all()
        self.refresh_corrections()
        if not moved:
            self.status_text.set(
                f"Placed {self.names[kp]} in {self.seq.cam_names[cam]} where the "
                f"other cameras see it")

    def swap_lr(self):
        if self.seq is None or not self.editable(self.active_cam):
            return
        partner = self.partners.get(self.keypoint)
        if partner is None:
            messagebox.showinfo(
                "No counterpart",
                f"{self.names[self.keypoint]} has no left/right partner.")
            return
        cam = self.active_cam
        start = len(self.seq.undo_stack)
        frames = [t for t in self.weight_window(cam, self.frame_index)
                  if self.seq.swap_lr(cam, t, self.keypoint, partner)]
        self.seq.group(start)
        self.redraw_all()
        self.refresh_corrections()
        where = (f"frame {self.frame_number(frames[0])}" if len(frames) == 1 else
                 f"frames {self.frame_number(frames[0])} to {self.frame_number(frames[-1])}")
        self.status_text.set(
            f"Swapped {self.names[self.keypoint]} and {self.names[partner]} in "
            f"{self.seq.cam_names[cam]}, {where}")

    def swap_person(self):
        """Exchange the selected point with another person, in this frame and
        those within the weight.
        """
        if self.seq is None or not self.editable(self.active_cam):
            return
        try:
            target = int(self.swap_target.get()) - 1
        except ValueError:
            return
        if target < 0 or target == self.seq.person:
            self.status_text.set("Choose another person to swap with.")
            return
        other = self.person_sequence(target)
        if other is None:
            return
        cam, kp = self.active_cam, self.keypoint
        done, same = [], 0
        start = len(self.seq.undo_stack)
        for t in self.weight_window(cam, self.frame_index):
            result = self.seq.swap_person(other, cam, t, kp)
            if result == "done":
                done.append(t)
            elif result == "same":
                same += 1
        self.seq.group(start)
        self.redraw_all()
        self.refresh_corrections()
        who = f"person {self.seq.person + 1} and person {target + 1}"
        if not done:
            reason = (f"person {target + 1} is person {self.seq.person + 1} in this "
                      f"file" if same else "neither person has this point here")
            self.status_text.set(f"Nothing swapped in {self.seq.cam_names[cam]}: {reason}.")
            return
        frames = (f"frame {self.frame_number(done[0])}" if len(done) == 1 else
                  f"frames {self.frame_number(done[0])} to {self.frame_number(done[-1])}")
        self.status_text.set(
            f"Swapped {self.names[kp]} between {who} in {self.seq.cam_names[cam]}, "
            f"{frames}. Save all corrections writes both people. One Undo puts "
            f"them back.")

    def clear_point(self):
        if self.seq is None or not self.editable(self.active_cam):
            return
        self.seq.clear_point(self.active_cam, self.frame_index, self.keypoint)
        self.redraw_all()
        self.refresh_corrections()
        self.status_text.set(f"Cleared {self.names[self.keypoint]}")

    def undo(self):
        if self.seq and self.seq.undo():
            self.redraw_all()
            self.refresh_corrections()
            self.status_text.set("Undone")

    # ==================================================================
    # Scanning
    # ==================================================================
    def scan(self):
        """List the points worth a look: low confidence, jumps, camera
        disagreement, gaps.
        """
        self.problem_list.delete(0, "end")
        self.problems = []
        if self.seq is None:
            return

        threshold = float(self.conf_threshold.get())
        jump_limit = self.jump_limit_pct()
        limits = self.camera_limits()
        self.measure_heights()
        for cam in range(self.seq.n_cameras):
            block = self.seq.data[cam]
            jump_pct = pose_checks.jumps(block, threshold)
            cam_pct = None
            if self.cameras is not None and self.flag_cameras.get():
                try:
                    cam_pct = pose_checks.disagreement(
                        self.seq.data, self.cameras, cam, threshold,
                        usable=self.seq.usable())[1]
                except Exception:
                    cam_pct = None
            conf = block[:, :, 2]
            xy = block[:, :, :2]
            present = (conf > 0) & ~((xy[:, :, 0] == 0) & (xy[:, :, 1] == 0))

            for kp in range(self.seq.n_keypoints):
                column = present[:, kp]
                if column.sum() < 3:
                    continue

                low = (np.flatnonzero(column & (conf[:, kp] < threshold))
                       if self.flag_low_conf.get() and self.seq.kind != "trc" else [])
                for first, last in contiguous(low):
                    self.problems.append(
                        (cam, kp, int(first),
                         f"low confidence {conf[first:last + 1, kp].min():.2f}",
                         float(threshold / max(conf[first:last + 1, kp].min(), 0.05)),
                         int(last)))

                # jumps, in % of the person's height, against the limit set
                if self.flag_jumps.get():
                    step = jump_pct[:, kp]
                    hits = np.flatnonzero(np.nan_to_num(step) > jump_limit)
                    for first, last in contiguous(hits):
                        worst = float(np.nanmax(step[first:last + 1]))
                        self.problems.append(
                            (cam, kp, int(first), f"jumps {self.to_cm(worst):.0f} cm",
                             worst / max(jump_limit, 1e-6), int(last)))

                # the other cameras see it elsewhere
                if cam_pct is not None:
                    off = cam_pct[:, kp]
                    cam_limit = float(limits[kp])
                    hits = np.flatnonzero(np.nan_to_num(off) > cam_limit)
                    for first, last in contiguous(hits):
                        worst = float(np.nanmax(off[first:last + 1]))
                        self.problems.append(
                            (cam, kp, int(first),
                             f"cameras disagree {self.to_cm(worst):.0f} cm",
                             worst / max(cam_limit, 1e-6), int(last)))

                gaps = ~column
                inside = np.flatnonzero(column)
                if len(inside):
                    gaps[: inside[0]] = False
                    gaps[inside[-1] + 1:] = False
                for first, last in contiguous(np.flatnonzero(gaps)):
                    self.problems.append(
                        (cam, kp, int(first),
                         f"missing {last - first + 1} frame(s)",
                         1.0 + (last - first) / 10, int(last)))

        self.fill_problem_list()
        self.status_text.set(f"{len(self.shown)} suspect point(s) to check")

    def fill_problem_list(self):
        """Show the suspect points in the order chosen under Sort by."""
        self.problem_list.delete(0, "end")
        if self.seq is None:
            return
        key = self.SORT_ORDERS.get(self.sort_by.get(), self.SORT_ORDERS["Frame"])
        self.problems.sort(key=key)
        # a point you corrected leaves this list: it is under Corrections now.
        # A run of frames stays listed from its first frame not yet corrected.
        changed = self.seq.changed()
        self.shown, hidden = [], 0
        for index, p in enumerate(self.problems):
            cam, kp, first, last = p[0], p[1], p[2], p[5]
            left = [t for t in range(first, last + 1)
                    if t < self.seq.n_frames and not changed[cam][t, kp]]
            if left:
                self.shown.append((index, left[0]))
            else:
                hidden += 1
        for index, frame in self.shown[:self.MAX_LISTED]:
            cam, kp, _first, detail = self.problems[index][:4]
            self.problem_list.insert(
                "end", f" f{self.frame_number(frame):04d}   {self.seq.cam_names[cam]}   "
                       f"{self.names[kp]}   {detail}")
        if len(self.shown) > self.MAX_LISTED:
            self.problem_list.insert(
                "end", f"  ... {len(self.shown) - self.MAX_LISTED} more, raise the "
                       f"limits to see fewer")
        if not self.shown:
            self.problem_list.insert(
                "end", "  nothing left to check" if hidden else "  nothing suspect found")
        text = f"Suspect points  ({len(self.shown)} to check"
        if hidden:
            text += f", {hidden} corrected"
        self.suspect_box.configure(text=text + ")")

    def show_in_problem_list(self, cam, kp, frame):
        """Highlight the suspect row of this point, if it has one."""
        self.problem_list.selection_clear(0, "end")
        for row, (index, _frame) in enumerate(self.shown[:self.MAX_LISTED]):
            p = self.problems[index]
            if p[0] == cam and p[1] == kp and p[2] <= frame <= p[5]:
                self.problem_list.selection_set(row)
                self.problem_list.see(row)
                return True
        return False

    def on_problem_selected(self, _event=None):
        selection = self.problem_list.curselection()
        if not selection or not getattr(self, "problems", None):
            return
        row = selection[0]
        if row >= len(self.shown) or row >= self.MAX_LISTED:
            return
        index, frame = self.shown[row]
        cam, kp = self.problems[index][:2]
        self.active_cam = cam
        self.select_keypoint(kp)
        self.show_frame(frame)

        # zoom the offending camera onto the point
        view = self.views[cam]
        point = self.seq.point(cam, frame, kp)
        if point[2] > 0:
            view["scale"] = 3.0
            view["ox"] = view["canvas"].winfo_width() / 2 - point[0] * 3.0
            view["oy"] = view["canvas"].winfo_height() / 2 - point[1] * 3.0
            self.redraw(view)

    # ==================================================================
    # Saving
    # ==================================================================
    def unsaved_total(self):
        return sum(q.unsaved for q in self.sequences.values())

    def save(self):
        """Save every correction; asks once, the first time, when the backup
        is made.
        """
        total = self.unsaved_total()
        if self.seq is None or not total:
            self.status_text.set("Nothing to save.")
            return
        if not self.seq.has_backup and self.seq.kind == "trc":
            if not messagebox.askyesno(
                    "Save all corrections",
                    f"{total} frame(s) changed.\n\n"
                    f"The TRC files of '{self.seq.root.name}' will first be copied "
                    f"to '{self.seq.backup_dir.name}', untouched. That copy is kept "
                    f"and never overwritten.\n\nYour corrections then go into the "
                    f"TRC file of each person.\n\nSave?"):
                return
        elif not self.seq.has_backup:
            if not messagebox.askyesno(
                    "Save all corrections",
                    f"{total} frame(s) changed.\n\n"
                    f"The whole '{self.seq.root.name}' folder will first be copied "
                    f"to '{self.seq.backup_dir.name}', untouched. That copy is kept "
                    f"and never overwritten: your original keypoints stay "
                    f"available.\n\nYour corrections then go into "
                    f"'{self.seq.root.name}', the folder Pose2Sim reads when it "
                    f"triangulates.\n\nSave?"):
                return
        written, created = 0, False
        try:
            for person in sorted(self.sequences):
                report = self.sequences[person].save()
                written += report["written"]
                created = created or report["created_backup"]
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.redraw_all()
        self.refresh_corrections()
        note = f" Originals kept in {self.seq.backup_dir.name}." if created else ""
        if self.seq.kind == "trc":
            self.status_text.set(
                f"Saved: {written} frame(s) into the TRC files of "
                f"{self.seq.root.name}.{note} For Sports2D to use a corrected "
                f"file, set load_trc_px to it in Advanced Settings, then launch "
                f"Sports2D again.")
            return
        self.status_text.set(
            f"Saved: {written} file write(s) into {self.seq.root.name}.{note}")

    def restore_all(self):
        """Back to the original keypoints, for every person."""
        if self.seq is None:
            return
        if not self.seq.has_backup:
            if not self.unsaved_total():
                self.status_text.set("Nothing to restore: no correction yet.")
                return
            if not messagebox.askyesno(
                    "Restore everything",
                    "Drop every correction made in this window, for every "
                    "person?\n\nNothing has been saved yet, so the files are "
                    "still the original ones."):
                return
            for q in self.sequences.values():
                q.undo_stack.clear()
                q.carried.clear()
                q.reload_rows()
            self.redraw_all()
            self.scan()
            self.refresh_corrections()
            self.status_text.set("All corrections dropped: back to the original keypoints")
            return
        if not messagebox.askyesno(
                "Restore everything",
                f"Copy every file from '{self.seq.backup_dir.name}' back over "
                f"'{self.seq.root.name}', discarding all corrections of every "
                f"person?\n\nThe backup itself is kept."):
            return
        n = self.seq.restore_all()
        for q in self.sequences.values():
            if q is not self.seq:
                q.undo_stack.clear()
                q.carried.clear()
                q.reload_rows()
        self.redraw_all()
        self.scan()
        self.refresh_corrections()
        self.status_text.set(f"Restored {n} file(s) from the backup")

    # ==================================================================
    # Comparing with the original
    # ==================================================================
    def folder_text(self):
        """The folder, and what kind of keypoints it holds."""
        kind = {"pose": "pose estimation, not synchronised",
                "pose-sync": "synchronised; people not matched across cameras",
                "pose-associated": "synchronised, people matched across cameras"}
        name = self.seq.root.name
        what = kind.get(name, "")
        if self.seq.kind == "trc":
            what = "Sports2D, pixel coordinates"
        return f"{self.seq.root}  ({what})" if what else str(self.seq.root)

    def toggle_original(self):
        self.show_original.set(not self.show_original.get())
        self.on_show_original()

    def on_show_original(self):
        """Show or hide the original position of each corrected point."""
        self.view_mode.set(self.MODE_BOTH if self.show_original.get()
                           else self.MODE_CORRECTED)
        if self.seq is not None:
            self.redraw_all()

    def refresh_corrections(self):
        """The list of every corrected point, for this person."""
        self.correction_list.delete(0, "end")
        self.corrections = []
        if self.seq is None:
            self.history_box.configure(text="History")
            return
        changed = self.seq.changed()
        for cam, block in enumerate(changed):
            for t, kp in zip(*np.nonzero(block)):
                a = self.seq.data[cam][t, kp]
                o = self.seq.original[cam][t, kp]
                if not (o[2] > 0 or o[0] != 0 or o[1] != 0):
                    what = "added"
                elif not (a[2] > 0 or a[0] != 0 or a[1] != 0):
                    what = "cleared"
                else:
                    what = f"moved {float(np.hypot(*(a[:2] - o[:2]))):.0f} px"
                self.corrections.append((int(t), cam, int(kp), what))
        self.corrections.sort()
        unsaved = self.unsaved_total()
        text = "History"
        if self.corrections:
            text += f"  ({len(self.corrections)} corrected, " + (
                f"{unsaved} frame(s) not saved)" if unsaved else "all saved)")
        elif unsaved:
            text += f"  ({unsaved} frame(s) not saved)"
        self.history_box.configure(text=text)
        for t, cam, kp, what in self.corrections[:500]:
            self.correction_list.insert(
                "end", f" f{self.frame_number(t):04d}   {self.seq.cam_names[cam]}   "
                       f"{self.names[kp]}   {what}")
        self.fill_problem_list()

    def delete_correction(self):
        """Put the selected corrected point back where the detector had it."""
        selection = self.correction_list.curselection()
        if self.seq is None or not selection or not getattr(self, "corrections", None):
            self.status_text.set("Select a correction in the History list first.")
            return
        t, cam, kp, _what = self.corrections[selection[0]]
        if self.view_mode.get() == self.MODE_ORIGINAL:
            self.status_text.set("You are looking at the original points. Press O "
                                 "to edit.")
            return
        o = self.seq.original[cam][t, kp]
        if not self.seq.set_point(cam, t, kp, float(o[0]), float(o[1]), float(o[2])):
            return
        self.seq.carried.pop((cam, t, kp), None)
        self.redraw_all()
        self.refresh_corrections()
        self.status_text.set(
            f"Deleted the correction of {self.names[kp]} in {self.seq.cam_names[cam]}, "
            f"frame {self.frame_number(t)}: back to the detection. Undo brings it back.")

    def on_correction_selected(self, _event=None):
        selection = self.correction_list.curselection()
        if not selection or not getattr(self, "corrections", None):
            return
        t, cam, kp, _what = self.corrections[selection[0]]
        self.active_cam = cam
        self.select_keypoint(kp)
        self.show_frame(t)

    def update_readout(self):
        if self.seq is None:
            return
        parts = [f"{self.names[self.keypoint]}  f{self.frame_number(self.frame_index)}"]
        for view in self.views:
            point = self.seq.point(view["index"], self.frame_index, self.keypoint)
            mark = "*" if view["index"] == self.active_cam else " "
            if point[2] <= 0:
                parts.append(f"{mark}{view['name']} --")
            else:
                parts.append(f"{mark}{view['name']} {point[2]:.2f}")
        self.readout.configure(text="   ".join(parts))

    def on_close(self):
        total = self.unsaved_total()
        if self.seq and total:
            answer = messagebox.askyesnocancel(
                "Unsaved corrections",
                f"{total} frame(s) have unsaved changes.\n\nSave them all "
                f"before closing?")
            if answer is None:
                return
            if answer:
                self.save()
                if self.unsaved_total():
                    return                      # the save was cancelled
        for view in self.views:
            if view["video"] is not None:
                try:
                    view["video"].release()
                except Exception:
                    pass
        self.root.destroy()


def contiguous(indices):
    """Runs of consecutive indices as (first, last) pairs."""
    indices = np.asarray(indices)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.concatenate([[indices[0]], indices[breaks + 1]])
    ends = np.concatenate([indices[breaks], [indices[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def pose_correction_func(*args):
    '''
    Manually correct the 2D keypoints produced by pose estimation, before
    they are triangulated.

    Usage:
        from Pose2Sim.Utilities import pose_correction; pose_correction.pose_correction_func(r'<pose_folder>')
        OR pose_correction
        OR pose_correction -i <pose_folder>
    '''

    if not args:
        pose_folder = None                       # invoked with no argument
    elif isinstance(args[0], dict):
        pose_folder = args[0].get('pose_folder')  # invoked with argparse
    else:
        pose_folder = args[0]                    # invoked as a function

    root = tk.Tk()
    app = PoseCorrectionApp(root, pose_folder)
    root.geometry("1500x900")
    root.minsize(1100, 700)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
