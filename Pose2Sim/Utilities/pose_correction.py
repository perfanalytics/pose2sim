#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
    ###########################################################
    ## POSE CORRECTION                                       ##
    ###########################################################

    Manually correct the 2D keypoints produced by pose estimation, before
    they are triangulated.

    Everything downstream reads these points. A wrist that the detector put
    on a chair leg becomes a 3D marker on a chair leg, and no amount of
    filtering will recover it. This is the last cheap moment to fix one.

    Opens a tkinter window showing every camera of a trial side by side, with
    the video frame behind the skeleton when the videos are available. Click a
    keypoint to select it, drag it to move it, and scrub through frames. Low
    confidence points are ringed in red so they are easy to find.

    Reads the OpenPose-format json that poseEstimation writes, accepting both
    naming conventions in the wild:
        cam01_000000.json
        cam01_000000_keypoints.json

    Triangulation reads `pose-associated`, then `pose-sync`, then `pose` - in
    that order. Corrections therefore go back into whichever of those exists,
    under the original filenames, or nothing downstream would see them. The
    first save copies that entire folder to `<name>-backup` once, and never
    overwrites it, so the originals are always recoverable.

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
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: Pillow not available. Video frames will not be shown. "
          "Install with: pip install Pillow")


## AUTHORSHIP INFORMATION
__author__ = "AYLARDJ"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["AYLARDJ"]
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
# Skeletons — only what is needed to draw and to know left from right
# ===========================================================================
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


def skeleton_for(n_keypoints):
    """Pick a layout from the keypoint count, which is what the file tells us."""
    if n_keypoints >= 26:
        return HALPE_26_NAMES, HALPE_26_LINKS, HALPE_26_LEFT, HALPE_26_RIGHT
    if n_keypoints >= 25:
        return BODY_25_NAMES, BODY_25_LINKS, BODY_25_LEFT, BODY_25_RIGHT
    if n_keypoints >= 17:
        return COCO_17_NAMES, COCO_17_LINKS, (1, 3, 5, 7, 9, 11, 13, 15), \
            (2, 4, 6, 8, 10, 12, 14, 16)
    names = tuple(f"kp{i}" for i in range(n_keypoints))
    return names, (), (), ()


# ===========================================================================
# Reading and writing the keypoint files
# ===========================================================================
FRAME_RE = re.compile(r"_(\d{4,})(?:_keypoints)?\.json$", re.IGNORECASE)
POSE_ORDER = ("pose-associated", "pose-sync", "pose")


def frame_files(directory):
    """Numbered keypoint files in a camera folder, in frame order.

    Both naming conventions are accepted, and anything that is not a numbered
    frame — metadata sidecars, .DS_Store — is ignored.
    """
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
    return list(best.values())


class Sequence:
    """Every camera of one recording, held in memory so scrubbing is instant."""

    def __init__(self, pose_dir, person=0):
        self.root = Path(pose_dir)
        self.person = person
        self.cam_dirs = []
        self.cam_names = []
        self.files = []
        self.data = []           # one (frames, keypoints, 3) array per camera
        self.meta = []
        self.dirty = []          # frame indices needing a write, per camera
        self.undo_stack = []
        self._load()

    # ------------------------------------------------------------------
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

        width = 0
        for d in subdirs:
            files = frame_files(d)
            rows, meta = [], []
            for f in files:
                with open(f, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                people = payload.get("people") or []
                idx = self.person if self.person < len(people) else (0 if people else -1)
                if idx < 0:
                    rows.append(np.zeros((0, 3), np.float32))
                    meta.append({"person_index": 0, "n_people": 0})
                    continue
                flat = people[idx].get("pose_keypoints_2d") or []
                arr = np.asarray(flat, np.float32).reshape(-1, 3)
                width = max(width, arr.shape[0])
                rows.append(arr)
                meta.append({"person_index": idx, "n_people": len(people)})

            self.cam_dirs.append(d)
            self.cam_names.append(re.sub(r"[_-]?json$", "", d.name, flags=re.I) or d.name)
            self.files.append(files)
            self.data.append(rows)
            self.meta.append(meta)
            self.dirty.append(set())

        # square every camera off to the widest keypoint count found
        for c, rows in enumerate(self.data):
            block = np.zeros((len(rows), width, 3), np.float32)
            for i, r in enumerate(rows):
                if r.size:
                    block[i, : r.shape[0]] = r
            self.data[c] = block

        self.n_keypoints = width
        self.n_frames = max(len(f) for f in self.files)
        self.max_people = max((m.get("n_people", 1) for cam in self.meta
                               for m in cam), default=1)

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
        return d.is_dir() and any(has_keypoints(s) for s in d.iterdir() if s.is_dir())

    @property
    def unsaved(self):
        return sum(len(d) for d in self.dirty)

    def point(self, cam, frame, kp):
        if frame >= len(self.data[cam]):
            return np.zeros(3, np.float32)
        return self.data[cam][frame, kp]

    def frame(self, cam, frame):
        if frame >= len(self.data[cam]):
            return np.zeros((self.n_keypoints, 3), np.float32)
        return self.data[cam][frame]

    # ------------------------------------------------------------------
    def set_point(self, cam, frame, kp, x, y, conf=1.0, record=True):
        if frame >= len(self.data[cam]):
            return
        if record:
            self.undo_stack.append((cam, frame, kp, self.data[cam][frame, kp].copy()))
        self.data[cam][frame, kp] = (x, y, conf)
        self.dirty[cam].add(frame)

    def clear_point(self, cam, frame, kp):
        self.set_point(cam, frame, kp, 0.0, 0.0, 0.0)

    def swap_lr(self, cam, frame, a, b):
        if frame >= len(self.data[cam]):
            return
        self.undo_stack.append((cam, frame, a, self.data[cam][frame, a].copy()))
        self.undo_stack.append((cam, frame, b, self.data[cam][frame, b].copy()))
        block = self.data[cam][frame]
        block[[a, b]] = block[[b, a]]
        self.dirty[cam].add(frame)

    def undo(self):
        if not self.undo_stack:
            return False
        cam, frame, kp, old = self.undo_stack.pop()
        self.data[cam][frame, kp] = old
        self.dirty[cam].add(frame)
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
        return sum(len(frame_files(s)) for s in self.backup_dir.iterdir()
                   if s.is_dir())

    def save(self):
        """Write changed frames back, backing the originals up first."""
        if not self.unsaved:
            return {"written": 0, "created_backup": False}
        _, created = self.create_backup()
        written = 0
        for cam in range(self.n_cameras):
            for fi in sorted(self.dirty[cam]):
                target = self.files[cam][fi]
                with open(target, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                people = payload.get("people") or [{}]
                idx = min(self.meta[cam][fi].get("person_index", 0), len(people) - 1)
                people[idx]["pose_keypoints_2d"] = [
                    round(float(v), 3) for v in self.data[cam][fi].reshape(-1)]
                payload["people"] = people
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                written += 1
            self.dirty[cam].clear()
        return {"written": written, "created_backup": created}

    def restore_frame(self, frame):
        """Put one frame back from the backup, in every camera."""
        if not self.has_backup:
            return 0
        restored = 0
        for cam in range(self.n_cameras):
            src = self.backup_dir / self.cam_dirs[cam].name / self.files[cam][frame].name
            if not src.exists():
                continue
            shutil.copy2(src, self.files[cam][frame])
            with open(src, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            people = payload.get("people") or []
            if people:
                idx = min(self.meta[cam][frame].get("person_index", 0), len(people) - 1)
                flat = np.asarray(people[idx].get("pose_keypoints_2d") or [],
                                  np.float32).reshape(-1, 3)
                k = min(flat.shape[0], self.data[cam].shape[1])
                self.data[cam][frame, :k] = flat[:k]
            self.dirty[cam].discard(frame)
            restored += 1
        return restored

    def restore_all(self):
        if not self.has_backup:
            return 0
        restored = 0
        for cam in range(self.n_cameras):
            src_dir = self.backup_dir / self.cam_dirs[cam].name
            if not src_dir.is_dir():
                continue
            for f in self.files[cam]:
                src = src_dir / f.name
                if src.exists():
                    shutil.copy2(src, f)
                    restored += 1
            self.dirty[cam].clear()
        self.undo_stack.clear()
        self._reload_arrays()
        return restored

    def _reload_arrays(self):
        for cam in range(self.n_cameras):
            for fi, f in enumerate(self.files[cam]):
                with open(f, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                people = payload.get("people") or []
                if not people:
                    continue
                idx = min(self.meta[cam][fi].get("person_index", 0), len(people) - 1)
                flat = np.asarray(people[idx].get("pose_keypoints_2d") or [],
                                  np.float32).reshape(-1, 3)
                k = min(flat.shape[0], self.data[cam].shape[1])
                self.data[cam][fi, :k] = flat[:k]


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
    COLOUR_BG = "#14171a"

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

        self.views = []          # one dict per camera: canvas, transform, video
        self.status_text = tk.StringVar(value="Open a keypoint folder to begin.")
        self.conf_threshold = tk.DoubleVar(value=self.DEFAULT_CONF_THRESHOLD)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_all_points = tk.BooleanVar(value=True)
        self.flag_low_conf = tk.BooleanVar(value=True)
        self.carry_frames = tk.IntVar(value=0)
        self.person_var = tk.StringVar(value="1")

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
        panel = ttk.Frame(self.root, width=self.CONTROL_PANEL_WIDTH)
        panel.grid(row=0, column=0, sticky="ns", padx=8, pady=8)
        panel.grid_propagate(False)

        ttk.Label(panel, text="Keypoint Correction",
                  font=("Helvetica", 15, "bold")).pack(anchor="w", pady=(0, 2))
        ttk.Label(panel, text="Fix the 2D points before triangulation.",
                  foreground="gray", wraplength=280).pack(anchor="w", pady=(0, 10))

        # ---- folder
        box = ttk.LabelFrame(panel, text="Keypoint folder")
        box.pack(fill="x", pady=(0, 10))
        ttk.Button(box, text="Open folder…",
                   command=self.choose_folder).pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(box, text="Find in a project…",
                   command=self.choose_project).pack(fill="x", padx=8, pady=(0, 8))
        self.folder_label = ttk.Label(box, text="none", foreground="gray",
                                      wraplength=270)
        self.folder_label.pack(anchor="w", padx=8, pady=(0, 8))

        # ---- what to edit
        box = ttk.LabelFrame(panel, text="Editing")
        box.pack(fill="x", pady=(0, 10))
        ttk.Label(box, text="Keypoint").pack(anchor="w", padx=8, pady=(8, 2))
        self.kp_combo = ttk.Combobox(box, state="readonly", values=[])
        self.kp_combo.pack(fill="x", padx=8)
        self.kp_combo.bind("<<ComboboxSelected>>", self.on_keypoint_selected)

        ttk.Label(box, text="Person").pack(anchor="w", padx=8, pady=(8, 2))
        self.person_combo = ttk.Combobox(box, state="readonly",
                                         textvariable=self.person_var, values=["1"])
        self.person_combo.pack(fill="x", padx=8)
        self.person_combo.bind("<<ComboboxSelected>>", self.on_person_selected)

        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Carry forward").pack(side="left")
        ttk.Spinbox(row, from_=0, to=200, width=5,
                    textvariable=self.carry_frames).pack(side="right")

        ttk.Button(box, text="Swap left / right",
                   command=self.swap_lr).pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(box, text="Clear this point",
                   command=self.clear_point).pack(fill="x", padx=8, pady=(0, 8))

        # ---- display
        box = ttk.LabelFrame(panel, text="Display")
        box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(box, text="Skeleton", variable=self.show_skeleton,
                        command=self.redraw_all).pack(anchor="w", padx=8, pady=(8, 0))
        ttk.Checkbutton(box, text="All points", variable=self.show_all_points,
                        command=self.redraw_all).pack(anchor="w", padx=8)
        ttk.Checkbutton(box, text="Ring low confidence",
                        variable=self.flag_low_conf,
                        command=self.redraw_all).pack(anchor="w", padx=8)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Below").pack(side="left")
        ttk.Spinbox(row, from_=0.0, to=1.0, increment=0.05, width=5,
                    textvariable=self.conf_threshold,
                    command=self.redraw_all).pack(side="right")

        # ---- problems
        box = ttk.LabelFrame(panel, text="Suspect points")
        box.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Button(box, text="Scan", command=self.scan).pack(fill="x", padx=8, pady=8)
        wrap = ttk.Frame(box)
        wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.problem_list = tk.Listbox(wrap, height=10, activestyle="none")
        bar = ttk.Scrollbar(wrap, command=self.problem_list.yview)
        self.problem_list.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        self.problem_list.pack(fill="both", expand=True)
        self.problem_list.bind("<<ListboxSelect>>", self.on_problem_selected)

        # ---- saving
        box = ttk.LabelFrame(panel, text="Saving")
        box.pack(fill="x")
        ttk.Button(box, text="Save corrections",
                   command=self.save).pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(box, text="Restore this frame",
                   command=self.restore_frame).pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(box, text="Restore everything",
                   command=self.restore_all).pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(box, text="Undo  (Ctrl+Z)",
                   command=self.undo).pack(fill="x", padx=8, pady=(0, 8))
        self.backup_label = ttk.Label(box, text="", foreground="gray",
                                      wraplength=270)
        self.backup_label.pack(anchor="w", padx=8, pady=(0, 8))

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

        # ---- transport
        transport = ttk.Frame(area)
        transport.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(transport, text="|◀", width=4,
                   command=lambda: self.seek(0)).pack(side="left")
        ttk.Button(transport, text="◀◀", width=4,
                   command=lambda: self.seek(self.frame_index - 10)).pack(side="left")
        ttk.Button(transport, text="◀", width=4,
                   command=lambda: self.seek(self.frame_index - 1)).pack(side="left")
        ttk.Button(transport, text="▶", width=4,
                   command=lambda: self.seek(self.frame_index + 1)).pack(side="left")
        ttk.Button(transport, text="▶▶", width=4,
                   command=lambda: self.seek(self.frame_index + 10)).pack(side="left")

        self.frame_label = ttk.Label(transport, text="frame 0 / 0", width=18)
        self.frame_label.pack(side="left", padx=10)

        self.slider = ttk.Scale(transport, from_=0, to=0, orient="horizontal",
                                command=self.on_slider)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)

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

    # ==================================================================
    # Loading
    # ==================================================================
    def choose_folder(self):
        path = filedialog.askdirectory(
            title="Select the folder holding the camera subfolders")
        if path:
            self.load_folder(path)

    def choose_project(self):
        path = filedialog.askdirectory(title="Select the project folder")
        if not path:
            return
        found = find_pose_folders(path)
        if not found:
            messagebox.showinfo(
                "Nothing to correct",
                f"No keypoint folders under\n{path}\n\nRun pose estimation first.")
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
            seq = Sequence(pose_dir)
        except Exception as exc:
            messagebox.showerror("Could not read that folder", str(exc))
            return

        self.seq = seq
        self.pose_dir = pose_dir
        self.names, self.links, self.left, self.right = skeleton_for(seq.n_keypoints)
        self.frame_index = 0
        self.keypoint = 0
        self.active_cam = 0

        self.folder_label.configure(text=str(pose_dir))
        self.kp_combo.configure(values=[f"{n}  ({i})" for i, n in enumerate(self.names)])
        self.kp_combo.set(f"{self.names[0]}  (0)")
        self.person_combo.configure(
            values=[str(i + 1) for i in range(max(seq.max_people, 1))])
        self.person_var.set("1")
        self.slider.configure(to=max(seq.n_frames - 1, 0))

        self.build_camera_views()
        self.update_backup_label()
        self.show_frame(0)
        self.status_text.set(
            f"{seq.n_cameras} cameras · {seq.n_frames} frames · "
            f"{seq.n_keypoints} keypoints · drag a point to fix it")
        self.scan()

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
        c.bind("<MouseWheel>", lambda e, v=view: self.on_wheel(v, e))
        c.bind("<Button-4>", lambda e, v=view: self.on_wheel(v, e, 1))
        c.bind("<Button-5>", lambda e, v=view: self.on_wheel(v, e, -1))

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

    def on_press(self, view, event):
        if self.seq is None:
            return
        self.active_cam = view["index"]
        i, d = self.nearest(view, event.x, event.y)
        if i >= 0 and d <= self.HIT_RADIUS:
            self.select_keypoint(i)
            view["drag"] = {"kp": i, "moved": False}

    def on_drag(self, view, event):
        if not view["drag"] or self.seq is None:
            return
        x, y = self.to_image(view, event.x, event.y)
        # a hand-placed point is certain, so it is written with confidence 1.0
        self.seq.set_point(view["index"], self.frame_index, view["drag"]["kp"],
                           float(x), float(y), 1.0,
                           record=not view["drag"]["moved"])
        view["drag"]["moved"] = True
        self.redraw(view)
        self.update_readout()

    def on_release(self, view, event):
        if view["drag"] and view["drag"]["moved"]:
            self.carry_forward(view["index"], view["drag"]["kp"])
            self.status_text.set(
                f"Moved {self.names[view['drag']['kp']]} in {view['name']} "
                f"— {self.seq.unsaved} frame(s) unsaved")
        view["drag"] = None

    def on_right_click(self, view, event):
        if self.seq is None:
            return
        i, d = self.nearest(view, event.x, event.y)
        if i >= 0 and d <= self.HIT_RADIUS:
            self.seq.clear_point(view["index"], self.frame_index, i)
            self.redraw(view)
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
        self.update_readout()

    def redraw(self, view):
        c = view["canvas"]
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4 or self.seq is None:
            return

        # background: the video frame if we have one, else a grid so the
        # transform is still readable
        image = self.video_frame(view, self.frame_index)
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

        pts = self.seq.frame(view["index"], self.frame_index)
        present = (pts[:, 2] > 0) | (pts[:, 0] != 0) | (pts[:, 1] != 0)
        threshold = float(self.conf_threshold.get())

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
            if selected:
                r = radius + 5
                c.create_oval(x - r, y - r, x + r, y + r,
                              outline=self.COLOUR_SELECTED, width=2)

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

        # camera name, and whether this frame has unsaved edits
        label = view["name"]
        if self.frame_index in self.seq.dirty[view["index"]]:
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
        self.kp_combo.set(f"{self.names[self.keypoint]}  ({self.keypoint})")
        self.redraw_all()

    def on_keypoint_selected(self, _event=None):
        text = self.kp_combo.get()
        try:
            self.select_keypoint(int(text.rsplit("(", 1)[1].rstrip(")")))
        except (ValueError, IndexError):
            pass

    def on_person_selected(self, _event=None):
        if self.seq is None:
            return
        if self.seq.unsaved:
            messagebox.showwarning(
                "Unsaved corrections",
                "Save or restore your corrections before switching person — "
                "they belong to the person currently loaded.")
            self.person_var.set(str(self.seq.person + 1))
            return
        self.seq = Sequence(self.pose_dir, person=int(self.person_var.get()) - 1)
        self.show_frame(self.frame_index)
        self.scan()

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

    def show_frame(self, frame):
        self.frame_index = frame
        self.frame_label.configure(text=f"frame {frame} / {self.seq.n_frames - 1}")
        try:
            self.slider.set(frame)
        except Exception:
            pass
        self.redraw_all()

    def carry_forward(self, cam, kp):
        """Apply the same shift to the next N frames, fading out.

        After a manual fix the following frames usually need the same
        correction, decaying as the tracker recovers.
        """
        n = int(self.carry_frames.get() or 0)
        if n <= 0 or self.seq is None:
            return
        fixed = self.seq.point(cam, self.frame_index, kp)[:2].copy()
        base = fixed.copy()
        moved = 0
        for step in range(1, n + 1):
            f = self.frame_index + step
            if f >= self.seq.n_frames:
                break
            current = self.seq.point(cam, f, kp)
            weight = 1.0 - step / (n + 1)
            if current[2] > 0:
                nx, ny = current[:2] + (fixed - base) * weight
            else:
                nx, ny = fixed
            self.seq.set_point(cam, f, kp, float(nx), float(ny),
                               float(max(current[2], 0.5)))
            moved += 1
        if moved:
            self.status_text.set(f"Carried the fix forward {moved} frame(s)")

    def swap_lr(self):
        if self.seq is None:
            return
        partner = None
        if self.keypoint in self.left:
            partner = self.right[self.left.index(self.keypoint)]
        elif self.keypoint in self.right:
            partner = self.left[self.right.index(self.keypoint)]
        if partner is None:
            messagebox.showinfo(
                "No counterpart",
                f"{self.names[self.keypoint]} has no left/right partner.")
            return
        self.seq.swap_lr(self.active_cam, self.frame_index, self.keypoint, partner)
        self.redraw_all()
        self.status_text.set(
            f"Swapped {self.names[self.keypoint]} and {self.names[partner]}")

    def clear_point(self):
        if self.seq is None:
            return
        self.seq.clear_point(self.active_cam, self.frame_index, self.keypoint)
        self.redraw_all()
        self.status_text.set(f"Cleared {self.names[self.keypoint]}")

    def undo(self):
        if self.seq and self.seq.undo():
            self.redraw_all()
            self.status_text.set("Undone")

    # ==================================================================
    # Scanning
    # ==================================================================
    def scan(self):
        """List the points worth a look: low confidence, jumps, gaps."""
        self.problem_list.delete(0, "end")
        self.problems = []
        if self.seq is None:
            return

        threshold = float(self.conf_threshold.get())
        for cam in range(self.seq.n_cameras):
            block = self.seq.data[cam]
            conf = block[:, :, 2]
            xy = block[:, :, :2]
            present = (conf > 0) & ~((xy[:, :, 0] == 0) & (xy[:, :, 1] == 0))

            for kp in range(self.seq.n_keypoints):
                column = present[:, kp]
                if column.sum() < 3:
                    continue

                low = np.flatnonzero(column & (conf[:, kp] < threshold))
                for first, last in contiguous(low):
                    self.problems.append(
                        (cam, kp, int(first),
                         f"low confidence  {conf[first:last + 1, kp].min():.2f}",
                         float(threshold - conf[first:last + 1, kp].min())))

                step = np.full(len(block), np.nan)
                pair = column[1:] & column[:-1]
                step[1:][pair] = np.linalg.norm(np.diff(xy[:, kp], axis=0),
                                                axis=1)[pair]
                median = np.nanmedian(step)
                limit = max(median, 2.0) * 6
                jumps = np.flatnonzero(np.nan_to_num(step) > limit)
                for first, last in contiguous(jumps):
                    self.problems.append(
                        (cam, kp, int(first),
                         f"jumps {np.nanmax(step[first:last + 1]):.0f} px",
                         float(np.nanmax(step[first:last + 1]) / max(limit, 1))))

                gaps = ~column
                inside = np.flatnonzero(column)
                if len(inside):
                    gaps[: inside[0]] = False
                    gaps[inside[-1] + 1:] = False
                for first, last in contiguous(np.flatnonzero(gaps)):
                    self.problems.append(
                        (cam, kp, int(first),
                         f"missing for {last - first + 1} frame(s)",
                         1.0 + (last - first) / 10))

        self.problems.sort(key=lambda p: -p[4])
        for cam, kp, frame, detail, _sev in self.problems[:300]:
            self.problem_list.insert(
                "end", f" {self.names[kp]:<12} {self.seq.cam_names[cam]:<8} "
                       f"f{frame:<5} {detail}")
        if not self.problems:
            self.problem_list.insert("end", "  nothing suspect found")
        self.status_text.set(f"{len(self.problems)} suspect point(s)")

    def on_problem_selected(self, _event=None):
        selection = self.problem_list.curselection()
        if not selection or not getattr(self, "problems", None):
            return
        index = selection[0]
        if index >= len(self.problems):
            return
        cam, kp, frame, _detail, _sev = self.problems[index]
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
    def save(self):
        if self.seq is None or not self.seq.unsaved:
            self.status_text.set("Nothing to save.")
            return

        first_time = not self.seq.has_backup
        message = f"{self.seq.unsaved} frame(s) changed.\n\n"
        if first_time:
            message += (
                f"The whole '{self.seq.root.name}' folder will be copied to "
                f"'{self.seq.backup_dir.name}' first — every camera, "
                f"untouched.\n"
                f"Your corrections then go into '{self.seq.root.name}', the "
                f"folder Pose2Sim reads when it triangulates.\n\nProceed?")
        else:
            message += (
                f"'{self.seq.backup_dir.name}' already holds the originals and "
                f"will not be touched.\nCorrections go into "
                f"'{self.seq.root.name}'.\n\nProceed?")
        if not messagebox.askyesno("Save corrections", message):
            return

        try:
            report = self.seq.save()
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.update_backup_label()
        self.redraw_all()
        note = "  Originals backed up." if report["created_backup"] else ""
        self.status_text.set(
            f"Wrote {report['written']} file(s) into "
            f"{self.seq.root.name}.{note}")

    def restore_frame(self):
        if self.seq is None:
            return
        if not self.seq.has_backup:
            self.status_text.set("No backup folder yet — nothing to restore.")
            return
        n = self.seq.restore_frame(self.frame_index)
        self.redraw_all()
        self.status_text.set(
            f"Restored frame {self.frame_index} in {n} camera(s) from the backup")

    def restore_all(self):
        if self.seq is None or not self.seq.has_backup:
            self.status_text.set("No backup folder yet — nothing to restore.")
            return
        if not messagebox.askyesno(
                "Restore everything",
                f"Copy every file from '{self.seq.backup_dir.name}' back over "
                f"'{self.seq.root.name}', discarding all corrections?\n\n"
                f"This cannot be undone."):
            return
        n = self.seq.restore_all()
        self.redraw_all()
        self.scan()
        self.status_text.set(f"Restored {n} file(s) from the backup")

    def update_backup_label(self):
        if self.seq is None:
            return
        if self.seq.has_backup:
            self.backup_label.configure(
                text=f"Originals safe in {self.seq.backup_dir.name} "
                     f"({self.seq.backup_count()} files)")
        else:
            self.backup_label.configure(
                text="No backup yet. The first save creates one.")

    def update_readout(self):
        if self.seq is None:
            return
        parts = [f"{self.names[self.keypoint]}  f{self.frame_index}"]
        for view in self.views:
            point = self.seq.point(view["index"], self.frame_index, self.keypoint)
            mark = "*" if view["index"] == self.active_cam else " "
            if point[2] <= 0:
                parts.append(f"{mark}{view['name']} --")
            else:
                parts.append(f"{mark}{view['name']} {point[2]:.2f}")
        self.readout.configure(text="   ".join(parts))

    def on_close(self):
        if self.seq and self.seq.unsaved:
            answer = messagebox.askyesnocancel(
                "Unsaved corrections",
                f"{self.seq.unsaved} frame(s) have unsaved changes.\n\nSave "
                f"before closing?")
            if answer is None:
                return
            if answer:
                self.save()
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
