"""The (i) tooltips of the pose correction window."""

import tkinter as tk
from tkinter import ttk


## AUTHORSHIP INFORMATION
__author__ = "AY LARDJANE"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["AY LARDJANE"]
__license__ = "BSD 3-Clause License"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"

TIPS = {
    # Sections
    "keypoints": (
        "Where the points come from, and how suspect points are found.\n\n"
        "Pose2Sim project: the tool reads the 2D points of each camera (one JSON file per frame, "
        "OpenPose format: x, y, confidence c for each keypoint), the videos, and Calib_*.toml.\n\n"
        "Scan runs three tests on every point p(t) of every camera, with the limits below:\n"
        "  1. confidence: c < threshold  ->  red ring;\n"
        "  2. jump: ||p(t) - p(t-1)|| / H(t) x 170 cm > limit  ->  orange ring;\n"
        "  3. cameras: ||p(t) - p_others(t)|| / H(t) x 170 cm > limit  ->  purple ring.\n"
        "H(t) is the height of the person in pixels (see Flag jumps), so the limits are in "
        "centimetres whatever the distance to the camera."),
    "suspect": (
        "Every point that failed a test, one line each.\n\n"
        "Most suspect first: the lines are sorted by how far each point is beyond its limit (the "
        "worst ratio first). You can also sort by frame, camera or keypoint.\n\n"
        "Click a line: the tool jumps to that frame and that camera, and selects the point. Rings: "
        "red, low confidence; orange, jump; purple, the other cameras disagree; blue, moved by a "
        "correction (Weight); green, placed by hand."),
    "edit": (
        "Correcting one point by hand.\n\n"
        "Click: selects the nearest point. Drag: the point takes the position of the mouse, its "
        "confidence becomes 1 (placed by hand), and the move d = p_new - p_old is also spread over "
        "the neighbouring frames by the Weight.\n\n"
        "Right click: the point is cleared (x = y = 0, c = 0). Pose2Sim then triangulates it with the "
        "other cameras only, which is better than a wrong point.\n\n"
        "Every edit is written to memory only, until Save."),
    "history": (
        "Every corrected point, by frame, camera and keypoint. Click one to go there.\n\n"
        "A point counts as corrected when it differs from the detection loaded at opening. Nothing is "
        "written to the JSON files until Save. At the first save, the original files are copied once to "
        "a backup folder: Restore everything can always bring them back."),
    # Opening
    "open_project": (
        "Opens a Pose2Sim trial folder.\n\n"
        "The tool looks for the 2D points in pose-associated, then pose-sync, then pose (the most "
        "processed first), with one subfolder per camera. It pairs each camera with its video (a file "
        "whose name contains the camera name, in videos) and finds Calib_*.toml in calibration.\n\n"
        "The files of all cameras are put on the same timeline: row t of every camera is the same "
        "instant, which the camera test needs."),
    "open_folder": (
        "Opens a folder of 2D points chosen by hand: one folder per camera, one JSON file per frame "
        "(OpenPose format, pose_keypoints_2d = [x1, y1, c1, x2, y2, c2, ...]). Use it when the files are "
        "not in a usual Pose2Sim project. Without a calibration, the camera test and Place where other "
        "cameras see it are not available."),
    "calibration": (
        "Calib_*.toml: for each camera, its matrix K, its distortion and its position (R, t), as written "
        "by Pose2Sim. The tool builds the projection matrix of each camera:\n"
        "  P = K [R | t]   (3 x 4),   and a 3D point X is seen at  x ~ P X.\n"
        "This is what lets one camera be checked against the others."),
    # Scan
    "low_conf": (
        "The detector gives each point a confidence c between 0 and 1 (its heatmap score). A point with "
        "c below this value is ringed in red.\n\n"
        "Pose2Sim also ignores points below its own likelihood threshold when it triangulates, so a "
        "red point is often a point that will be missing in 3D. Typical values: 0.3 to 0.6."),
    "jumps": (
        "The move of a point between two frames, in centimetres:\n"
        "  jump(t) = ||p(t) - p(t-1)|| / H(t) x 170 cm\n"
        "H(t) is the height of the person in pixels, as compute_height in Pose2Sim: heel to ankle "
        "+ shank + thigh + hip to shoulder + shoulders to head (x 1.008, or to the nose x 1.5), each "
        "the mean of both sides, so it does not shrink when the person crouches. Where a segment is "
        "missing, or with a model without these points, the vertical extent of the detected points "
        "(lowest minus highest) is used. Then the median over 121 frames around t (about 4 s), so a "
        "wrong point does not change it. 170 cm is the assumed height.\n\n"
        "A jump above the limit (15 cm by default) rings the point in orange: a limb rarely moves 15 cm "
        "in 1/30 s, a detection error often does."),
    "cameras": (
        "Where the other cameras put the point, by triangulation (DLT, weighted by confidence):\n"
        "  for each camera i:  c_i (x_i P_i3 - P_i1) X = 0,   c_i (y_i P_i3 - P_i2) X = 0\n"
        "  X = last singular vector of the stacked rows (SVD), then x = P X on this camera.\n\n"
        "The cameras vote: the largest group whose points all fall within 5 % of the height of the "
        "triangulated point wins. If this camera is outside the winning group, it is the odd one out.\n\n"
        "Distance between the detection and that projection above the limit (10 cm, 6 cm for the "
        "feet): purple ring. It finds points the detector was sure of, and wrong about."),
    "scan": (
        "Runs the three tests (confidence, jump, cameras) on every point of every camera and every "
        "frame, with the current limits, and lists the failures in Suspect points. Points with fewer "
        "than 3 detections in the whole recording are skipped. Click Scan again after changing a limit."),
    # Edit
    "place": (
        "Puts the selected point where the other cameras see it.\n\n"
        "The point is triangulated from the winning group of cameras without this one (DLT weighted by "
        "confidence, see Flag camera disagreement), then projected on this camera with its "
        "distortion:  p_new = project(P_this, X_others).\n\n"
        "Needs the calibration, at least 3 cameras, and 2 other cameras that agree: one other camera "
        "alone always agrees, which proves nothing. The move is spread by the Weight like a drag."),
    "swap_lr": (
        "Exchanges the left and right points of the selected body part (left knee with right knee, "
        "and so on): the two rows (x, y, c) of the file are swapped.\n\n"
        "Detectors mix them up when the person is seen from the side or turns around, because both "
        "sides look the same. With a Weight w > 0, the swap is applied to frames t - w to t + w."),
    "swap_person": (
        "Exchanges this point between two people of the same file: p_me <-> p_other, for this keypoint "
        "and this frame (and the frames within the Weight).\n\n"
        "When two people are close, the detector may attach a wrist to the wrong body. Choose the other "
        "person on the right, then click. Only these two entries of the file change; one Undo puts "
        "both people back."),
    "weight": (
        "How far a correction spreads in time. When you move a point by d at frame t, frame t + k "
        "(on both sides) moves by:\n"
        "  d_k = w(k) d,   w(k) = 0.5 (1 + cos(pi k / (W + 1))),   |k| <= W\n"
        "so w = 1 at the frame, 0.5 halfway, 0 just beyond W frames (half a cosine: no break in speed).\n\n"
        "The spread stops at a frame without a file and at a point placed by hand. W = 0: this frame "
        "only."),
    "undo": (
        "Cancels the last edit as a whole (Ctrl+Z): the point and every frame that the Weight or a swap "
        "moved with it. Each edit is kept on a stack: click again to go further back."),
    "delete": (
        "Puts the point selected in History back to the detection: p = p_detected, on that frame and "
        "camera only (the frames moved with it stay). Undo brings your correction back."),
    "restore": (
        "Removes every correction. Before the first save: the points go back to the detection in "
        "memory. After it: every JSON file is copied back from the backup folder made at the first "
        "save, and the Undo stack is emptied."),
    "save": (
        "Writes the corrections (Ctrl+S). First save: the original JSON files are copied to a backup "
        "folder, once. Then, for every frame changed, only this person's entry is rewritten "
        "(pose_keypoints_2d, values rounded to 0.001 px); the other people of the file are left exactly "
        "as they were.\n\n"
        "Run the triangulation again afterwards: the 3D is computed from these files."),
    "original": (
        "Shows where the detector had each corrected point: a dashed white circle at p_detected, linked "
        "by a line to your point, to compare. Key O shows or hides them."),
}


def text(key):
    return TIPS[key]


class HoverTip:
    """A small window with the explanation, while the mouse is on the widget."""

    def __init__(self, widget, key, delay=400):
        self.widget, self.key, self.delay = widget, key, delay
        self.window = self.job = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None):
        self.cancel()
        self.job = self.widget.after(self.delay, self.show)

    def cancel(self):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None

    def show(self):
        if self.window is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(self.window, text=text(self.key), justify="left", wraplength=470,
                 background="#fffbe6", relief="solid", borderwidth=1,
                 font=("Consolas", 9), padx=10, pady=8).pack()

    def hide(self, _event=None):
        self.cancel()
        if self.window is not None:
            self.window.destroy()
            self.window = None


def tip(widget, key):
    """The explanation on a button or a checkbox."""
    HoverTip(widget, key)
    return widget


def info(parent, key):
    """An (i) next to a setting or a title."""
    label = ttk.Label(parent, text="ⓘ", foreground="#2563eb", cursor="question_arrow")
    HoverTip(label, key, delay=100)
    return label


def section(parent, title, key):
    """A LabelFrame whose title carries an (i)."""
    head = ttk.Frame(parent)
    ttk.Label(head, text=title).pack(side="left")
    info(head, key).pack(side="left", padx=(4, 0))
    return ttk.LabelFrame(parent, labelwidget=head)
