from enum import Enum
from functools import partial
import ast
import json
import logging
from typing import Any, Dict

from anytree.importer import DictImporter

from rtmlib import BodyWithFeet, Wholebody, Body, Hand, Custom

from Pose2Sim.skeletons import (
    HALPE_26, COCO_133_WRIST, COCO_133, COCO_17, HAND_21, FACE_106,
    ANIMAL2D_17, BODY_25B, BODY_25, BODY_135, BLAZEPOSE,
    HALPE_68, HALPE_136, COCO, MPII
)


class PoseModelEnum(Enum):
    #                 marker_file                scaling_file                              ik_file                                 model_class    skeleton
    BODY_25B       = ("Markers_Body25b.xml",     "Scaling_Setup_Pose2Sim_Body25b.xml",     "IK_Setup_Pose2Sim_Body25b.xml",        None,          BODY_25B)
    BODY_25        = ("Markers_Body25.xml",      "Scaling_Setup_Pose2Sim_Body25.xml",      "IK_Setup_Pose2Sim_Body25.xml",         None,          BODY_25)
    BODY_135       = ("Markers_Body135.xml",     "Scaling_Setup_Pose2Sim_Body135.xml",     "IK_Setup_Pose2Sim_Body135.xml",        None,          BODY_135)
    BLAZEPOSE      = ("Markers_Blazepose.xml",   "Scaling_Setup_Pose2Sim_Blazepose.xml",   "IK_Setup_Pose2Sim_Blazepose.xml",      None,          BLAZEPOSE)
    HALPE_26       = ("Markers_Halpe26.xml",     "Scaling_Setup_Pose2Sim_Halpe26.xml",     "IK_Setup_Pose2Sim_Halpe26.xml",        BodyWithFeet,  HALPE_26)
    HALPE_68       = ("Markers_Halpe68_136.xml", "Scaling_Setup_Pose2Sim_Halpe68_136.xml", "IK_Setup_Pose2Sim_Halpe68_136.xml",    None,          HALPE_68)
    HALPE_136      = ("Markers_Halpe68_136.xml", "Scaling_Setup_Pose2Sim_Halpe68_136.xml", "IK_Setup_Pose2Sim_Halpe68_136.xml",    None,          HALPE_136)
    COCO_133       = ("Markers_Coco133.xml",     "Scaling_Setup_Pose2Sim_Coco133.xml",     "IK_Setup_Pose2Sim_Coco133.xml",        Wholebody,     COCO_133)
    COCO_133_WRIST = ("Markers_Coco133.xml",     "Scaling_Setup_Pose2Sim_Coco133.xml",     "IK_Setup_Pose2Sim_Coco133.xml",        Wholebody,     COCO_133_WRIST)
    COCO_17        = ("Markers_Coco17.xml",      "Scaling_Setup_Pose2Sim_Coco17.xml",      "IK_Setup_Pose2Sim_Coco17.xml",         Body,          COCO_17)
    HAND_21        = ("",                        "",                                       "",                                     Hand,          HAND_21)
    FACE_106       = ("",                        "",                                       "",                                     None,          FACE_106)
    ANIMAL2D_17    = ("",                        "",                                       "",                                     None,          ANIMAL2D_17)
    COCO           = ("",                        "",                                       "",                                     None,          COCO)
    MPII           = ("",                        "",                                       "",                                     None,          MPII)
    LSTM           = ("Markers_LSTM.xml",        "Scaling_Setup_Pose2Sim_LSTM.xml",        "IK_Setup_Pose2Sim_withHands_LSTM.xml", None,          None)

    def __init__(self, marker_file: str, scaling_file: str, ik_file: str, model_class: Any, skeleton: Any):
        self.marker_file = marker_file
        self.scaling_file = scaling_file
        self.ik_file = ik_file
        self.model_class = model_class
        self.skeleton = skeleton

ALIASES = {
    "BODY_WITH_FEET":    "HALPE_26",
    "WHOLE_BODY_WRIST":  "COCO_133_WRIST",
    "WHOLE_BODY":        "COCO_133",
    "BODY":              "COCO_17",
    "HAND":              "HAND_21",
    "FACE":              "FACE_106",
    "ANIMAL":            "ANIMAL2D_17",
}

class PoseModel:
    def __init__(self, pose_model: str, backend: str = "auto", device: str = "auto",
                 det_frequency: int = 1, mode: str = "balanced"):
        name = ALIASES.get(pose_model.upper(), pose_model.upper())
        try:
            self._enum = PoseModelEnum[name]
            for attr in ("marker_file", "scaling_file", "ik_file", "model_class", "skeleton"):
                setattr(self, attr, getattr(self._enum, attr))
        except KeyError:
            logging.info(f"Unknown pose model : {pose_model}")

        self.backend, self.device = init_backend_device(backend, device)

        self.det_frequency = det_frequency
        self.mode = mode
        self.to_openpose = True

        if self.model_class is not None:
            self.det_input_size = self.model_class.MODE[self.mode]['det_input_size']
        else:
            try:
                params = self._parse_mode_dict(self.mode)
                det_class = params['det_class']
                det = params['det_model']
                self.det_input_size = params['det_input_size']
                pose_class = params['pose_class']
                pose = params['pose_model']
                pose_input_size = params['pose_input_size']

                self.model_class = partial(
                    Custom,
                    det_class=det_class, det=det, det_input_size=self.det_input_size,
                    pose_class=pose_class, pose=pose, pose_input_size=pose_input_size,
                    backend=self.backend, device=self.device,
                )
            except Exception:
                raise NameError(f"{name} invalid model settings. Must be 'lightweight', 'balanced', 'performance', or a dictionary of parameters defined in Config.toml.")

        if self.skeleton is None:
            try:
                self.skeleton = DictImporter().import_(name)
                if self.skeleton.id == 'None':
                    self.skeleton.id = None
            except Exception:
                raise NameError(f"Skeleton {name} not found in skeletons.py nor in Config.toml")

        logging.info(f"Using model {name} for pose estimation in mode : {self.mode}")

    @staticmethod
    def _parse_mode_dict(mode) -> Dict[str, Any]:
        try:
            return ast.literal_eval(mode)
        except Exception:
            cleaned = (
                str(mode).strip("'")
                .replace('\n', '').replace(' ', '')
                .replace(',', '", "').replace(':', '":"')
                .replace('{', '{"').replace('}', '"}')
                .replace('":"[', '":[')
            )
            return json.loads(cleaned)


def init_backend_device(backend: str, device: str):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if backend == device == "auto":
        try:
            import torch  # type: ignore
            import onnxruntime as ort  # type: ignore
            if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
                logging.info("Valid CUDA installation found: using ONNXRuntime backend with GPU.")
                return "onnxruntime", "cuda"
            elif torch.cuda.is_available() and "ROCMExecutionProvider" in ort.get_available_providers():
                logging.info("Valid ROCM installation found: using ONNXRuntime backend with GPU.")
                return "onnxruntime", "rocm"
            if {"MPSExecutionProvider", "CoreMLExecutionProvider"} & set(ort.get_available_providers()):
                logging.info("Valid MPS installation found: using ONNXRuntime backend with GPU.")
                return "onnxruntime", "mps"
        except Exception:
            pass
        logging.info("No valid CUDA installation found: using OpenVINO backend with CPU.")
        return "openvino", "cpu"

    if backend == "auto" or device == "auto":
        logging.warning("If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")
        return init_backend_device("auto", "auto")

    return backend.lower(), device.lower()
