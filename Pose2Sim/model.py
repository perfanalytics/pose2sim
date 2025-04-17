from enum import Enum
from functools import partial
import logging
from anytree.importer import DictImporter
import re
import json
import ast

from rtmlib import BodyWithFeet, Wholebody, Body, Hand, Custom
from Pose2Sim.skeletons import HALPE_26, COCO_133_WRIST, COCO_133, COCO_17, HAND_21, FACE_106, ANIMAL2D_17, BODY_25B, BODY_25, BODY_135, BLAZEPOSE, HALPE_68, HALPE_136, COCO, MPII

class PoseModelEnum(Enum):
    BODY_25B       = ('Markers_Body25b.xml',      'Scaling_Setup_Pose2Sim_Body25b.xml',      'IK_Setup_Pose2Sim_Body25b.xml')
    BODY_25        = ('Markers_Body25.xml',       'Scaling_Setup_Pose2Sim_Body25.xml',       'IK_Setup_Pose2Sim_Body25.xml')
    BODY_135       = ('Markers_Body135.xml',      'Scaling_Setup_Pose2Sim_Body135.xml',      'IK_Setup_Pose2Sim_Body135.xml')
    BLAZEPOSE      = ('Markers_Blazepose.xml',    'Scaling_Setup_Pose2Sim_Blazepose.xml',    'IK_Setup_Pose2Sim_Blazepose.xml')
    HALPE_26       = ('Markers_Halpe26.xml',      'Scaling_Setup_Pose2Sim_Halpe26.xml',      'IK_Setup_Pose2Sim_Halpe26.xml')
    HALPE_68       = ('Markers_Halpe68_136.xml',  'Scaling_Setup_Pose2Sim_Halpe68_136.xml',  'IK_Setup_Pose2Sim_Halpe68_136.xml')
    HALPE_136      = ('Markers_Halpe68_136.xml',  'Scaling_Setup_Pose2Sim_Halpe68_136.xml',  'IK_Setup_Pose2Sim_Halpe68_136.xml')
    COCO_133       = ('Markers_Coco133.xml',      'Scaling_Setup_Pose2Sim_Coco133.xml',      'IK_Setup_Pose2Sim_Coco133.xml')
    COCO_133_WRIST = ('Markers_Coco133.xml',      'Scaling_Setup_Pose2Sim_Coco133.xml',      'IK_Setup_Pose2Sim_Coco133.xml')
    COCO_17        = ('Markers_Coco17.xml',       'Scaling_Setup_Pose2Sim_Coco17.xml',       'IK_Setup_Pose2Sim_Coco17.xml')
    LSTM           = ('Markers_LSTM.xml',         'Scaling_Setup_Pose2Sim_LSTM.xml',         'IK_Setup_Pose2Sim_withHands_LSTM.xml')

    def __init__(self, marker_file: str, scaling_file: str, ik_file: str):
        self.marker_file = marker_file
        self.scaling_file = scaling_file
        self.ik_file = ik_file
        self.model_class = self.get_model_class()
        self.skeleton = self.get_skeleton()

    def get_model_class(self):
        """
        Renvoie la classe associée au modèle de pose.
        """
        if self.name == 'HALPE_26':
            logging.info("Using HALPE_26 model (body and feet) for pose estimation.")
            return BodyWithFeet
        elif self.name in ('COCO_133', 'COCO_133_WRIST'):
            logging.info("Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
            return Wholebody
        elif self.name == 'COCO_17':
            logging.info("Using COCO_17 model (body) for pose estimation.")
            return Body
        elif self.name == 'HAND_21':
            logging.info("Using HAND_21 model for pose estimation.")
            return Hand
        elif self.name == 'FACE_106':
            logging.info("Using FACE_106 model for pose estimation.")
            return None
        elif self.name == 'ANIMAL2D_17':
            logging.info("Using ANIMAL2D_17 model for pose estimation.")
            return None
        else:
            logging.info(f"Using model {self.name} for pose estimation.")
            return None

    def get_skeleton(self):
        """
        Renvoie le squelette associé au modèle.
        """
        if self.name == "LSTM":
            return None
        mapping = {
            'BODY_25B':      BODY_25B,
            'BODY_25':       BODY_25,
            'BODY_135':      BODY_135,
            'BLAZEPOSE':     BLAZEPOSE,
            'HALPE_26':      HALPE_26,
            'HALPE_68':      HALPE_68,
            'HALPE_136':     HALPE_136,
            'COCO_133':      COCO_133,
            'COCO_133_WRIST':COCO_133_WRIST,
            'COCO_17':       COCO_17,
            'HAND_21':       HAND_21,
            'FACE_106':      FACE_106,
            'ANIMAL2D_17':   ANIMAL2D_17,
            'COCO':          COCO,
            'MPII':          MPII,
        }
        try:
            return mapping[self.name]
        except KeyError:
            raise ValueError(f"No skeleton associated with the model: {self.name}.")


class PoseModel:
    def __init__(self, config):
        self.pose_model_enum = self.get_pose_model(config.pose_model)
        self.backend, self.device = init_backend_device(config.backend, config.device)
        self.det_frequency = config.det_frequency
        self.mode = config.mode
        self.to_openpose = True

        self.load_model_instance()

    def get_pose_model(self, pose_model):
        mapping = {
            'BODY_WITH_FEET': 'HALPE_26',
            'WHOLE_BODY_WRIST': 'COCO_133_WRIST',
            'WHOLE_BODY': 'COCO_133',
            'BODY': 'COCO_17',
            'HAND': 'HAND_21',
            'FACE': 'FACE_106',
            'ANIMAL': 'ANIMAL2D_17',
        }
        key = pose_model.upper()
        if key in mapping:
            key = mapping[key]
        try:
            return PoseModelEnum[key]
        except KeyError:
            raise ValueError(f"{pose_model}")

    def load_model_instance(self):
        if self.pose_model_enum.model_class is None:
            try:
                try:
                    mode = ast.literal_eval(mode)
                except:  # if within single quotes instead of double quotes when run with sports2d --mode """{dictionary}"""
                    mode = mode.strip("'").replace('\n', '').replace(" ", "").replace(",", '", "').replace(":", '":"').replace("{", '{"').replace("}", '"}').replace('":"/', ':/').replace('":"\\', ':\\')
                    mode = re.sub(r'"\[([^"]+)",\s?"([^"]+)\]"', r'[\1,\2]', mode)  # changes "[640", "640]" to [640,640]
                    mode = json.loads(mode)
                det_class = mode.get('det_class')
                det = mode.get('det_model')
                self.det_input_size = mode.get('det_input_size')
                pose_class = mode.get('pose_class')
                pose = mode.get('pose_model')
                pose_input_size = mode.get('pose_input_size')
                self.pose_model_enum.model_class = partial(Custom,
                                        det_class=det_class, det=det, det_input_size=self.det_input_size,
                                        pose_class=pose_class, pose=pose, pose_input_size=pose_input_size,
                                        backend=self.backend, device=self.device)
            except Exception:
                raise NameError(f"{self.name} invalid mode. Must be 'lightweight', 'balanced', 'performance', or a dictionary of parameters defined in Config.toml.")
        else:
            self.det_input_size = self.pose_model_enum.model_class.MODE[self.mode]['det_input_size']

        if self.pose_model_enum.skeleton is None:
            try:
                self.pose_model_enum.skeleton = DictImporter().import_(self.name)
                if self.pose_model_enum.skeletonton.id == 'None':
                    self.pose_model_enum.skeleton.id = None
            except Exception:
                raise NameError(f"Skeleton {self.name} not found in skeletons.py nor in Config.toml")

        logging.info(f'\nPose tracking set up for "{self.name}" model.')
        logging.info(f'Mode: {self.mode}.')


def init_backend_device(backend, device):
    '''
    Set up the backend and device for the pose tracker based on the availability of hardware acceleration.
    TensorRT is not supported by RTMLib yet: https://github.com/Tau-J/rtmlib/issues/12

    If device and backend are not specified, they are automatically set up in the following order of priority:
    1. GPU with CUDA and ONNXRuntime backend (if CUDAExecutionProvider is available)
    2. GPU with ROCm and ONNXRuntime backend (if ROCMExecutionProvider is available, for AMD GPUs)
    3. GPU with MPS or CoreML and ONNXRuntime backend (for macOS systems)
    4. CPU with OpenVINO backend (default fallback)
    '''

    if device == 'auto' or backend == 'auto':
        if device != 'auto' or backend != 'auto':
            logging.warning("If you set device or backend to 'auto', you must set the other to 'auto' as well. Both device and backend will be determined automatically.")

        try:
            import torch # type: ignore
            import onnxruntime as ort # type: ignore
            if torch.cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
                logging.info("Valid CUDA installation found: using ONNXRuntime backend with GPU.")
                return 'onnxruntime', 'cuda'
            elif torch.cuda.is_available() and 'ROCMExecutionProvider' in ort.get_available_providers():
                logging.info("Valid ROCM installation found: using ONNXRuntime backend with GPU.")
                return 'onnxruntime', 'rocm'
            else:
                raise
        except:
            try:
                import onnxruntime as ort # type: ignore
                if ('MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers()):
                    logging.info("Valid MPS installation found: using ONNXRuntime backend with GPU.")
                    return 'onnxruntime', 'mps'
                else:
                    raise
            except:
                logging.info("No valid CUDA installation found: using OpenVINO backend with CPU.")
                return 'openvino', 'cpu'
    else:
        return backend.lower(), device.lower()
