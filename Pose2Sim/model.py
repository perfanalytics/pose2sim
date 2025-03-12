import logging
from enum import Enum
from pathlib import Path
from rtmlib import BodyWithFeet, Wholebody, Body, Hand
from anytree.importer import DictImporter
from Pose2Sim.model import PoseModel

class PoseModel(Enum):
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

    @classmethod
    def from_config(cls, pose_model_str: str) -> "PoseModel":
        """
        Transforme la chaîne de config en valeur canonique de l’énumération.
        Par exemple :
          - "BODY_WITH_FEET"  → "HALPE_26"
          - "WHOLE_BODY_WRIST" → "COCO_133_WRIST"
          - "WHOLE_BODY"      → "COCO_133"
          - "BODY"            → "COCO_17"
        """
        mapping = {
            'BODY_WITH_FEET': 'HALPE_26',
            'WHOLE_BODY_WRIST': 'COCO_133_WRIST',
            'WHOLE_BODY': 'COCO_133',
            'BODY': 'COCO_17',
            'HAND': 'HAND_21',
            'FACE': 'FACE_106',
            'ANIMAL': 'ANIMAL2D_17',
        }
        key = pose_model_str.upper()
        if key in mapping:
            key = mapping[key]
        try:
            return cls[key]
        except :
            raise ValueError(f'{pose_model_str}')

    def get_model_class(self):
        """
        Renvoie la classe associée au modèle de pose.
        Les classes (BodyWithFeet, Wholebody, Body, Hand, etc.) doivent être importées.
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

    def load_model_instance(self):
        """
        Tente de charger l'instance du modèle.
        D'abord, on essaye d'utiliser la classe associée (via eval),
        puis on essaie de l'importer via DictImporter en cas d'échec.
        """
        model_class = self.get_model_class()
        try:
            if model_class:
                return model_class, eval(self.name)
            else:
                raise NameError
        except Exception:
            try:
                pose_instance = DictImporter().import_(self.name)
                if pose_instance.id == 'None':
                    pose_instance.id = None
                return model_class, pose_instance
            except Exception:
                raise NameError(f"{self.name} not found in skeletons.py nor in Config.toml")
