'''
Example:

import cv2

from rtmlib import Hand, draw_skeleton

device = 'cuda'
backend = 'onnxruntime'  # opencv, onnxruntime

cap = cv2.VideoCapture('./demo.mp4')

openpose_skeleton = True  # True for openpose-style, False for mmpose-style

hand = Hand(to_openpose=openpose_skeleton,
            backend=backend,
            device=device)

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_idx += 1

    if not success:
        break

    keypoints, scores = hand(frame)

    img_show = frame.copy()

    img_show = draw_skeleton(img_show,
                             keypoints,
                             scores,
                             openpose_skeleton=openpose_skeleton,
                             kpt_thr=0.43)

    img_show = cv2.resize(img_show, (960, 540))
    cv2.imshow('img', img_show)
    cv2.waitKey(10)

'''
import numpy as np

from .. import RTMDet, RTMPose


class Hand:
    MODE = {
        'lightweight': {
            'det':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',  # noqa
            'det_input_size': (320, 320),
            'pose':
            'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',  # noqa
            'pose_input_size': (256, 256),
        }
    }

    def __init__(self,
                 det: str = None,
                 det_input_size: tuple = (320, 320),
                 pose: str = None,
                 pose_input_size: tuple = (256, 256),
                 mode: str = 'lightweight',
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        print('hand', backend, device)
        assert mode == 'lightweight', (
            'Currently only support lightweight mode.')

        if det is None:
            det = self.MODE[mode]['det']
            det_input_size = self.MODE[mode]['det_input_size']

        if pose is None:
            pose = self.MODE[mode]['pose']
            pose_input_size = self.MODE[mode]['pose_input_size']

        self.det_model = RTMDet(det,
                                model_input_size=det_input_size,
                                backend=backend,
                                device=device)
        self.pose_model = RTMPose(pose,
                                  model_input_size=pose_input_size,
                                  to_openpose=to_openpose,
                                  backend=backend,
                                  device=device)

    def __call__(self, image: np.ndarray):
        bboxes = self.det_model(image)
        keypoints, scores = self.pose_model(image, bboxes=bboxes)

        return keypoints, scores
