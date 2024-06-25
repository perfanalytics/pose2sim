from typing import List, Tuple

import cv2
import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose


class RTMO(BaseTool):

    def __init__(self,
                 onnx_model: str,
                 model_input_size: tuple = (640, 640),
                 mean: tuple = None,
                 std: tuple = None,
                 to_openpose: bool = False,
                 backend: str = 'onnxruntime',
                 device: str = 'cpu'):
        super().__init__(onnx_model, model_input_size, mean, std, backend,
                         device)
        self.to_openpose = to_openpose

    def __call__(self, image: np.ndarray):
        image, ratio = self.preprocess(image)
        outputs = self.inference(image)

        keypoints, scores = self.postprocess(outputs, ratio)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMO model inference.

        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.

        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        det_outputs, pose_outputs = outputs

        # onnx contains nms module
        pack_dets = (det_outputs[0, :, :4], det_outputs[0, :, 4])
        final_boxes, final_scores = pack_dets
        final_boxes /= ratio
        isscore = final_scores > 0.3
        isbbox = [i for i in isscore]
        # final_boxes = final_boxes[isbbox]

        # decode pose outputs
        keypoints, scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
        keypoints = keypoints / ratio

        keypoints = keypoints[isbbox]
        scores = scores[isbbox]

        return keypoints, scores
