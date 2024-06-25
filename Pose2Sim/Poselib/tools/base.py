import os
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np

from .file import download_checkpoint
def check_mps_support():
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return 'MPSExecutionProvider' in providers or 'CoreMLExecutionProvider' in providers
    except ImportError:
        return False

RTMLIB_SETTINGS = {
    'opencv': {
        'cpu': (cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU),

        # You need to manually build OpenCV through cmake
        'cuda': (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    },
    'onnxruntime': {
        'cpu': 'CPUExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'mps': 'CoreMLExecutionProvider' if check_mps_support() else 'CPUExecutionProvider'
    },
}


class BaseTool(metaclass=ABCMeta):

    def __init__(self,
                 onnx_model: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 backend: str = 'opencv',
                 device: str = 'cpu'):

        if not os.path.exists(onnx_model):
            onnx_model = download_checkpoint(onnx_model)

        if backend == 'opencv':
            try:
                providers = RTMLIB_SETTINGS[backend][device]

                session = cv2.dnn.readNetFromONNX(onnx_model)
                session.setPreferableBackend(providers[0])
                session.setPreferableTarget(providers[1])
                self.session = session
            except Exception:
                raise RuntimeError(
                    'This model is not supported by OpenCV'
                    ' backend, please use `pip install'
                    ' onnxruntime` or `pip install'
                    ' onnxruntime-gpu` to install onnxruntime'
                    ' backend. Then specify `backend=onnxruntime`.')  # noqa

        elif backend == 'onnxruntime':
            import onnxruntime as ort
            providers = RTMLIB_SETTINGS[backend][device]

            self.session = ort.InferenceSession(path_or_bytes=onnx_model,
                                                providers=[providers])

        elif backend == 'openvino':
            from openvino.runtime import Core
            core = Core()
            model_onnx = core.read_model(model=onnx_model)

            if device != 'cpu':
                print('OpenVINO only supports CPU backend, automatically'
                      ' switched to CPU backend.')

            self.compiled_model = core.compile_model(
                model=model_onnx,
                device_name='CPU',
                config={'PERFORMANCE_HINT': 'LATENCY'})
            self.input_layer = self.compiled_model.input(0)
            self.output_layer0 = self.compiled_model.output(0)
            self.output_layer1 = self.compiled_model.output(1)

        else:
            raise NotImplementedError

        print(f'load {onnx_model} with {backend} backend')

        self.onnx_model = onnx_model
        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img: np.ndarray):
        """Inference model.

        Args:
            img (np.ndarray): Input image in shape.

        Returns:
            outputs (np.ndarray): Output of RTMPose model.
        """
        # build input to (1, 3, H, W)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img[None, :, :, :]

        # run model
        if self.backend == 'opencv':
            outNames = self.session.getUnconnectedOutLayersNames()
            self.session.setInput(input)
            outputs = self.session.forward(outNames)
        elif self.backend == 'onnxruntime':
            sess_input = {self.session.get_inputs()[0].name: input}
            sess_output = []
            for out in self.session.get_outputs():
                sess_output.append(out.name)

            outputs = self.session.run(sess_output, sess_input)
        elif self.backend == 'openvino':
            results = self.compiled_model(input)
            output0 = results[self.output_layer0]
            output1 = results[self.output_layer1]
            outputs = [output0, output1]

        return outputs
