import cv2
import numpy as np


class RetinaProcessor:
    def __init__(self, neuron_network):
        self.network = neuron_network

    def process_image(self, frame):
        """
        이미지를 뉴런 신호로 변환 (망막 모방).
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flattened = gray_frame.flatten() / 255.0  # [0, 1]로 정규화
        signals = np.pad(flattened, (0, self.network.num_neurons - len(flattened)), 'constant')
        return signals

    def capture_webcam(self):
        """
        웹캠에서 이미지를 캡처하고 망막 처리.
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image from webcam.")
        return self.process_image(frame)
