import cv2
import numpy as np
from typing import Optional, Tuple


# Класс для обнаружения движения
class MotionDetector:
    def __init__(self, scale_factor: float, min_area: int):
        self.scale_factor = scale_factor
        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        small_frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
        fg_mask = self.bg_subtractor.apply(small_frame)

        if self.prev_gray is None:
            self.prev_gray = small_frame
            return None

        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, small_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])

        motion_mask = ((magnitude > 1) & (fg_mask > 0)).astype(np.uint8) * 255

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > self.min_area:
                M = cv2.moments(max_contour)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"] / self.scale_factor), int(M["m01"] / M["m00"] / self.scale_factor)
                    self.prev_gray = small_frame
                    return cx, cy

        self.prev_gray = small_frame
        return None