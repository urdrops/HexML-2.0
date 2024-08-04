import cv2
import numpy as np
from typing import List, Tuple, Optional
from face_recognizer import FaceRecognizer


class Camera:
    def __init__(self, width: int = 1280, height: int = 720):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.video_capture.read()
        return cv2.flip(frame, 1) if ret else None

    def release(self) -> None:
        self.video_capture.release()


class FrameProcessor:
    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class FaceDetector(FrameProcessor):
    def __init__(self, cascade_path: str = "haarcascade_frontalface_default.xml"):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray_frame = self.preprocess_frame(frame)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))
        return faces[0] if len(faces) > 0 else None

    @staticmethod
    def draw(frame: np.ndarray, detection: Optional[Tuple[int, int, int, int]]) -> Tuple[
        np.ndarray, Optional[Tuple[int, int]]]:
        center = None
        if detection is not None:
            x, y, w, h = detection
            if 100 < w < 400 and 100 < h < 400:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                cv2.putText(frame, f"Face: {x}, {y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame, center


class MotionDetector(FrameProcessor):
    def __init__(self):
        self.previous_frame: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray_frame = cv2.GaussianBlur(self.preprocess_frame(frame), (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return None

        frame_delta = cv2.absdiff(self.previous_frame, gray_frame)
        self.previous_frame = gray_frame

        thresh = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
        return cv2.boundingRect(max(significant_contours, key=cv2.contourArea)) if significant_contours else None

    @staticmethod
    def draw(frame: np.ndarray, detection: Optional[Tuple[int, int, int, int]]) -> Tuple[
        np.ndarray, Optional[Tuple[int, int]]]:
        center = None
        if detection is not None:
            x, y, w, h = detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Motion: {x}, {y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame, center


class VideoProcessor:
    def __init__(self, camera: Camera, detectors: List[FrameProcessor], recognizer: FaceRecognizer):
        self.camera = camera
        self.detectors = detectors
        self.recognizer = recognizer

    def process_frame(self, frame: np.ndarray) -> Tuple[
        np.ndarray, List[Optional[Tuple[int, int]]], List[Tuple[str, Tuple[int, int, int, int]]]]:
        centers = []
        for detector in self.detectors:
            detection = detector.detect(frame)
            frame, center = detector.draw(frame, detection)
            centers.append(center)

        recognitions = self.recognizer.recognize(frame)
        frame = self.recognizer.draw(frame, recognitions)

        return frame, centers, recognitions

    def run(self) -> None:
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                break

            processed_frame, centers, recognitions = self.process_frame(frame)

            face_center, motion_center = centers
            if face_center:
                print(f"Face center: {face_center}")
                face_detector = self.detectors[0]
                if isinstance(face_detector, FaceDetector):
                    print(f"Mouth moving: {face_detector.mouth_moving}")
            if motion_center:
                print(f"Motion center: {motion_center}")

            for name, _ in recognitions:
                print(f"Recognized: {name}")

            cv2.imshow('Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()
