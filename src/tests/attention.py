import time
import os
from collections import deque
from typing import List, Tuple, Optional, Dict
import face_recognition
import cv2
import asyncio
import concurrent.futures
import numpy as np
from abc import ABC, abstractmethod


# Абстрактный базовый класс для трекеров
class BaseTracker(ABC):
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def track(self):
        pass

    @abstractmethod
    async def shutdown(self):
        pass


# Интерфейс для распознавания лиц
class FaceRecognizer(ABC):
    @abstractmethod
    async def recognize(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> str:
        pass


# Класс для распознавания лиц
class SimpleFaceRecognizer(FaceRecognizer):
    def __init__(self, known_face_encodings: List[np.ndarray], known_face_names: List[str]):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.face_cache: Dict[bytes, Tuple[str, float]] = {}

    async def recognize(self, frame: np.ndarray, face_locations: List[Tuple[int, int, int, int]]) -> str:
        face_encodings = await asyncio.get_event_loop().run_in_executor(
            None, face_recognition.face_encodings, frame, face_locations
        )

        current_time = time.time()
        for face_encoding in face_encodings:
            cache_key = face_encoding.tobytes()
            cached_result = self.face_cache.get(cache_key)
            if cached_result and current_time - cached_result[1] < 10:
                return cached_result[0]

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            if any(matches):
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"

            self.face_cache[cache_key] = (name, current_time)
            return name

        return "Unknown"


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


# Основной класс EyeTracker
class EyeTracker(BaseTracker):
    def __init__(self):
        self.frame_size = (1280, 720)
        self.camera: Optional[cv2.VideoCapture] = None
        self.face_locations: List[Tuple[int, int, int, int]] = []
        self.current_speaker = "Unknown"
        self.focus_point = (0, 0)
        self.last_recognition_time = 0
        self.recognition_interval = 1.0
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.history = deque(maxlen=3)
        self.face_recognizer: Optional[FaceRecognizer] = None
        self.motion_detector = MotionDetector(scale_factor=0.5, min_area=500)

    async def initialize(self):
        print("Initializing EyeTracker...")
        self.camera = cv2.VideoCapture(2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        known_face_encodings, known_face_names = await self._load_known_faces()
        self.face_recognizer = SimpleFaceRecognizer(known_face_encodings, known_face_names)
        print("EyeTracker initialized.")

    async def _load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        known_faces_dir = "known_faces/"
        os.makedirs(known_faces_dir, exist_ok=True)

        async def load_face(filename):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(known_faces_dir, filename)
                encoding = await self._get_face_encoding(image_path)
                if encoding is not None:
                    return encoding, os.path.splitext(filename)[0]
            return None, None

        tasks = [load_face(filename) for filename in os.listdir(known_faces_dir)]
        results = await asyncio.gather(*tasks)

        encodings = []
        names = []
        for encoding, name in results:
            if encoding is not None:
                encodings.append(encoding)
                names.append(name)

        print(f"Loaded {len(names)} known faces.")
        return encodings, names

    @staticmethod
    async def _get_face_encoding(image_path: str) -> Optional[np.ndarray]:
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(None, face_recognition.load_image_file, image_path)
        encodings = await loop.run_in_executor(None, face_recognition.face_encodings, image)
        return encodings[0] if encodings else None

    async def track(self):
        while True:
            frame = await self._capture_frame()
            if frame is None:
                print("Failed to capture frame. Exiting.")
                break

            await self._process_frame(frame)
            print("proccess ended")
            self._display_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)

    async def _capture_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            raise ValueError("Camera is not initialized. Call initialize() first.")
        ret, frame = await asyncio.get_event_loop().run_in_executor(None, self.camera.read)
        return cv2.flip(frame, 1) if ret else None

    async def _process_frame(self, frame: np.ndarray):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self.face_cascade.detectMultiScale,
            gray_frame, 1.3, 7, 0, (50, 50)
        )

        self.face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]

        if self.face_locations:
            largest_face = max(self.face_locations, key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
            self._update_focus_point(largest_face)

            current_time = time.time()
            if current_time - self.last_recognition_time >= self.recognition_interval:
                self.current_speaker = await self.face_recognizer.recognize(frame, [largest_face])
                self.last_recognition_time = current_time

            self._draw_face_info(frame, largest_face)
        else:
            motion_point = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.motion_detector.detect,
                gray_frame
            )
            if motion_point:
                self.history.append(motion_point)
                self.focus_point = tuple(map(int, np.mean(self.history, axis=0)))
                self._draw_motion_info(frame)

    def _update_focus_point(self, face_location: Tuple[int, int, int, int]):
        top, right, bottom, left = face_location
        new_x = (left + right) >> 1
        new_y = top + ((bottom - top) // 3)

        dx = new_x - self.focus_point[0]
        dy = new_y - self.focus_point[1]

        if dx * dx + dy * dy < 10000:
            smoothing_factor = 0.4
            self.focus_point = (
                int(self.focus_point[0] + dx * smoothing_factor),
                int(self.focus_point[1] + dy * smoothing_factor)
            )
        else:
            self.focus_point = (new_x, new_y)

    def _draw_face_info(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]):
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.circle(frame, self.focus_point, 5, (0, 255, 0), -1)

        info_text = f"Person: {self.current_speaker} | Focus: {self.focus_point}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def _draw_motion_info(self, frame: np.ndarray):
        x, y = self.focus_point
        cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    @staticmethod
    def _display_frame(frame: np.ndarray):
        cv2.imshow('Mechanical Eye Tracker', frame)

    async def shutdown(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        self.executor.shutdown()
        print("EyeTracker shut down.")

    def get_focus_coordinates(self) -> Tuple[int, int]:
        return self.focus_point

    def get_current_speaker(self) -> str:
        return self.current_speaker
