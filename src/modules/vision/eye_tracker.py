import os
import random
import time

import cv2
import asyncio
import concurrent.futures

import face_recognition
import numpy as np
from collections import deque
from typing import List, Tuple, Optional

from src.modules.vision.base_tracker import BaseTracker
# from src.modules.vision.face_recognizer import SimpleFaceRecognizer, FaceRecognizer
from src.modules.vision.motion_detector import MotionDetector
from src.modules.vision.eye_controller import MechanicalEyes


# Основной класс EyeTracker
class EyeTracker(BaseTracker):
    def __init__(self, context):
        self.eyes_control = MechanicalEyes()
        self.context = context
        self.camera_id = 0
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
        # self.face_recognizer: Optional[FaceRecognizer] = None
        self.motion_detector = MotionDetector(scale_factor=0.5, min_area=500)
        self.is_sleeping = False

    async def initialize(self):
        print("Initializing EyeTracker...")
        self.camera = cv2.VideoCapture(self.camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        # known_face_encodings, known_face_names = self._load_known_faces()
        # self.face_recognizer = SimpleFaceRecognizer(known_face_encodings, known_face_names)
        print("EyeTracker initialized.")

    def _load_known_faces(self) -> Tuple[List[np.ndarray], List[str]]:
        known_faces_dir = "known_faces/"
        os.makedirs(known_faces_dir, exist_ok=True)

        encodings = []
        names = []

        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(known_faces_dir, filename)
                name = os.path.splitext(filename)[0]

                print(f"Processing {filename}")

                encoding = self._get_face_encoding(image_path)
                if encoding is not None:
                    encodings.append(encoding)
                    names.append(name)
                else:
                    print(f"Failed to encode face in {filename}")

        print(f"Loaded {len(names)} known faces.")
        return encodings, names

    @staticmethod
    def _get_face_encoding(image_path: str) -> Optional[np.ndarray]:
        try:
            # Загрузка изображения с помощью OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None

            # Уменьшение размера изображения, если оно слишком большое
            max_size = 1024
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)

            # Конвертация в RGB (face_recognition ожидает RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Поиск лиц на изображении
            face_locations = face_recognition.face_locations(rgb_image)

            if not face_locations:
                print(f"No faces found in {image_path}")
                return None

            # Кодирование лица
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if face_encodings:
                return face_encodings[0]
            else:
                print(f"Failed to encode face in {image_path}")
                return None

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    async def track(self):
        while True:
            frame = await self._capture_frame()
            if frame is None:
                print("Failed to capture frame. Exiting.")
                break

            if not self.is_sleeping and self.context.request() != "sleep":
                await self._process_frame(frame)
                self._display_frame(frame)
            else:
                self._display_sleeping_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                await self.toggle_sleep_mode()

            await asyncio.sleep(0.01)

    async def toggle_sleep_mode(self):
        self.is_sleeping = not self.is_sleeping
        if self.is_sleeping:
            print("Entering sleep mode...")
        else:
            print("Waking up from sleep mode...")

    @staticmethod
    def _display_sleeping_frame(frame: np.ndarray):
        height, width = frame.shape[:2]
        sleeping_text = "SLEEPING (Press '1' to wake up)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White color
        thickness = 2
        text_size = cv2.getTextSize(sleeping_text, font, font_scale, thickness)[0]

        # Calculate position to center the text
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        # Create a dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)

        # Add text to the overlay
        cv2.putText(overlay, sleeping_text, (text_x, text_y), font, font_scale, font_color, thickness)

        # Blend the overlay with the original frame
        alpha = 0.7  # Transparency factor
        frame_with_overlay = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow('Mechanical Eye Tracker', frame_with_overlay)

    async def _capture_frame(self) -> Optional[np.ndarray]:
        if self.camera is None:
            raise ValueError("Camera is not initialized. Call initialize() first.")
        ret, frame = await asyncio.get_event_loop().run_in_executor(None, self.camera.read)
        return cv2.flip(frame, 1) if ret else None

    async def _process_frame(self, frame: np.ndarray):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            # image: cv2.typing.MatLike, scaleFactor: float = ..., minNeighbors: int = ..., flags: int = ..., minSize: cv2.typing.Size = ..., maxSize: cv2.typing.Size = ...
            self.face_cascade.detectMultiScale,
            gray_frame, 1.3, 7, 0, (80, 80)
        )

        self.face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in face_locations]
        if random.randint(1, 25) == 5:
            self.eyes_control.close_eyes()
            await asyncio.sleep(0.1)
            self.eyes_control.open_eyes()

        if self.face_locations:
            largest_face = max(self.face_locations, key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
            self._update_focus_point(largest_face)

            self.last_recognition_time = time.time()

            # current_time = time.time()
            # if current_time - self.last_recognition_time >= self.recognition_interval:
            #     self.current_speaker = await self.face_recognizer.recognize(frame, [largest_face])
            #     self.last_recognition_time = current_time

            self._draw_face_info(frame, largest_face)
        elif time.time() - self.last_recognition_time >= 3:
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

        x, y = self.focus_point
        x = self.eyes_control.px_to_angle(x, 40, 1240, 50, 130)
        y = self.eyes_control.px_to_angle(720 - y, 40, 680, 70, 110)
        self.eyes_control.send_data([x, y])

    def _draw_motion_info(self, frame: np.ndarray):
        x, y = self.focus_point
        cv2.circle(frame, (x, y), 7, (255, 0, 0), -1)
        cv2.putText(frame, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        x = self.eyes_control.px_to_angle(x, 0, 1280, 50, 130)
        y = self.eyes_control.px_to_angle(720 - y, 0, 720, 70, 110)
        self.eyes_control.send_data([x, y])

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
