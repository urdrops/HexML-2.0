import asyncio
from abc import ABC, abstractmethod
import face_recognition
import numpy as np
import time
from typing import List, Tuple, Dict


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
