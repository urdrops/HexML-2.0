import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Dict
from attention import FrameProcessor


class FaceRecognizer(FrameProcessor):
    def __init__(self, known_faces: Dict[str, np.ndarray]):
        self.known_face_encodings = list(known_faces.values())
        self.known_face_names = list(known_faces.keys())

    def recognize(self, frame: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        # Уменьшаем размер кадра для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Находим все лица на кадре
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Сравниваем лицо с известными лицами
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            face_names.append(name)

        # Возвращаем к исходному масштабу
        return [(name, (top * 4, right * 4, bottom * 4, left * 4))
                for (top, right, bottom, left), name in zip(face_locations, face_names)]

    def draw(self, frame: np.ndarray, recognitions: List[Tuple[str, Tuple[int, int, int, int]]]) -> np.ndarray:
        for name, (top, right, bottom, left) in recognitions:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        return frame
