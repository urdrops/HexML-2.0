from modules.vision.attention import Camera, FaceDetector, MotionDetector, VideoProcessor
import face_recognition
from modules.vision.face_recognizer import FaceRecognizer


def main() -> None:
    camera = Camera()
    detectors = [FaceDetector(), MotionDetector()]

    # Здесь нужно добавить известные лица
    known_faces = {
        "Person1": face_recognition.face_encodings(face_recognition.load_image_file("person1.jpg"))[0],
        "Person2": face_recognition.face_encodings(face_recognition.load_image_file("person2.jpg"))[0],
        # Добавьте больше известных лиц по необходимости
    }

    recognizer = FaceRecognizer(known_faces)
    video_processor = VideoProcessor(camera, detectors, recognizer)
    video_processor.run()


if __name__ == "__main__":
    main()
