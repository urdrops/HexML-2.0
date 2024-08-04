import cv2
import face_recognition
import threading

import numpy

# Загрузка известных лиц и их имен
known_face_encodings = []
known_face_names = []

# Пример добавления известных лиц
image_of_person_1 = face_recognition.load_image_file("urdrops.jpg")
person_1_encoding = face_recognition.face_encodings(image_of_person_1)[0]
known_face_encodings.append(person_1_encoding)
known_face_names.append("Person 1")

# Захват видео с камеры
video_capture = cv2.VideoCapture(0)
# Уменьшаем разрешение видео
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1020)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Глобальные переменные для хранения данных кадров и распознанных лиц
frame = None
face_locations = []
face_encodings = []
face_names = []

# Флаг для остановки потока
stop_thread = False

# Объект блокировки для синхронизации доступа к разделяемым данным
lock = threading.Lock()


def process_frame():
    global frame, face_locations, face_encodings, face_names

    while not stop_thread:
        if frame is not None:
            # Преобразование изображения из BGR (OpenCV) в RGB (face_recognition)
            rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

            # Найти все лица и их энкодинги в текущем кадре видео
            current_face_locations = face_recognition.face_locations(rgb_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_frame, current_face_locations)

            current_face_names = []
            for face_encoding in current_face_encodings:
                # Сравнение лица с известными лицами
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Неизвестный"

                # Если нашлось совпадение с известным лицом, берём имя
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                current_face_names.append(name)

            # Синхронизация доступа к разделяемым данным
            with lock:
                face_locations = current_face_locations
                face_encodings = current_face_encodings
                face_names = current_face_names


# Поток для обработки кадров
thread = threading.Thread(target=process_frame)
thread.start()
try:
    while True:
        # Захват одного кадра видео
        ret, frame = video_capture.read()

        # Синхронизация доступа к разделяемым данным
        with lock:
            display_face_locations = face_locations.copy()
            display_face_names = face_names.copy()

        # Отображение результатов
        for (top, right, bottom, left), name in zip(display_face_locations, display_face_names):
            # Рисуем прямоугольник вокруг лица
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Рисуем метку с именем под лицом
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Показать изображение
        cv2.imshow('Video', frame)

        # Нажмите 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_thread = True
            break

    # Ожидание завершения потока
    thread.join()
except KeyboardInterrupt:
    # Освободить захват видео
    video_capture.release()
    cv2.destroyAllWindows()
