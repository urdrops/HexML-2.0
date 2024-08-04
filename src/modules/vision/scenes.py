import cv2
import numpy as np
import os

# Параметры для настройки чувствительности движения
motion_sensitivity = 1000  # Чем меньше значение, тем выше чувствительность

# Инициализация детектора лиц и рта
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_cascade_path = cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml'

if not os.path.exists(mouth_cascade_path):
    raise FileNotFoundError(f'Классификатор для рта не найден по пути: {mouth_cascade_path}')

mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# Инициализация видео захвата
cap = cv2.VideoCapture(0)

# Установка максимального разрешения
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Получение фактического разрешения
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {width} x {height}")

ret, frame1 = cap.read()
ret, frame2 = cap.read()
previous_mouth = None

while cap.isOpened():
    # Разница между текущим и предыдущим кадром для детекции движения
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Инициализация переменных для хранения самого значимого движения
    max_contour = None
    max_area = 0

    # Поиск самого значимого контура
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < motion_sensitivity:
            continue
        if area > max_area:
            max_area = area
            max_contour = contour

    # Отображение самого значимого контура
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Детекция лиц
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Поиск наибольшего лица
    largest_face = None
    max_face_area = 0
    for (x, y, w, h) in faces:
        face_area = w * h
        if face_area > max_face_area:
            max_face_area = face_area
            largest_face = (x, y, w, h)

    # Отображение наибольшего лица и детекция движения рта
    if largest_face is not None:
        x, y, w, h = largest_face
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Поиск рта
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame1[y:y + h, x:x + w]
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(mouths) > 0:
            mouth = max(mouths, key=lambda rect: rect[2] * rect[3])  # Находим самый большой рот
            mx, my, mw, mh = mouth
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 255, 255), 2)

            # Проверка движения рта
            current_mouth = roi_gray[my:my + mh, mx:mx + mw]
            if previous_mouth is not None:
                mouth_diff = cv2.absdiff(previous_mouth, current_mouth)
                _, mouth_thresh = cv2.threshold(mouth_diff, 20, 255, cv2.THRESH_BINARY)
                motion = np.sum(mouth_thresh)

                if motion > motion_sensitivity / 10:
                    cv2.putText(frame1, "Он говорит", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            previous_mouth = current_mouth.copy()

    # Отображение результата
    cv2.imshow('Significant Motion and Largest Face Detection with Speaking Detection', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(10) == 27:  # Нажмите 'ESC' для выхода
        break

cap.release()
cv2.destroyAllWindows()
