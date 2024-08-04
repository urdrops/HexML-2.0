import cv2
import serial
import time


ser = serial.Serial('/dev/ttyACM0', 9600)

# Инициализация детектора лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# Установка максимального разрешения камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)


def send_angles(angles):
    # Преобразуем список углов в строку, разделенную запятыми
    data_string = ','.join(str(angle) for angle in angles) + '\n'
    ser.write(data_string.encode())


def map_value(x, A1, A2, B1, B2):
    return B1 + (x - A1) * (B2 - B1) / (A2 - A1)


def wakeup_scene():
    send_angles([90, 90, 100, 70, 80, 110])  # open
    time.sleep(0.5)
    send_angles([90, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([90, 90, 100, 70, 80, 110])  # open
    time.sleep(0.5)
    send_angles([90, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([90, 90, 100, 70, 80, 110])  # open
    time.sleep(0.1)
    send_angles([90, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([90, 90, 100, 70, 80, 110])  # open
    time.sleep(0.3)
    send_angles([120, 90, 100, 70, 80, 110])  # open
    time.sleep(0.1)
    send_angles([120, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([120, 90, 100, 70, 80, 110])  # open
    time.sleep(0.9)
    send_angles([120, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([50, 90, 100, 70, 80, 110])  # open
    time.sleep(0.2)
    send_angles([50, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([50, 90, 100, 70, 80, 110])  # open
    time.sleep(0.1)
    send_angles([50, 90, 60, 120, 120, 70])  # close
    time.sleep(0.1)
    send_angles([90, 90, 100, 70, 80, 110])  # open


try:
    wakeup_scene()
    while True:
        # Считывание кадра
        ret, frame = cap.read()
        if not ret:
            break
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"Текущее разрешение камеры: {int(width)}x{int(height)}")

        # Зеркальное отображение кадра
        frame = cv2.flip(frame, 1)

        # Преобразование в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Рисование квадрата вокруг лица
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Вычисление координат точки
            center_x = x + w // 2
            center_y = y + h // 3

            # Отображение точки
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Вывод координат
            center_x = round(map_value(center_x, 0, 1250, 40, 130))
            center_y = round(map_value(700 - center_y, 0, 680, 50, 100))
            send_angles([center_x, center_y, 100, 70, 80, 110])
            print(f"Центр точки: ({center_x}, {center_y})")

            if random.randint(0, 50) == 8:
                send_angles([center_x, center_y, 60, 120, 120, 70])  # close
                time.sleep(0.1)
                send_angles([center_x, center_y, 100, 70, 80, 110])  # open
                time.sleep(0.1)
                send_angles([center_x, center_y, 60, 120, 120, 70])  # close
                time.sleep(0.1)
                send_angles([center_x, center_y, 100, 70, 80, 110])  # open

        # Отображение результата
        cv2.imshow('HexML view', frame)

        # Выход при нажатии 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Освобождение ресурсовьи
    cap.release()
    cv2.destroyAllWindows()

except KeyboardInterrupt:
    send_angles([90, 90, 60, 120, 120, 70])
finally:
    send_angles([90, 90, 60, 120, 120, 70])