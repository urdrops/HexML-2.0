import cv2
import numpy as np


class MotionDetector:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.min_area = 500

    def detect_motion(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_point = (0, 0)
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_point = (x + w // 2, y + h // 2)
                break

        cv2.circle(frame, motion_point, 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Motion: {motion_point}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame, motion_point


def main():
    cap = cv2.VideoCapture(2)
    detector = MotionDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, motion_point = detector.detect_motion(frame)

        cv2.imshow('Background Subtraction Motion Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()