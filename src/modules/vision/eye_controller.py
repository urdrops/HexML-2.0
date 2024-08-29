import numpy as np
import serial


class MechanicalEyes:
    # top left, bottom left, top right, bottom right
    OPEN_EYES: list[int] = [70, 100, 100, 70]
    CLOSE_EYES: list[int] = [130, 50, 50, 130]

    def __init__(self):
        self.prev_angles_array = []
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Connected!")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def send_data(self, angles_array: list[int]) -> None:
        if len(angles_array) != len(self.prev_angles_array):
            self.prev_angles_array = np.zeros(len(angles_array))
        np_arr1 = np.array(angles_array)
        np_arr2 = np.array(self.prev_angles_array)
        differences = np.abs(np_arr1 - np_arr2)
        if np.any(differences >= 1):
            # Преобразуем список углов в строку, разделенную запятыми
            data_string = ','.join(str(angle) for angle in angles_array) + '\n'
            try:
                self.ser.write(data_string.encode())
                print("Data sent")
                self.prev_angles_array = angles_array
            except serial.SerialException as e:
                print(f"Error sending data: {e}")
        else:
            print("skip data")
            return

    @staticmethod
    def px_to_angle(x, min_in, max_in, min_out, max_out):
        # px to angles
        return round(min_out + (x - min_in) * (max_out - min_out) / (max_in - min_in))

    def close(self):
        if self.ser.is_open:
            self.ser.close()
            print("Serial port closed")

    def open_eyes(self):
        self.send_data(self.OPEN_EYES)

    def close_eyes(self):
        self.send_data(self.CLOSE_EYES)
