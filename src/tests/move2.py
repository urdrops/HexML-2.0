import serial


class MechanicalEyes:
    # top left, bottom left, top right, bottom right
    OPEN_EYES = [100, 70, 80, 110]
    CLOSE_EYES = [60, 120, 120, 70]

    def __init__(self):
        try:
            self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            print("Connected!")
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            raise

    def send_data(self, angles_array):
        # Преобразуем список углов в строку, разделенную запятыми
        data_string = ','.join(str(angle) for angle in angles_array) + '\n'
        try:
            self.ser.write(data_string.encode())
            print("Data sent")
        except serial.SerialException as e:
            print(f"Error sending data: {e}")

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


eyes = MechanicalEyes()
try:
    flag = ""
    while flag != "q":
        flag = input("Enter:")
        if flag == "1":
            eyes.open_eyes()
        elif flag == "2":
            eyes.close_eyes()

finally:
    eyes.close()