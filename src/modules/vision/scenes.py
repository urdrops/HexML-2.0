import json
import time

import playsound

from src.modules.vision.eye_controller import MechanicalEyes
eyes = MechanicalEyes()


def wakeup():
    eyes.open_eyes()
    time.sleep(0.1)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    eyes.send_data([130, 90])
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    eyes.send_data([50, 90])
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.2)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.2)
    eyes.send_data([90, 90])


def draw_scenario():
    eyes.open_eyes()
    time.sleep(0.1)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    eyes.send_data([130, 90])
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    eyes.send_data([50, 90])
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.5)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.2)
    eyes.close_eyes()
    time.sleep(0.1)
    eyes.open_eyes()
    time.sleep(0.2)
    eyes.send_data([90, 90])
    playsound.playsound("core/audio_files/draw1.mp3")
    time.sleep(1)
    playsound.playsound("core/audio_files/draw2.mp3")

    return json.dumps({"result": "draw finished"})
