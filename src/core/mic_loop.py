import os
import wave
import pvcobra
import pvporcupine
from pvrecorder import PvRecorder
from dotenv import load_dotenv

load_dotenv()
ACCESS_KEY = os.getenv("YOUR_PICOVOICE_ACCESS_KEY")


class AudioRecorder:
    def __init__(self):
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=['pv_files/arif.ppn'],
            model_path='pv_files/porcupine_params_ar.pv'
        )
        self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
        self.cobra = pvcobra.create(access_key=ACCESS_KEY)

        self.channels = 1
        self.output_file = "audio_files/recording.wav"
        self.sample_rate = self.porcupine.sample_rate

        self.STOP_THRESHOLD = 0.9
        self.SILENCE_DURATION = 1   # Лимит тишины в сек
        self.MAX_RECORDING_TIME = 30  # Максимальное время записи в секундах

    def _save_audio(self, frames):
        print("Сохранение данных в .wav файл...")
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frame.tobytes() for frame in frames))
        print("Файл сохранен как:", self.output_file)

    def _cleanup(self):
        if hasattr(self, 'recorder'):
            self.recorder.delete()
        if hasattr(self, 'cobra'):
            self.cobra.delete()
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
