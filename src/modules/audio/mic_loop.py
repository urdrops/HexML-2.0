import os
import time
import wave
import numpy as np
import pvcobra
import pvporcupine
import requests
from pvrecorder import PvRecorder
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ACCESS_KEY = os.getenv("YOUR_PICOVOICE_ACCESS_KEY")
OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
UZBEKVOICE_API_KEY = os.getenv("UZBEKVOICE_API_KEY")

client = OpenAI(api_key=OPENAI_TOKEN)


class TTSModel:
    @staticmethod
    def whisper_voice(result_text: str):
        with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="echo",
                input=result_text,
        ) as response:
            with open("output.mp3", mode="wb") as f:
                start = time.perf_counter()
                for data in response.iter_bytes(chunk_size=32000):
                    print(data)
                    f.write(data)
                    print("chunk write time: ", time.perf_counter() - start)
        print("time: ", time.perf_counter() - start)

    def azure_voice(self):
        pass


class STTModel:
    @staticmethod
    def whisper_stt(file_path):
        audio_file = open(file_path, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text

    @staticmethod
    def mohirai_stt(file_path):
        url = 'https://uzbekvoice.ai/api/v1/stt'
        headers = {
            "Authorization": UZBEKVOICE_API_KEY
        }
        files = {
            "file": open(file_path, "rb"),
        }
        data = {
            "return_offsets": "true",
            "run_diarization": "false",
            "language": "uz",
            "blocking": "true",
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            if response.status_code == 200:
                return response.json().get('result').get('conversation_text')
            else:
                return f"Request failed with status code {response.status_code}: {response.text}"
        except requests.exceptions.Timeout:
            return "Request timed out. The API response took too long to arrive."


class AudioRecorder:
    def __init__(self):
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=['arif.ppn'],
            model_path='porcupine_params_ar.pv'
        )
        self.recorder = PvRecorder(device_index=-1, frame_length=self.porcupine.frame_length)
        self.cobra = pvcobra.create(access_key=ACCESS_KEY)

        self.channels = 1
        self.output_file = "recording.wav"
        self.sample_rate = self.porcupine.sample_rate

        self.STOP_THRESHOLD = 0.6
        self.SILENCE_DURATION = 1   # Лимит тишины в сек
        self.MAX_RECORDING_TIME = 30  # Максимальное время записи в секундах

    def voice_recording(self):
        frames = []
        is_first_iter = 1
        is_recording = 0
        start_recording_time = time.perf_counter()
        last_activity_time = start_recording_time

        print("Начало записи...")

        while True:
            if is_recording:
                audio_frame = self.recorder.read()
                frames.append(np.array(audio_frame, dtype=np.int16))

                activity = self.cobra.process(audio_frame)
                current_time = time.perf_counter()

                if activity >= self.STOP_THRESHOLD:
                    # Refresh timer if voice still active
                    last_activity_time = current_time

                if current_time - last_activity_time > self.SILENCE_DURATION:
                    print("Запись остановлена (тишина)")
                    self._save_audio(frames)
                    start_stt_time = time.perf_counter()
                    print(STTModel.mohirai_stt(self.output_file))
                    print("Time: ", time.perf_counter() - start_stt_time, "sec")
                    is_recording = 0
            else:
                if time.perf_counter() - start_recording_time > 60:
                    print("session end.")
                    break
                # Инициализация фреймов при первой итерации
                if is_first_iter:
                    frames = [
                        np.array(self.recorder.read(), dtype=np.int16)
                        for _ in range(int(self.porcupine.sample_rate / self.porcupine.frame_length))
                    ]
                    is_first_iter = 0

                # Чтение новых фреймов и удаление старых
                audio_frame = self.recorder.read()
                frames.append(np.array(audio_frame, dtype=np.int16))
                frames.pop(0)

                # Проверка на детекцию голоса
                if self.cobra.process(audio_frame) >= self.STOP_THRESHOLD:
                    print("voice activity detected.")
                    last_activity_time = time.perf_counter()
                    is_recording = 1
                    is_first_iter = 1
                    print("Начало записи...")

    def _save_audio(self, frames):
        print("Сохранение данных в .wav файл...")
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frame.tobytes() for frame in frames))
        print("Файл сохранен как:", self.output_file)

    def run(self):
        self.recorder.start()
        try:
            print("Начало прослушивания... ")
            while True:
                audio_frame = self.recorder.read()
                keyword_index = self.porcupine.process(audio_frame)
                if keyword_index == 0:
                    print("Hello sir! How can i help you?")
                    self.voice_recording()
        except KeyboardInterrupt:
            print("Остановка...")
        finally:
            self._cleanup()

    def _cleanup(self):
        if hasattr(self, 'recorder'):
            self.recorder.delete()
        if hasattr(self, 'cobra'):
            self.cobra.delete()
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()


if __name__ == "__main__":
    audio = AudioRecorder()
    audio.run()
