import argparse
import os
import time
import io
import threading
import queue
import tempfile
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from abc import ABC, abstractmethod
import azure.cognitiveservices.speech as speechsdk


class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        pass


load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_TOKEN)

# Очередь для передачи аудио данных между потоками
audio_queue = queue.Queue()


class WhisperTTS(BaseTTS):
    def __init__(self, client: OpenAI):
        self.client = client

    @staticmethod
    def play_audio():
        buffer = io.BytesIO()
        not_first_execution = 0
        while True:
            try:
                chunk = audio_queue.get(timeout=1)
                if chunk is None:  # Сигнал о завершении
                    print("Speach finished")
                    break
                buffer.write(chunk)

                # Если накоплено достаточно данных, начинаем воспроизведение
                print("Buffer: ", buffer.tell())
                if buffer.tell() > 64000 or not_first_execution:  # Примерно 1 секунда аудио
                    buffer.seek(0)
                    # Используем временный файл для обхода проблем с форматом
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                        temp_file.write(buffer.getvalue())
                        temp_file_path = temp_file.name

                    # Воспроизводим аудио
                    audio = AudioSegment.from_mp3(temp_file_path)
                    print("Play sound time:", time.perf_counter())
                    play(audio)

                    # Удаляем временный файл
                    os.unlink(temp_file_path)
                    not_first_execution = 1
                    # Очищаем буфер и начинаем накапливать снова
                    buffer = io.BytesIO()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка при воспроизведении: {e}")

    def synthesize(self, text: str) -> bytes:
        # Запускаем поток воспроизведения
        play_thread = threading.Thread(target=self.play_audio)
        play_thread.start()

        try:
            with client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="echo",
                    input=text,
            ) as response:
                with open("audio_files/output.mp3", mode="wb") as f:
                    for data in response.iter_bytes(chunk_size=64000):
                        f.write(data)
                        audio_queue.put(data)
        except Exception as e:
            print(f"Ошибка при получении аудио: {e}")

        # Сигнализируем о завершении потока воспроизведения
        audio_queue.put(None)
        play_thread.join()


class AzureTTS:
    def __init__(self):
        self.speech_key = os.getenv('SPEECH_KEY')
        self.speech_region = os.getenv('SPEECH_REGION')
        self.args = self.parse_arguments()
        self.voice_name = self.args.voice
        self.speech_config = self._create_speech_config()
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config
        )

        self._warm_up()

    def _create_speech_config(self) -> speechsdk.SpeechConfig:
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        speech_config.speech_synthesis_voice_name = self.voice_name
        return speech_config

    def _warm_up(self):
        """Выполняет 'прогрев' системы, синтезируя короткую фразу."""
        warm_up_text = "Asistent yoqildi"
        warm_up_ssml = self._create_ssml(warm_up_text, self.args.rate, self.args.pitch)
        self.speech_synthesizer.speak_ssml_async(warm_up_ssml).get()
        print("Система инициализирована и готова к быстрой работе.")

    def synthesize(self, text: str) -> speechsdk.SpeechSynthesisResult:
        ssml = self._create_ssml(text, self.args.rate, self.args.pitch)
        speech_synthesizer = self._get_speech_synthesizer(self.args.output)

        result = speech_synthesizer.speak_ssml_async(ssml).get()
        self._handle_synthesis_result(result, text, self.args.output)
        return result

    def _create_ssml(self, text: str, rate: str, pitch: str) -> str:
        return f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="uz-UZ">
          <voice name="{self.speech_config.speech_synthesis_voice_name}">
            <prosody rate="{rate}" pitch="{pitch}">
              {text}
            </prosody>
          </voice>
        </speak>
        """

    def _get_speech_synthesizer(self, output_file: Optional[str]) -> speechsdk.SpeechSynthesizer:
        if output_file:
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)
            return speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
        return self.speech_synthesizer

    @staticmethod
    def _handle_synthesis_result(result: speechsdk.SpeechSynthesisResult, text: str, output_file: Optional[str]):
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesized for text: [{text}]")
            if output_file:
                print(f"Audio saved to {output_file}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")

    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Test Azure Text-to-Speech synthesis")
        parser.add_argument("--voice", default="uz-UZ-SardorNeural", help="Voice name for synthesis")
        parser.add_argument("--rate", default="1.18", help="Speech rate")
        parser.add_argument("--pitch", default="0%", help="Speech pitch")
        parser.add_argument("--output", help="Output audio_files file path")
        return parser.parse_args()
