import os
import time
import io
import threading
import queue
import tempfile

from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play

load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_TOKEN)

# Очередь для передачи аудио данных между потоками
audio_queue = queue.Queue()


def play_audio():
    buffer = io.BytesIO()
    while True:
        try:
            chunk = audio_queue.get(timeout=1)
            if chunk is None:  # Сигнал о завершении
                break
            buffer.write(chunk)

            # Если накоплено достаточно данных, начинаем воспроизведение
            if buffer.tell() > 32000:  # Примерно 1 секунда аудио
                buffer.seek(0)
                # Используем временный файл для обхода проблем с форматом
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                    temp_file.write(buffer.getvalue())
                    temp_file_path = temp_file.name

                # Воспроизводим аудио
                audio = AudioSegment.from_mp3(temp_file_path)
                play(audio)

                # Удаляем временный файл
                os.unlink(temp_file_path)

                # Очищаем буфер и начинаем накапливать снова
                buffer = io.BytesIO()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Ошибка при воспроизведении: {e}")


# Запускаем поток воспроизведения
play_thread = threading.Thread(target=play_audio)
play_thread.start()

try:
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="echo",
            input="""Привет друг! что нового расскажешь? например какая погода сегодня прекрасная и идеальная""",
    ) as response:
        with open("output.mp3", mode="wb") as f:
            start = time.perf_counter()
            for data in response.iter_bytes(chunk_size=32000):
                f.write(data)
                audio_queue.put(data)
                print("chunk write time: ", time.perf_counter() - start)
except Exception as e:
    print(f"Ошибка при получении аудио: {e}")

# Сигнализируем о завершении потока воспроизведения
audio_queue.put(None)
play_thread.join()