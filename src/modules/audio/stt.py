from abc import ABC, abstractmethod
from openai import OpenAI


class BaseSTT(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        pass


class WhisperSTT(BaseSTT):
    async def transcribe(self, audio_data: bytes) -> str:
        audio_file = open("/path/to/file/german.mp3", "rb")
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file
        )
        print(translation.text)
        pass


class UzbekVoiceSTT(BaseSTT):
    async def transcribe(self, file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcription.text



