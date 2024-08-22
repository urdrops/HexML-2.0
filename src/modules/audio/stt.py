import os
import requests
from abc import ABC, abstractmethod
from openai import OpenAI

UZBEKVOICE_API_KEY = os.getenv("UZBEKVOICE_API_KEY")


class BaseSTT(ABC):
    @abstractmethod
    async def transcribe(self, file_path: str) -> str:
        pass


class WhisperSTT(BaseSTT):
    def __init__(self, client: OpenAI):
        self.client = client

    def transcribe(self, file_path: str) -> str:
        audio_file = open(file_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text


class UzbekVoiceSTT(BaseSTT):

    def transcribe(self, file_path: str) -> str:
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
