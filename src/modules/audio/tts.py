from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        pass


class ElevenLabsTTS(BaseTTS):
    async def synthesize(self, text: str) -> bytes:
        # Implement ElevenLabs TTS logic here
        pass


class AzureTTS(BaseTTS):
    async def synthesize(self, text: str) -> bytes:
        # Implement Azure TTS logic here
        pass