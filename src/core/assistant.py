from src.modules.audio.stt import BaseSTT, WhisperSTT, UzbekVoiceSTT
from src.modules.audio.tts import BaseTTS, ElevenLabsTTS, AzureTTS
from src.modules.intelligence.llm import BaseLLM, OpenAIGPT, GroqLLM


class Assistant:
    def __init__(self, stt: BaseSTT, llm: BaseLLM, tts: BaseTTS):
        self.stt = stt
        self.llm = llm
        self.tts = tts

    async def process_audio_input(self, audio_data: bytes) -> str:
        text = await self.stt.transcribe(audio_data)
        response = await self.llm.generate_response(text)
        audio_response = await self.tts.synthesize(response)
        return audio_response

    @classmethod
    def create(cls, stt_type: str, llm_type: str, tts_type: str) -> 'Assistant':
        stt = Assistant._create_stt(stt_type)
        llm = Assistant._create_llm(llm_type)
        tts = Assistant._create_tts(tts_type)
        return cls(stt, llm, tts)

    @staticmethod
    def _create_stt(stt_type: str) -> BaseSTT:
        if stt_type == "whisper":
            return WhisperSTT()
        elif stt_type == "azure":
            return UzbekVoiceSTT()
        else:
            raise ValueError(f"Unsupported STT type: {stt_type}")

    @staticmethod
    def _create_llm(llm_type: str) -> BaseLLM:
        if llm_type == "openai":
            return OpenAIGPT()
        elif llm_type == "groq":
            return GroqLLM()
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

    @staticmethod
    def _create_tts(tts_type: str) -> BaseTTS:
        if tts_type == "elevenlabs":
            return ElevenLabsTTS()
        elif tts_type == "azure":
            return AzureTTS()
        else:
            raise ValueError(f"Unsupported TTS type: {tts_type}")


# Example usage:
# assistant = Assistant.create(stt_type="whisper", llm_type="openai", tts_type="elevenlabs")
#
# assistant.process_audio_input(audio_data)
client = OpenAIGPT()
client.generate_response("salom!")