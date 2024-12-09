import time
import numpy as np
from src.core.mic_loop import AudioRecorder
from src.modules.audio.stt import BaseSTT, WhisperSTT, UzbekVoiceSTT
from src.modules.audio.tts import BaseTTS, AzureTTS, WhisperTTS
from src.modules.intelligence.llm import BaseLLM, OpenAIGPT, GroqLLM, openai_client
import pygame.mixer as player
import playsound
from src.modules.vision.eye_controller import MechanicalEyes
from src.modules.vision.scenes import wakeup

eye = MechanicalEyes

player.init()
player.music.load("core/audio_files/think.mp3")


class Assistant(AudioRecorder):
    def __init__(self, stt: BaseSTT, llm: BaseLLM, tts: BaseTTS):
        super().__init__()
        self.stt = stt
        self.llm = llm
        self.tts = tts

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
                    #   =========================================
                    #   Send and Transcribe
                    #   =========================================
                    start_time = time.perf_counter()
                    # player.music.play()
                    print("Запись остановлена (тишина)")
                    self._save_audio(frames)
                    transcribed_text = self.stt.transcribe(self.output_file)
                    print(transcribed_text)
                    self.recorder.stop()
                    for chunk_response in self.llm.generate_response(transcribed_text):
                        print("Chunk sent: ", chunk_response)
                        # player.music.stop()
                        print("Full time: ", time.perf_counter() - start_time, "sec")
                        self.tts.synthesize(chunk_response)
                        print("play end")
                    self.recorder.start()
                    is_recording = 0
                    start_recording_time = time.perf_counter()
            else:
                if time.perf_counter() - start_recording_time > self.MAX_RECORDING_TIME:
                    print("Session end. Say Arif to speak again")
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

    def run(self, context, TalkState):
        self.recorder.start()
        try:
            print("Начало прослушивания... ")
            while True:
                audio_frame = self.recorder.read()
                keyword_index = self.porcupine.process(audio_frame)
                if keyword_index == 0:
                    wakeup()
                    context.set_state(TalkState())
                    playsound.playsound("core/audio_files/gretting.mp3")
                    print("Salom dostim! Qanday yordam kerak?")

                    self.voice_recording()

        except KeyboardInterrupt:
            print("Остановка...")
        finally:
            self._cleanup()

    @classmethod
    def create(cls, stt_type: str, llm_type: str, tts_type: str) -> 'Assistant':
        stt = Assistant._create_stt(stt_type)
        llm = Assistant._create_llm(llm_type)
        tts = Assistant._create_tts(tts_type)
        return cls(stt, llm, tts)

    @staticmethod
    def _create_stt(stt_type: str) -> BaseSTT:
        if stt_type == "whisper":
            return WhisperSTT(openai_client)
        elif stt_type == "mohirai":
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
        if tts_type == "whisper":
            return WhisperTTS(openai_client)
        elif tts_type == "azure":
            return AzureTTS()
        else:
            raise ValueError(f"Unsupported TTS type: {tts_type}")
