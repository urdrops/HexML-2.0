import os
import time
import io
import wave
import requests
import pyaudio
import azure.cognitiveservices.speech as speechsdk
from openai import OpenAI, AssistantEventHandler
from typing_extensions import override
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# API Keys and IDs
OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
UZBEKVOICE_API_KEY = os.getenv("UZBEKVOICE_API_KEY")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_TOKEN)
thread = openai_client.beta.threads.create()

# Initialize Azure Speech SDK
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
speech_config.speech_synthesis_voice_name = 'uz-UZ-SardorNeural'
audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)


class VoiceAssistant:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=CHUNK)

    def record_audio(self):
        print("* Recording...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = self.stream.read(CHUNK)
            frames.append(data)
        print("* Recording finished")
        return frames

    def speech_to_text(self, frames):
        self.start = time.perf_counter()
        print("* Converting speech to text...")
        audio_data = io.BytesIO()
        wf = wave.open(audio_data, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        audio_data.seek(0)

        url = 'https://uzbekvoice.ai/api/v1/stt'
        headers = {"Authorization": UZBEKVOICE_API_KEY}
        files = {"file": ("audio_files.wav", audio_data)}
        data = {
            "return_offsets": "true",
            "run_diarization": "false",
            "language": "uz",
            "blocking": "true",
        }

        try:
            response = requests.post(url, headers=headers, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                text = result.get('result', {}).get('conversation_text', 'No result')
                print(f"Recognized text: {text}")
                return text
            else:
                print(f"Error in STT request: {response.status_code}: {response.text}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error sending STT request: {e}")
            return None

    def process_with_ai(self, text):
        print("* Processing with AI...")
        message = openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=text,
        )

        class EventHandler(AssistantEventHandler):
            def __init__(self):
                super().__init__()
                self.full_response = ""

            @override
            def on_text_created(self, text) -> None:
                print(f"\nassistant > ", end="", flush=True)

            @override
            def on_text_delta(self, delta, snapshot):
                print(delta.value, end="", flush=True)
                self.full_response += delta.value

        handler = EventHandler()

        with openai_client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=OPENAI_ASSISTANT_ID,
                event_handler=handler,
        ) as stream:
            stream.until_done()

        return handler.full_response

    def text_to_speech(self, text):
        print("* Converting text to speech...")
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="uz-UZ">
          <voice name="{speech_config.speech_synthesis_voice_name}">
            <prosody rate="1.15" pitch="0%">
              {text}
            </prosody>
          </voice>
        </speak>
        """
        self.end = time.perf_counter()
        print(f"\nTime taken before voice play: {self.end - self.start}\n")
        result = speech_synthesizer.speak_ssml_async(ssml).get()
        self.end = time.perf_counter()
        print(f"\nTime taken after voice play: {self.end - self.start}\n")

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized successfully")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled: {cancellation_details.reason}")

    def run(self):
        try:
            while True:
                input("Press Enter to start speaking...")
                audio_frames = self.record_audio()
                text = self.speech_to_text(audio_frames)
                if text:
                    ai_response = self.process_with_ai(text)
                    self.text_to_speech(ai_response)
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping the assistant...")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
