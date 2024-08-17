import os
import argparse
import time
from typing import Optional

from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk


class TTSError(Exception):
    """Custom exception for TTS-related errors."""
    pass


class AzureTTSConfig:
    def __init__(self):
        load_dotenv()
        self.speech_key = os.getenv('SPEECH_KEY')
        self.speech_region = os.getenv('SPEECH_REGION')

        if not self.speech_key or not self.speech_region:
            raise TTSError("Speech key or region not found in .env file")


class TTSTester:
    def __init__(self, voice_name: str = 'uz-UZ-SardorNeural'):
        config = AzureTTSConfig()
        self.speech_config = self._create_speech_config(config, voice_name)
        self.audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        self.speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config
        )
        self.start_time = 0
        self.synthesis_start_time = 0

    @staticmethod
    def _create_speech_config(config: AzureTTSConfig, voice_name: str) -> speechsdk.SpeechConfig:
        speech_config = speechsdk.SpeechConfig(
            subscription=config.speech_key,
            region=config.speech_region
        )
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        speech_config.speech_synthesis_voice_name = voice_name
        return speech_config

    def synthesize_speech(self, text: str, rate: str = "0.9", pitch: str = "0%",
                          output_file: Optional[str] = None) -> speechsdk.SpeechSynthesisResult:
        self.start_time = time.perf_counter()
        ssml = self._create_ssml(text, rate, pitch)

        speech_synthesizer = self._get_speech_synthesizer(output_file)

        # Set up the callbacks
        speech_synthesizer.synthesis_started.connect(self._synthesis_started_cb)
        speech_synthesizer.synthesis_completed.connect(self._synthesis_completed_cb)
        result = speech_synthesizer.speak_ssml_async(ssml).get()

        self._handle_synthesis_result(result, text, output_file)
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

    def _synthesis_started_cb(self, event: speechsdk.SpeechSynthesisEventArgs):
        self.synthesis_start_time = time.perf_counter()
        processing_time = self.synthesis_start_time - self.start_time
        print(f"Processing time (request to synthesis start): {processing_time:.2f} seconds")

    def _synthesis_completed_cb(self, event: speechsdk.SpeechSynthesisEventArgs):
        synthesis_time = time.perf_counter() - self.synthesis_start_time
        print(f"Synthesis time: {synthesis_time:.2f} seconds")

    def _handle_synthesis_result(self, result: speechsdk.SpeechSynthesisResult, text: str, output_file: Optional[str]):
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            total_time = time.perf_counter() - self.start_time
            print(f"Total time (request to synthesis completion): {total_time:.2f} seconds")
            print(f"Speech synthesized for text: [{text}]")
            if output_file:
                print(f"Audio saved to {output_file}")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            raise TTSError(f"Speech synthesis canceled: {cancellation_details.reason}")


def main():
    parser = argparse.ArgumentParser(description="Test Azure Text-to-Speech synthesis")
    parser.add_argument("--voice", default="uz-UZ-SardorNeural", help="Voice name for synthesis")
    parser.add_argument("--rate", default="1.15", help="Speech rate")
    parser.add_argument("--pitch", default="0%", help="Speech pitch")
    parser.add_argument("--output", help="Output audio file path")
    args = parser.parse_args()

    try:
        tts_tester = TTSTester(voice_name=args.voice)
        while True:
            text = input("Enter the text you want to synthesize (or 'q' to quit) > ")
            if text.lower() == 'q':
                break
            tts_tester.synthesize_speech(text, rate=args.rate, pitch=args.pitch, output_file=args.output)
    except TTSError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()