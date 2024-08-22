import os
import argparse
from typing import Optional
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk


class AzureTTS:
    def __init__(self):
        load_dotenv()
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
        warm_up_text = "Салом"
        warm_up_ssml = self._create_ssml(warm_up_text, self.args.rate, self.args.pitch)
        self.speech_synthesizer.speak_ssml_async(warm_up_ssml).get()
        print("Система инициализирована и готова к быстрой работе.")

    def synthesize_speech(self, text: str) -> speechsdk.SpeechSynthesisResult:
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
        parser.add_argument("--rate", default="1.16", help="Speech rate")
        parser.add_argument("--pitch", default="0%", help="Speech pitch")
        parser.add_argument("--output", help="Output audio_files file path")
        return parser.parse_args()


if __name__ == '__main__':
    tts = AzureTTS()
    try:
        while True:
            text = input("Enter: ")
            tts.synthesize_speech(text)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
