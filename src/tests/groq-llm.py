import time

from groq import Groq
import io
from pyht import Client, TTSOptions, Format
from pydub import AudioSegment
from pydub.playback import play

# Initialize PlayHT API with your credentials
tts_client = Client("fmvEgnXDKNduJSqsROCzeOxlSqU2", "59a777b4518f40dd861aad3e864b5bc6")

# configure your stream
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/male-cs/samuel.json",
    sample_rate=44100,
    format=Format.FORMAT_MP3,
    speed=1,
)

client = Groq(
    api_key="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",
)
messages = [
    {
        "role": "system",
        "content": "you are a helpful assistant. Always return answer"
    },
]

stream = client.chat.completions.create(
    messages=messages,
    model="llama3-70b-8192",
    temperature=0.5,
    max_tokens=50,
    top_p=1,
    stop=None,
    stream=True,
)
try:
    while True:
        text = input("input: ")
        start = time.perf_counter()
        messages.append({
            "role": "user",
            "content": text,
        })
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            stop=None,
        )
        print("output: ", end="")
        print("time before: ", time.perf_counter() - start)
        text = chat_completion.choices[0].message.content
        print(text)
        print("time: ", time.perf_counter() - start)

        # # Generate audio_files and store it in a BytesIO object
        # audio_data = io.BytesIO()
        # for chunk in tts_client.tts(text=text, voice_engine="PlayHT2.0-turbo", options=options):
        #     audio_data.write(chunk)
        # print("ended")
        #
        # # Reset the BytesIO object's position
        # audio_data.seek(0)
        #
        # # Load the audio_files data with pydub
        # audio_files = AudioSegment.from_mp3(audio_data)
        #
        # # Play the audio_files
        # play(audio_files)

finally:
    pass
