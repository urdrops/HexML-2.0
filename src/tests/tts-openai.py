import os
import time
import io

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_TOKEN)

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="echo",
    input="""Привет друг! что нового расскажешь? например какая погода сегодня прекрасная и идеальная""",
) as response:
    with open("output.mp3", mode="wb") as f:
        start = time.perf_counter()
        for data in response.iter_bytes(chunk_size=32000):
            print(data)
            f.write(data)
            print("chunk write time: ", time.perf_counter() - start)
print("time: ", time.perf_counter() - start)
