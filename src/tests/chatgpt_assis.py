import time
import pygame as player
from typing_extensions import override
from openai import AssistantEventHandler
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
print(OPENAI_TOKEN)
OPENAI_ASSISTANT_Ipath_to_your_audio_fileD = os.getenv("OPENAI_ASSISTANT_ID")
client = OpenAI(api_key=OPENAI_TOKEN)

thread = client.beta.threads.create()
player.init()


class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
        end = time.perf_counter()
        print(f"Время выполнения: {end - run_start} секунд")

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)

    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)

    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)


# Then, we use the `stream` SDK helper
# with the `EventHandler` class to create the Run
# and stream the response.
while True:
    text = input("\nyou > ")
    message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=text,
            )
    run_start = time.perf_counter()
    with client.beta.threads.runs.stream(
            thread_id=thread.id,
            assistant_id="asst_wdvlPqOXwcXFHcKjzEPJEx7L",
            event_handler=EventHandler(),
    ) as stream:
        stream.until_done()
