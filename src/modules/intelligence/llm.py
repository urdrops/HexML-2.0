import os
import time
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
print(OPENAI_TOKEN)
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
client = OpenAI(api_key=OPENAI_TOKEN)


class BaseLLM(ABC):
    @abstractmethod
    async def generate_response(self, input_text: str) -> str:
        pass


class OpenAIGPT(BaseLLM):
    def __init__(self):
        self.my_assistant = client.beta.assistants.retrieve(OPENAI_ASSISTANT_ID)
        self.thread = client.beta.threads.create()
        self.conversation = client.beta.threads.retrieve(self.thread.id)

    def generate_response(self, input_text: str) -> str:
        pass

    def add_message(self):
        pass

    def add_image_message(self):
        pass


class GroqLLM(BaseLLM):
    async def generate_response(self, input_text: str) -> str:
        # Implement Groq LLM logic here
        pass
