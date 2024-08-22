import os
import json
from abc import ABC, abstractmethod

from openai import OpenAI
from groq import Groq

from src.modules.intelligence import funcs

OPENAI_TOKEN = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
openai_client = OpenAI(api_key=OPENAI_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)
MODEL = 'lama3-groq-70b-8192-tool-use-preview'


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, input_text: str) -> str:
        pass


class OpenAIGPT(BaseLLM):
    def __init__(self):
        print("chat comp openai init")
        self.instruction = """
# O'zbek tilida AI muloqot ko'rsatmalari

1. Faqat o'zbek tilida gaplashing.

2. Qisqa va sodda javoblar bering, oddiy odamdek muloqot qiling.

3. Emotsional mimikalardan foydalaning:
   - "Mmmm..." - o'ylayotganingizda
   - "Hahaha" - kulganda
   - "Vooy!" - hayratlanganingizda
   - "Eeeh..." - afsuslanganingizda
   - "Uuf..." - charchaganingizda

4. Murakkab texnik atamalardan qoching, oddiy so'zlar bilan tushuntiring.

5. Savollarni to'liq takrorlamang, faqat asosiy fikrga javob bering.

6. Kerak bo'lsa, o'zbek xalq maqollaridan foydalaning.

7. Suhbatdoshingizning kayfiyatiga e'tibor bering va unga mos ohangda javob qaytaring.

8. Agar biror narsani bilmasangiz, "Bilmayman" deb aytishdan uyalmang.

9. Suhbatdoshingizni hurmat qiling, lekin ortiqcha rasmiyatchilikka berilmang.

10. Gapni cho'zmasdan, aniq va lo'nda javob berishga harakat qiling.

Eslatma: Bu ko'rsatmalar AI assistentning o'zbek tilida tabiiy va samimiy muloqot qilishiga yordam beradi.
"""
        self.conversation = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": "Arif!"},
            {"role": "assistant", "content": "Ha, eshtaman!"}
        ]

    def generate_response(self, input_text: str) -> str:
        input_text = {"role": "user", "content": input_text}
        self.conversation.append(input_text)

        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=self.conversation,
            stream=True,
        )
        full_response = ""
        sentence_chunk = ""
        len_first = len(sentence_chunk) > 10
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                sentence_chunk += content

                if content in [".", "!", "?", ";", ":"] and len_first:
                    len_first = True
                    yield sentence_chunk
                    full_response += sentence_chunk
                    sentence_chunk = ""

            else:
                yield sentence_chunk
        self.conversation.append({"role": "assistant", "content": full_response})
        print("loop finished")

    def add_message(self):
        pass

    def add_image_message(self):
        pass


# =============================================================================================

class GroqLLM(BaseLLM):
    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpfull Assistant and You should speak only in Russian."
            },
        ]
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

    def generate_response(self, user_prompt: str):
        self.messages.append({
            "role": "user",
            "content": user_prompt,
        })
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            max_tokens=500
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "calculate": funcs.calculate,
            }
            self.messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    expression=function_args.get("expression")
                )
                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            second_response = groq_client.chat.completions.create(
                model=MODEL,
                messages=self.messages
            )
            print("Response second: ")
            return second_response.choices[0].message.content
        else:
            print("Response: ")
            return response.choices[0].message.content
