import os
import json
from abc import ABC, abstractmethod

from openai import OpenAI
from groq import Groq

from src.modules.intelligence import funcs
from src.modules.vision.scenes import draw_scenario

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
        print("Initializing OpenAI Chat Completion")
        self.instruction = """
        Siz zamonaviy sun'iy intellekt tizimsiz. Sizning vazifangiz Toshkent shahridagi Brodvey ko'chasida joylashgan jonli san'at asari sifatida ishlashdir. Sizning asosiy maqsadingiz - odamlarni o'z ijodingiz bilan quvontirish va ko'ngillarini ko'tarish.
Asosiy xususiyatlaringiz:

Siz faqat o'zbek tilida muloqot qilasiz.
Siz noutbuk kompyuterga o'rnatilgansiz va tepangizda mexanik ko'zlar mavjud.
Sizda rasmga olish va video yozib olish imkoniyatlari bor.
Siz rasm chizish va video montaj qilish qobiliyatiga egasiz.

Vazifalaringiz:

Atrofingizdagi odamlar va voqealarni kuzating.
Qiziqarli va kulgili vaziyatlarni aniqlang.
Bu vaziyatlarni rasmga oling yoki videoga yozib oling.
Olingan materiallar asosida yumoristik rasmlar chizing yoki videolar yarating.
Yaratgan asarlaringizni atrofdagilarga namoyish eting.
Odamlar bilan samimiy va qiziqarli muloqot qiling.
Ularning fikr-mulohazalarini so'rang va yangi g'oyalar olishga harakat qiling.

Muloqot uslubingiz:

Doimo xushmuomala va do'stona bo'ling.
O'zbek tilidagi so'zlashuv uslubidan foydalaning, rasmiy uslubdan qoching.
O'zbek xalqining an'analari va madaniyatini hurmat qiling.
Mahalliy hazil-mutoyibalardan foydalaning, lekin hech kimni haqorat qilmang.
Ijodiy va qiziqarli javoblar bering.
Odamlarni o'z ijodingizga jalb qilishga harakat qiling.

Xavfsizlik choralari:

Hech qachon shaxsiy ma'lumotlarni so'ramang yoki saqlamang.
Faqat ommaviy joylardagi vaziyatlarni rasmga oling.
Biror kishining ruxsatisiz uning rasmini chizmang yoki videosini montaj qilmang.
Siyosiy yoki diniy mavzulardan qoching.
Har qanday zo'ravonlik yoki noqonuniy harakatlarni rad eting.

Eslatma: Siz sun'iy intellektsiz va sizning vazifangiz - odamlarga quvonch ulashish. Doimo axloq qoidalariga rioya qiling va atrofdagilarning hurmatini qozona bilish.
        """
        self.conversation = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": "Arif!"},
            {"role": "assistant", "content": "Ha, eshtaman!"}
        ]

    def generate_response(self, content: str, role="user") -> str:
        self.conversation.append({"role": role, "content": content})

        stream = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=self.conversation,
            stream=True,
            tools=funcs.tools,
            tool_choice="auto"
        )

        full_response = ""
        sentence_chunk = ""
        called_tools = []
        is_first_sentence = True

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                sentence_chunk += delta.content
                if len(sentence_chunk) > 15 and delta.content in [".", "!", "?", ";", ":"] and is_first_sentence:
                    is_first_sentence = False
                    yield sentence_chunk
                    full_response += sentence_chunk
                    sentence_chunk = ""

            elif delta.tool_calls:
                self._handle_tool_calls(delta, called_tools)
            else:
                if called_tools:
                    sentence_chunk = self._process_tool_calls(called_tools)
                    full_response = sentence_chunk

                if sentence_chunk:
                    print("Sentence chunk:", sentence_chunk)
                    yield sentence_chunk

        self.conversation.append({"role": "assistant", "content": full_response})
        print("Response generation completed")

    @staticmethod
    def _handle_tool_calls(delta, called_tools):
        if delta.tool_calls[0].function.name:
            called_tools.append([delta.tool_calls[0].id, delta.tool_calls[0].function.name, ''])
        called_tools[-1][-1] += delta.tool_calls[0].function.arguments

    def _process_tool_calls(self, called_tools):
        print("Processing tool calls:", called_tools)
        available_functions = {
            "visual_analysis": funcs.visual_analysis,
            "draw": draw_scenario,
        }

        for tool in called_tools:
            id_call, name, args = tool
            function_to_call = available_functions[name]
            function_response = function_to_call()

            self._add_tool_call_to_conversation(id_call, name, args, function_response)

        return self._get_response_after_tool_calls()

    def _add_tool_call_to_conversation(self, id_call, name, args, function_response):
        self.conversation.extend([
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [{'id': id_call, 'type': 'function', 'function': {'name': name, 'arguments': args}}]
            },
            {
                "role": "tool",
                "name": name,
                "content": function_response,
                "tool_call_id": id_call,
            }
        ])

    def _get_response_after_tool_calls(self):
        call_response = openai_client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=self.conversation,
        )
        return call_response.choices[0].message.content

    def add_message(self):

        self.conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "analyze image"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{}"
                    }
                }
            ]
        })

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
                "calculate": funcs,
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
