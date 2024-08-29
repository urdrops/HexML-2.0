import openai
import json
from dotenv import load_dotenv
import os

load_dotenv()

# Настройка ключа API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")


# Определение функции для получения погоды
def get_weather(location):
    # В реальном приложении здесь был бы запрос к API погоды
    return f"Погода в {location}: 22°C, солнечно"


# Определение доступных функций
functions = [
    {
        "name": "get_weather",
        "description": "Получить текущую погоду в указанном месте",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "Город или населенный пункт, например 'Москва' или 'Санкт-Петербург'"
                }
            },
            "required": ["location"]
        }
    }
]


def chat_with_gpt(user_input):
    messages = [{"role": "user", "content": user_input}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto"
    )

    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        function_name = response_message["function_call"]["name"]
        function_args = json.loads(response_message["function_call"]["arguments"])

        if function_name == "get_weather":
            function_response = get_weather(function_args["location"])

            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": function_name,
                "content": function_response
            })

            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages
            )
            return second_response["choices"][0]["message"]["content"]
    else:
        return response_message["content"]


# Пример использования
user_input = "Какая сегодня погода в Москве?"
response = chat_with_gpt(user_input)
print(response)