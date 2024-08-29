import json
import os

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    },
    {
            "type": "function",
            "function": {
                "name": "switch_light",
                "description": "switch room light on",
            }
        }
]


def calculate():
    """Evaluate a mathematical expression"""
    try:
        print("ETOT DALBAEB CALL FUNC")
        return json.dumps({"result": "24 C"})

    except:
        return json.dumps({"error": "Invalid expression"})


def switch_light():
    print("Light on")
    return json.dumps({"result": "light turn on"})
