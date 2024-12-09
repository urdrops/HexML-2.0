import json
import os

tools = [
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "visual_analysis",
    #         "description": "Get descriptions of what the camera sees. (ability to see)",
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "draw",
            "description": "To draw as fun users with AI.",
        }
    }
]


def visual_analysis():
    """Evaluate a mathematical expression"""
    try:
        print("ETOT DALBAEB CALL FUNC")
        return json.dumps({"result": "24 C"})
    except:
        return json.dumps({"result": "error to camera"})

