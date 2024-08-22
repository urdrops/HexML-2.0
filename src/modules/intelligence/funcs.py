import json
import os


def calculate(expression):
    """Evaluate a mathematical expression"""
    try:
        print("ETOT DALBAEB CALL FUNC")
        result = eval(expression)
        return json.dumps({"result": result})

    except:
        return json.dumps({"error": "Invalid expression"})
