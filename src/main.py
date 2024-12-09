import asyncio
import threading

from core.state import Context, TalkState
from modules.vision.eye_tracker import EyeTracker
from src.core.assistant import Assistant
from src.core.state import SleepState
from src.modules.vision.eye_controller import MechanicalEyes

context = Context(SleepState())
Eye_controller = MechanicalEyes()
tracker = EyeTracker(context)


async def main():
    try:
        await tracker.initialize()

        await tracker.track()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        Eye_controller.send_data(Eye_controller.CLOSE_EYES)
        await tracker.shutdown()


if __name__ == "__main__":

    # Example usage:
    print("Initilization..")
    assistant = Assistant.create(stt_type="mohirai", llm_type="openai", tts_type="azure")
    assis_thread = threading.Thread(target=assistant.run, args=[context, TalkState])
    assis_thread.start()

    asyncio.run(main())
