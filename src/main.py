import asyncio
import random
import threading

from core.state import Context, ConcreteStateA, ConcreteStateB
from modules.vision.eye_tracker import EyeTracker
from src.modules.vision.eye_controller import MechanicalEyes


async def blink(Eye_controller=MechanicalEyes):
    while True:
        if random.randint(1, 30) == 3:
            Eye_controller.send_data(Eye_controller.CLOSE_EYES)
            await asyncio.sleep(0.1)
            Eye_controller.send_data(Eye_controller.OPEN_EYES)


async def main():
    # print("test states:")
    # context = Context(ConcreteStateA())
    # context.request()  # Выведет: Handling request in State A
    # context.set_state(ConcreteStateB())
    # context.request()  # Выведет: Handling request in State B

    tracker = EyeTracker()
    try:
        await tracker.initialize()
        thread = threading.Thread(target=blink)
        thread.start()
        await tracker.track()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await tracker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
