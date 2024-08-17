import asyncio
from core.state import Context, ConcreteStateA, ConcreteStateB
from modules.vision.eye_tracker import EyeTracker


async def main():
    print("test states:")
    context = Context(ConcreteStateA())
    context.request()  # Выведет: Handling request in State A
    context.set_state(ConcreteStateB())
    context.request()  # Выведет: Handling request in State B

    tracker = EyeTracker()
    try:
        await tracker.initialize()
        await tracker.track()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await tracker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
