from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def handle(self):
        pass


class SleepState(State):
    def handle(self):
        return "sleep"
        # Логика обработки для состояния B


class WakeUpState(State):
    def handle(self):
        return "wakeup"
        # Логика обработки для состояния A


class TalkState(State):
    def handle(self):
        return "talk"
        # Логика обработки для состояния B


class FuncState(State):
    def handle(self):
        return "func"
        # Логика обработки для состояния B


class Context:
    def __init__(self, initial_state: State):
        self._state = initial_state

    def set_state(self, state: State):
        self._state = state

    def request(self):
        return self._state.handle()

    def get_current_state(self):
        return self._state
