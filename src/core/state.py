from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def handle(self):
        pass


class ConcreteStateA(State):
    def handle(self):
        print("Handling request in State A")
        # Логика обработки для состояния A


class ConcreteStateB(State):
    def handle(self):
        print("Handling request in State B")
        # Логика обработки для состояния B


class Context:
    def __init__(self, initial_state: State):
        self._state = initial_state

    def set_state(self, state: State):
        self._state = state

    def request(self):
        self._state.handle()

    def get_current_state(self):
        return self._state
