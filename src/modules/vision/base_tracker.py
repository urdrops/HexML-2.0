from abc import ABC, abstractmethod


class BaseTracker(ABC):
    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def track(self):
        pass

    @abstractmethod
    async def shutdown(self):
        pass