import random
from abc import ABC, abstractmethod
from typing import Any


class ValueDistribution(ABC):
    @abstractmethod
    def sample(self, rng: random.Random) -> int:
        pass

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class UniformDistribution(ValueDistribution):
    def __init__(self, low: int, high: int) -> None:
        self.low = low
        self.high = high

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    def to_dict(self) -> dict[str, Any]:
        return {"type": str(self), "low": self.low, "high": self.high}

    def __str__(self) -> str:
        return "uniform"


class SkewedDistribution(ValueDistribution):
    def __init__(self, low: int, high: int, negative_ratio: float) -> None:
        self.low = low
        self.high = high
        self.negative_ratio = negative_ratio

    def sample(self, rng: random.Random) -> int:
        decision = rng.random()
        if decision > self.negative_ratio:
            return rng.randint(1, self.high)
        else:
            return rng.randint(self.low, -1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self),
            "low": self.low,
            "high": self.high,
            "negative_ratio": self.negative_ratio,
        }

    def __str__(self) -> str:
        return "skewed"
