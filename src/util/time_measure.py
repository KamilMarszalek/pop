import functools
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


_measurement_active = False


def measure_time(
    print_enabled: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, tuple[R, float]]]:
    def decorator(func: Callable[P, R]) -> Callable[P, tuple[R, float]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
            global _measurement_active
            if _measurement_active:
                return (func(*args, **kwargs), 0.0)

            _measurement_active = True
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                return result, elapsed
            finally:
                elapsed = time.perf_counter() - start
                if print_enabled:
                    print(f"[{func.__name__}] took {elapsed:.6f}s")
                _measurement_active = False

        return wrapper

    return decorator
