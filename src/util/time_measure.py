import functools
import time
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def measure_time(
    print_enabled: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, tuple[R, float]]]:
    def decorator(func: Callable[P, R]) -> Callable[P, tuple[R, float]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[R, float]:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            if print_enabled:
                print(f"[{func.__name__}] took {elapsed:.6f}s")

            return result, elapsed

        return wrapper

    return decorator
