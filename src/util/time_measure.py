import time
from functools import wraps


def measure_time(store_results=None):
    def decorator(func):
        label = None
        log_func = print

        @wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__qualname__
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                log_func(f"[{name}] took {duration:.6f}s")

        return wrapper

    return decorator
