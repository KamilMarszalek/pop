import time
import functools

_measurement_active = False


def measure_time(print_enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global _measurement_active

            if _measurement_active:
                return func(*args, **kwargs)

            _measurement_active = True
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if print_enabled:
                    print(f"[{func.__name__}] took {elapsed:.6f}s")
                _measurement_active = False

        return wrapper

    return decorator
