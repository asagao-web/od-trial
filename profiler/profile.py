from functools import wraps
import time

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@ " * 20)
        print(f"@ timefn: {fn.__name__} took {t2 - t1} seconds")
        print("@ " * 20)
        return result
    return measure_time
