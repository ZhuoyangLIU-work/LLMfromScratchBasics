import cProfile
import os

def profile_worker(func):
    def wrapped(*args, **kwargs):
        prof = cProfile.Profile()
        prof.enable()
        result = func(*args, **kwargs)
        prof.disable()
        pid = os.getpid()
        prof.dump_stats(f"profile_worker_{pid}.prof")
        return result
    return wrapped