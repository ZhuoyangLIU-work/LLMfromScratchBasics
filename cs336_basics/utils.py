import cProfile
import pstats
import os
from functools import wraps
from datetime import datetime


def profile(
    output_path: str = "profile_output.prof",
    sort_by: str = "cumtime",
    print_top_n: int = 20,
    strip_dirs: bool = True,
):
    """
    Decorator for profiling a function with cProfile.

    Args:
        output_path (str): File path to save profiling stats (.prof file).
        sort_by (str): Sorting key for printed stats (e.g., 'cumtime', 'tottime', 'ncalls').
        print_top_n (int): Number of top lines to print from profiling.
        strip_dirs (bool): Whether to strip directory paths in output.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profile_filename = output_path
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            profiler.dump_stats(profile_filename)

            # Print summary to console
            print(f"\n[cProfile] Profiling complete. Stats saved to {profile_filename}")
            stats = pstats.Stats(profile_filename)
            if strip_dirs:
                stats.strip_dirs()
            stats.sort_stats(sort_by)
            stats.print_stats(print_top_n)

            return result
        return wrapper
    return decorator
