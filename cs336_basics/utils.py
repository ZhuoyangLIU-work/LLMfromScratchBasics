import cProfile
import pstats
import os
from functools import wraps
from datetime import datetime
import torch
import math
from jaxtyping import Float, Int
from einops import rearrange, einsum




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


def init_xnt_std(d_in, d_out):
    return math.sqrt(2.0 / (d_in + d_out))

def softmax(x: torch.Tensor, dim: int = -1):
    normalized_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp_normalized_x = torch.exp(normalized_x)
    return exp_normalized_x / torch.sum(exp_normalized_x, dim=dim, keepdim=True)

def scaled_dot_product_attention(input_q: Float[torch.Tensor, "batch_size ... query d_k"],
                                 input_k: Float[torch.Tensor, "batch_size ... key d_k"],
                                 input_v: Float[torch.Tensor, "batch_size ... key d_v"],
                                 mask: Float[torch.Tensor, "seq_len seq_len"] = None) -> Float[torch.Tensor, "batch_size ... d_v"]:
    pre_soft = einsum(input_q, input_k, "... query d_k, ... key d_k -> ... query key") / math.sqrt(input_k.shape[-1])

    if mask is not None:
        pre_soft = torch.where(mask, pre_soft, float("-inf"))

    attention_probs = softmax(pre_soft, dim=-1) # softmax over the dimension for key
    return einsum(attention_probs, input_v, "... query key, ... key d_v -> ... query d_v")


