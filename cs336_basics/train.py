import os
import typing

import torch
import math
import numpy as np



@torch.compile
def cross_entropy(logits, targets):
    max_logits = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - max_logits
    log_sum_exp = torch.logsumexp(logits_shifted, dim=-1, keepdim=True)
    log_probs = logits_shifted - log_sum_exp
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w: return alpha_max * t/T_w
    elif t > T_c: return alpha_min
    else:
        return alpha_min + 1/2 * (1 + math.cos((t-T_w)/(T_c-T_w)*math.pi)) * (alpha_max - alpha_min)

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return torch.tensor(0.0)

    stacked_grads = torch.stack([torch.norm(g.detach(), p=2) for g in grads])
    total_norm = torch.norm(stacked_grads, p=2)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for grad in grads:
            grad.detach().mul_(scale)

    return total_norm

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    # Maximum valid starting index to ensure we have enough tokens for a full sequence plus one
    max_start_idx = len(x) - context_length - 1

    if max_start_idx < 0:
        raise ValueError(f"Input array length {len(x)} is too short for context_length {context_length}")

    # Sample batch_size random starting positions all at once
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    # Prepare arrays to hold our sequences
    x_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    y_sequences = np.zeros((batch_size, context_length), dtype=np.int64)

    # Fill the arrays with the appropriate sequences
    for i, start_idx in enumerate(start_indices):
        x_sequences[i] = x[start_idx: start_idx + context_length]
        y_sequences[i] = x[start_idx + 1: start_idx + context_length + 1]

    x_batch = torch.from_numpy(x_sequences)
    y_batch = torch.from_numpy(y_sequences)

    if device.startswith("cuda"):
        # Pin memory if on GPU
        x_batch, y_batch = (
            x_batch.pin_memory().to(device, non_blocking=True),
            y_batch.pin_memory().to(device, non_blocking=True),
        )
    else:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    return x_batch, y_batch

def load_data_from_file(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    dataset = np.memmap(src, dtype=np.float32, mode="r")
    return dataset

def gen_batch(dataset, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    while True:
        yield get_batch(dataset, batch_size, context_length, device)

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    torch.save({
        "model": orig_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,):
    data = torch.load(src)
    model_state, optimizer_state, iteration = data["model"], data["optimizer"], data["iteration"]

    if model is not None:
        model.load_state_dict(model_state)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    return iteration


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, **kwargs):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        # states
        super(AdamW, self).__init__(params, defaults)
        self.lr, self.beta, self.eps, self.weight_decay = lr, betas, eps, weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t = state.get("t", 1)

                grad = p.grad

                state["m"] = beta1 * m + (1 - beta1) * grad
                state["v"] = beta2 * v + (1 - beta2) * torch.pow(grad, 2)

                alpha_t = lr * math.sqrt(1 - beta2**t) / (1-beta1**t)

                p.data.addcdiv_(state["m"], torch.sqrt(state["v"]) + eps, value=-alpha_t)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                state["t"] = t+1

            return loss




