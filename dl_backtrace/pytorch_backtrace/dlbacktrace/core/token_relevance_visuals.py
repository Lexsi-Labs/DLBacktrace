"""
DLBacktrace Visualization Utilities
===================================
Provides:
1. Token-wise relevance map (input→generated)
2. Input heatmap for n-th generated token

Author: <Your Name>
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import re

# ================================================================
# ----------------------- HELPER FUNCTIONS -----------------------
# ================================================================

def _to_numpy(x):
    """Convert tensor/list to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x


def _find_input_key(rel_dict, preferred="input_ids"):
    """Identify the correct key for input relevance tensors."""
    if preferred in rel_dict:
        return preferred
    candidates = [k for k in rel_dict.keys() if isinstance(k, str)]
    for key in ("input_ids", "inputs", "input", "tokens", "embedding_input", "input_embeds"):
        for k in candidates:
            if key in k.lower():
                return k
    return next(iter(rel_dict.keys()))


def _reduce_to_token_axis(arr, target_len):
    """Reduce a tensor to per-token relevance values."""
    a = _to_numpy(arr)
    if a.ndim == 1 and a.shape[0] == target_len:
        return a.astype(np.float64)
    if target_len in a.shape:
        axis = list(a.shape).index(target_len)
        a = np.moveaxis(a, axis, 0)
        if a.ndim > 1:
            a = a.sum(axis=tuple(range(1, a.ndim)))
        return a.astype(np.float64)
    size = a.size
    if size % target_len == 0:
        return a.reshape(target_len, size // target_len).sum(axis=1)
    flat = a.ravel()
    out = np.zeros(target_len)
    out[:min(len(flat), target_len)] = flat[:target_len]
    return out


def _clean_tokens(tokens):
    """Remove tokenizer markers (Ġ, ▁) and show placeholder for empty."""
    cleaned = []
    for t in tokens:
        t = re.sub(r"^[Ġ▁]+", "", t).strip()
        cleaned.append(t if t else "␣")
    return cleaned


# ================================================================
# -------------------- TOKENWISE RELEVANCE MAP -------------------
# ================================================================

def plot_tokenwise_relevance_map_swapped(
    timewise_relevance_out3,
    input_ids,
    tokenizer,
    generated_ids=None,
    input_key="input_ids",
    figsize=(12, 6)
):
    """
    Visualize normalized tokenwise relevance:
    x-axis → generated tokens
    y-axis → input tokens
    """
    # ---- prepare input tokens ----
    if isinstance(input_ids, torch.Tensor):
        inp = input_ids.detach().cpu().long().view(-1).numpy()
    else:
        inp = np.array(input_ids).reshape(-1)
    if inp.ndim > 1:
        inp = inp[0]
    prompt_len = int(inp.shape[0])
    y_labels = _clean_tokens(tokenizer.convert_ids_to_tokens(inp.tolist()))

    # ---- generated token labels ----
    T_steps = len(timewise_relevance_out3)
    if generated_ids is None:
        x_labels = [f"t={t}" for t in range(T_steps)]
    else:
        if isinstance(generated_ids, torch.Tensor):
            g = generated_ids.detach().cpu().long()
            if g.ndim == 2:
                g_new = g[0, -T_steps:].numpy()
            else:
                g_new = g.numpy()[-T_steps:]
        else:
            g_arr = np.array(generated_ids)
            g_new = g_arr[-T_steps:]
        toks = _clean_tokens(tokenizer.convert_ids_to_tokens(list(map(int, g_new.tolist() if hasattr(g_new, "tolist") else g_new))))
        x_labels = [f"{t}: {toks[t]}" for t in range(min(T_steps, len(toks)))]
        if len(x_labels) < T_steps:
            x_labels += [f"{t}" for t in range(len(x_labels), T_steps)]

    # ---- build matrix [input_len, T_steps] ----
    H = np.zeros((prompt_len, T_steps), dtype=np.float64)
    for t, rel_dict in enumerate(timewise_relevance_out3):
        k = _find_input_key(rel_dict, preferred=input_key)
        vec = _reduce_to_token_axis(rel_dict[k], prompt_len)
        H[:, t] = vec

    # ---- normalize ----
    H_min, H_max = H.min(), H.max()
    if H_max > H_min:
        H = (H - H_min) / (H_max - H_min)
    else:
        H[:] = 0.5

    # ---- plot ----
    plt.figure(figsize=figsize)
    im = plt.imshow(H, aspect="auto", origin="lower", vmin=0, vmax=1)
    plt.xlabel("Generated tokens (time)")
    plt.ylabel("Input tokens")
    plt.xticks(ticks=np.arange(T_steps), labels=x_labels, rotation=90)
    plt.yticks(ticks=np.arange(prompt_len), labels=y_labels)
    plt.title("Normalized Tokenwise relevance map (generated token timestep → input tokens)")
    plt.colorbar(im, shrink=0.8, label="Normalized relevance (0–1)")
    plt.tight_layout()
    plt.show()


# ================================================================
# --------------------- SINGLE TOKEN HEATMAP ---------------------
# ================================================================

def plot_input_heatmap_for_token(
    timewise_relevance_out3,
    n,
    input_ids,
    tokenizer,
    generated_ids=None,
    input_key="input_ids",
    figsize=(10, 3)
):
    """
    Plot the relevance of input tokens for the n-th generated token,
    showing the actual token string in the title.
    """
    if n < 0 or n >= len(timewise_relevance_out3):
        raise IndexError(f"Invalid token index n={n}; must be in [0, {len(timewise_relevance_out3)-1}]")

    # ---- prepare input tokens ----
    if isinstance(input_ids, torch.Tensor):
        inp = input_ids.detach().cpu().long().view(-1).numpy()
    else:
        inp = np.array(input_ids).reshape(-1)
    if inp.ndim > 1:
        inp = inp[0]
    prompt_len = int(inp.shape[0])
    x_labels = _clean_tokens(tokenizer.convert_ids_to_tokens(inp.tolist()))

    # ---- identify generated token ----
    if generated_ids is not None:
        if isinstance(generated_ids, torch.Tensor):
            g = generated_ids.detach().cpu().long()
            if g.ndim == 2:
                g_new = g[0, prompt_len:].numpy()
            else:
                g_new = g.numpy()[prompt_len:]
        else:
            g_arr = np.array(generated_ids)
            if g_arr.ndim == 1 and g_arr.size >= prompt_len:
                g_new = g_arr[prompt_len:]
            else:
                g_new = g_arr
        gen_toks = _clean_tokens(tokenizer.convert_ids_to_tokens(g_new.tolist()))
        gen_token_str = gen_toks[n] if n < len(gen_toks) else f"t={n}"
    else:
        gen_token_str = f"t={n}"

    # ---- extract relevance ----
    rel_dict = timewise_relevance_out3[n]
    k = _find_input_key(rel_dict, preferred=input_key)
    vec = _reduce_to_token_axis(rel_dict[k], prompt_len)

    # ---- normalize ----
    vmin, vmax = vec.min(), vec.max()
    vec = (vec - vmin) / (vmax - vmin) if vmax > vmin else np.full_like(vec, 0.5)

    # ---- plot ----
    plt.figure(figsize=figsize)
    plt.imshow(vec[np.newaxis, :], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.yticks([])
    plt.xticks(np.arange(prompt_len), x_labels, rotation=90)
    plt.xlabel("Input tokens")
    plt.title(f"Input relevance heatmap for generated token: '{gen_token_str}'")
    plt.colorbar(label="Normalized relevance (0–1)")
    plt.tight_layout()
    plt.show()
    