"""mask construction for sparse attention patterns

Every mask is (seq_len, seq_len) boolean, True = allowed to attend.
The attention function never changes; only the mask does.
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def causal_mask(seq_len, device=None):
    """Query i may attend to key j <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def sliding_window_mask(seq_len, window_size, causal=True, device=None):
    """Query i may attend to keys within window_size of it."""
    idx = torch.arange(seq_len, device=device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    mask = (i - j).abs() <= window_size
    if causal:
        mask &= j <= i
    return mask


def block_sparse_mask(seq_len, block_size, n_global, n_random,
                      causal=True, seed=None, device=None):
    """BigBird-style: local blocks + global tokens + random blocks."""
    n_blocks = (seq_len + block_size - 1) // block_size

    # local — each block attends to itself and its two neighbours
    b = torch.arange(n_blocks)
    block_mask = (b.unsqueeze(1) - b.unsqueeze(0)).abs() <= 1

    # random — each block row picks n_random block columns, seeded
    if n_random > 0:
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        for row in range(n_blocks):
            picks = torch.randperm(n_blocks, generator=gen)[:n_random]
            block_mask[row, picks] = True

    # expand the block grid to the token grid, trim any partial final block
    mask = (block_mask.repeat_interleave(block_size, 0)
                      .repeat_interleave(block_size, 1))[:seq_len, :seq_len]
    mask = mask.to(device)

    # global — first n_global tokens, both directions
    idx = torch.arange(seq_len, device=device)
    mask |= idx.unsqueeze(0) < n_global   # everyone attends to them
    mask |= idx.unsqueeze(1) < n_global   # they attend to everyone

    # causal applied last, after all three components are ORed
    if causal:
        mask &= idx.unsqueeze(1) >= idx.unsqueeze(0)

    return mask


def sparsity(mask):
    """Fraction of positions a query is allowed to attend to."""
    return mask.float().mean().item()


def visualise_mask(mask, title, path):
    """Save a boolean mask as an image. White = allowed, black = masked."""
    plt.figure(figsize=(4, 4))
    plt.imshow(mask.cpu().numpy(), cmap="gray", interpolation="nearest")
    plt.title(f"{title} — {sparsity(mask):.1%} allowed")
    plt.xlabel("key j")
    plt.ylabel("query i")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    n = 1024

    patterns = {
        "causal": causal_mask(n),
        "sliding_window_w128": sliding_window_mask(n, window_size=128),
        "block_sparse_b64_g8_r3": block_sparse_mask(
            n, block_size=64, n_global=8, n_random=3, seed=0
        ),
    }

    for name, mask in patterns.items():
        print(f"{name:26s} sparsity {sparsity(mask):.3f}  "
              f"min row {mask.sum(-1).min().item()}")
        visualise_mask(mask, name, f"results/mask_{name}.png")