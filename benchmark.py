"""Deliverable 1.5 — wall-clock and peak memory, dense vs sparse patterns.

Forward pass only. Reports the hardware it ran on; relative numbers only.

Run:  python benchmark.py
"""

import argparse
import json
import platform
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sparse_attention.attention import attention
from sparse_attention.masks import causal_mask, sliding_window_mask, block_sparse_mask

SEQ_LENS = [512, 1024, 2048, 4096, 8192]
BATCH = 1
N_HEADS = 8
HEAD_DIM = 64
WARMUP = 5
ITERS = 10


def hardware_report(device):
    """Everything a reader needs to interpret the numbers."""
    info = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["cuda"] = torch.version.cuda
    return info


def build_masks(seq_len, device):
    """The patterns under test, at this sequence length."""
    return {
        "dense (causal)": causal_mask(seq_len, device=device),
        "sliding window": sliding_window_mask(
            seq_len, window_size=256, device=device
        ),
        "block sparse": block_sparse_mask(
            seq_len, block_size=64, n_global=8, n_random=3,
            seed=0, device=device,
        ),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure(q, k, v, mask, device):
    """Median wall-clock (ms) and peak memory (MB) for one configuration."""
    with torch.no_grad():
        # Warmup: the first CUDA call includes kernel compilation and would
        # otherwise dominate the first measurement.
        for _ in range(WARMUP):
            attention(q, k, v, mask=mask)
        sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(ITERS):
            sync(device)                      # CUDA is async: without these
            t0 = time.perf_counter()          # we would be timing how long it
            attention(q, k, v, mask=mask)     # takes to *queue* the work,
            sync(device)                      # not to do it.
            times.append((time.perf_counter() - t0) * 1000)

        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if device.type == "cuda" else float("nan"))

    times.sort()
    return times[len(times) // 2], peak_mb


def run(device):
    results = {name: {"seq": [], "ms": [], "mb": []}
               for name in build_masks(512, device)}

    for seq_len in SEQ_LENS:
        shape = (BATCH, N_HEADS, seq_len, HEAD_DIM)
        torch.manual_seed(0)
        q = torch.randn(shape, device=device)
        k = torch.randn(shape, device=device)
        v = torch.randn(shape, device=device)

        for name, mask in build_masks(seq_len, device).items():
            try:
                ms, mb = measure(q, k, v, mask, device)
                results[name]["seq"].append(seq_len)
                results[name]["ms"].append(ms)
                results[name]["mb"].append(mb)
                print(f"{name:16s} seq={seq_len:5d}  {ms:8.2f} ms  {mb:9.1f} MB")
            except torch.cuda.OutOfMemoryError:
                # OOM is a result, not a failure: it is the O(N^2) memory
                # cost becoming visible.
                print(f"{name:16s} seq={seq_len:5d}  OOM")
                torch.cuda.empty_cache()

            del mask
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del q, k, v
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def plot(results, key, ylabel, title, path):
    plt.figure(figsize=(6, 4))
    for name, data in results.items():
        if data["seq"]:
            plt.plot(data["seq"], data[key], marker="o", label=name)
    plt.xscale("log", base=2)
    plt.xlabel("sequence length")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    info = hardware_report(device)
    print("=" * 60)
    for key, value in info.items():
        print(f"{key:16s} {value}")
    print("=" * 60)

    results = run(device)

    plot(results, "ms", "time per forward pass (ms)",
         "Forward-pass wall clock", "results/benchmark_time.png")
    plot(results, "mb", "peak memory (MB)",
         "Peak memory allocated", "results/benchmark_memory.png")

    with open("results/benchmark.json", "w") as f:
        json.dump({"hardware": info, "results": results}, f, indent=2)

    print("\nwrote results/benchmark_time.png, benchmark_memory.png, benchmark.json")