"""Correctness harness: sparse patterns vs dense-with-the-same-mask.

Run:  python -m pytest tests/ -v
      python tests/test_correctness.py    (standalone, prints PASS/FAIL)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from sparse_attention.attention import attention
from sparse_attention.masks import (
    causal_mask,
    sliding_window_mask,
    block_sparse_mask,
)

ATOL = 1e-5
SEQ_LENS = [128, 512, 1024]
HEAD_DIMS = [32, 64]
BATCHES = [1, 4]
N_HEADS = 4


def reference_attention(q, k, v, mask):
    """Independent dense implementation, written out step by step.

    Deliberately not a call to attention() -- if both were the same code,
    the comparison would be vacuous.
    """
    scores = torch.einsum("bhid,bhjd->bhij", q, k)
    scores = scores / (q.shape[-1] ** 0.5)
    scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", probs, v)


def patterns(seq_len):
    """Every mask under test, at this sequence length."""
    return {
        "causal": causal_mask(seq_len),
        "sliding_window": sliding_window_mask(seq_len, window_size=seq_len // 8),
        "block_sparse": block_sparse_mask(
            seq_len,
            block_size=max(seq_len // 16, 8),
            n_global=8,
            n_random=3,
            seed=0,
        ),
    }


def check(name, seq_len, head_dim, batch):
    """Return (passed, message) for one configuration."""
    torch.manual_seed(0)
    shape = (batch, N_HEADS, seq_len, head_dim)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)

    mask = patterns(seq_len)[name]
    mine = attention(q, k, v, mask=mask)
    ref = reference_attention(q, k, v, mask)

    if torch.isnan(mine).any():
        return False, "NaN in output"
    if mine.shape != ref.shape:
        return False, f"shape {tuple(mine.shape)} != {tuple(ref.shape)}"

    diff = (mine - ref).abs().max().item()
    if diff > ATOL:
        return False, f"max diff {diff:.2e} > {ATOL:.0e}"
    return True, f"max diff {diff:.2e}"


def all_cases():
    for name in ["causal", "sliding_window", "block_sparse"]:
        for seq_len in SEQ_LENS:
            for head_dim in HEAD_DIMS:
                for batch in BATCHES:
                    yield name, seq_len, head_dim, batch


# --- pytest entry point -------------------------------------------------

import pytest


@pytest.mark.parametrize("name,seq_len,head_dim,batch", list(all_cases()))
def test_sparse_matches_masked_dense(name, seq_len, head_dim, batch):
    passed, msg = check(name, seq_len, head_dim, batch)
    assert passed, f"{name} seq={seq_len} d={head_dim} b={batch}: {msg}"


# --- standalone entry point ---------------------------------------------

if __name__ == "__main__":
    failures = 0
    for name, seq_len, head_dim, batch in all_cases():
        passed, msg = check(name, seq_len, head_dim, batch)
        status = "PASS" if passed else "FAIL"
        failures += not passed
        print(f"[{status}] {name:15s} seq={seq_len:5d} d={head_dim:3d} "
              f"b={batch}  {msg}")

    total = len(list(all_cases()))
    print()
    if failures:
        print(f"FAILED: {failures} of {total} cases")
        raise SystemExit(1)
    print(f"PASSED: all {total} cases")