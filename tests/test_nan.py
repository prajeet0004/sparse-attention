"""Deliverable 1.4 — fully-masked rows produce NaN.

A query whose entire allowed set is masked gives softmax over a row of
-inf: exp(-inf)/sum(exp(-inf)) = 0/0 = NaN. One NaN then propagates
through everything downstream.

Committed while attention() still has the bug, as evidence the case was
found rather than read about.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sparse_attention.attention import attention

SEQ_LEN = 128
WINDOW = 16
HEAD_DIM = 32


def strict_backward_window(seq_len, window_size):
    """Sliding window that excludes the query's own position.

    Allowed: i - window_size <= j < i  (strictly before i).
    Row 0 has nothing behind it and cannot see itself, so it is empty.
    """
    idx = torch.arange(seq_len)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    return (j < i) & (j >= i - window_size)


def test_mask_has_a_fully_masked_row():
    mask = strict_backward_window(SEQ_LEN, WINDOW)
    row_counts = mask.sum(-1)
    assert row_counts.min().item() == 0, "expected at least one empty row"
    print(f"\nempty rows: {(row_counts == 0).nonzero().flatten().tolist()}")


def test_fully_masked_row_produces_nan():
    """Documents current broken behaviour. Invert this once fixed."""
    torch.manual_seed(0)
    shape = (1, 4, SEQ_LEN, HEAD_DIM)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)

    mask = strict_backward_window(SEQ_LEN, WINDOW)
    out = attention(q, k, v, mask=mask)

    assert torch.isnan(out).any(), "expected NaN from the fully-masked row"
    nan_rows = torch.isnan(out).any(-1)[0, 0].nonzero().flatten().tolist()
    print(f"NaN at query rows: {nan_rows}")


if __name__ == "__main__":
    test_mask_has_a_fully_masked_row()
    test_fully_masked_row_produces_nan()
    print("\nconfirmed: fully-masked row -> NaN")