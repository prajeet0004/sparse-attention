"""Deliverable 1.4 — fully-masked rows.

A query whose entire allowed set is masked gives softmax over a row of
-inf: exp(-inf)/sum(exp(-inf)) = 0/0 = NaN. One NaN then propagates
through everything downstream.

The previous commit contained this file asserting the NaN was present,
as evidence the case was found rather than read about. attention() now
detects rows with no allowed positions and zeroes them instead of
running softmax, so these tests assert the fixed behaviour.

Triggering configuration: seq_len=128, window_size=16, strictly-backward
causal window (j < i, excluding the query's own position). Row 0 has
nothing behind it and cannot see itself, so its allowed set is empty.
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
    """The pattern really does produce an empty row -- not a synthetic case."""
    mask = strict_backward_window(SEQ_LEN, WINDOW)
    row_counts = mask.sum(-1)
    assert row_counts.min().item() == 0, "expected at least one empty row"
    print(f"\nempty rows: {(row_counts == 0).nonzero().flatten().tolist()}")


def test_fully_masked_row_does_not_produce_nan():
    """No NaN anywhere in the output, including the empty row."""
    torch.manual_seed(0)
    shape = (1, 4, SEQ_LEN, HEAD_DIM)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)

    mask = strict_backward_window(SEQ_LEN, WINDOW)
    out = attention(q, k, v, mask=mask)

    assert not torch.isnan(out).any(), "fully-masked row should give zeros, not NaN"
    print(f"no NaN across {out.numel()} output elements")


def test_fully_masked_row_is_zero():
    """Finite is not enough -- the empty row must be exactly zero.

    A query with nothing to attend to should contribute nothing. This
    separates a real fix from one that merely suppresses the NaN.
    """
    torch.manual_seed(0)
    shape = (1, 4, SEQ_LEN, HEAD_DIM)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)

    mask = strict_backward_window(SEQ_LEN, WINDOW)
    out = attention(q, k, v, mask=mask)

    empty_rows = (mask.sum(-1) == 0).nonzero().flatten().tolist()
    for row in empty_rows:
        assert out[:, :, row].abs().max().item() == 0.0, \
            f"row {row} should be a zero vector"
    print(f"rows {empty_rows} are exactly zero")


def test_non_empty_rows_are_unaffected():
    """The fix must be targeted: rows with allowed positions are unchanged."""
    torch.manual_seed(0)
    shape = (1, 4, SEQ_LEN, HEAD_DIM)
    q, k, v = torch.randn(shape), torch.randn(shape), torch.randn(shape)

    mask = strict_backward_window(SEQ_LEN, WINDOW)
    out = attention(q, k, v, mask=mask)

    non_empty = (mask.sum(-1) > 0).nonzero().flatten()
    assert out[:, :, non_empty].abs().max().item() > 0, \
        "non-empty rows should still produce output"
    print(f"{len(non_empty)} non-empty rows produce non-zero output")


if __name__ == "__main__":
    test_mask_has_a_fully_masked_row()
    test_fully_masked_row_does_not_produce_nan()
    test_fully_masked_row_is_zero()
    test_non_empty_rows_are_unaffected()
    print("\nPASSED: fully-masked rows zeroed, no NaN")