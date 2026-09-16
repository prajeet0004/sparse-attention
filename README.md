# Sparse Attention from Scratch

Manual implementation of dense and sparse attention patterns, with a
correctness harness and a benchmark characterising what sparsity actually
costs.

Postman AI/ML recruitment task, Task 1.

**Headline result:** implementing sparsity by masking saves neither time nor
memory. Peak memory is byte-for-byte identical across all three patterns at
every sequence length tested, because the full `(N, N)` score matrix is
materialised regardless of the pattern. See `WRITEUP.md` section 5.

---

## What is here

| Path | Contents |
|---|---|
| `sparse_attention/attention.py` | Manual scaled dot-product attention. No `F.scaled_dot_product_attention`. |
| `sparse_attention/masks.py` | Causal, sliding-window and block-sparse mask construction, plus a mask visualiser. |
| `tests/test_correctness.py` | 36-case harness: each pattern vs an independent dense reference with the same mask. |
| `tests/test_nan.py` | Fully-masked rows: the trigger, and the fix. |
| `benchmark.py` | Wall-clock and peak memory, 512 → 8192, forward pass only. |
| `check_dense.py` | Sanity check of dense attention against the PyTorch reference. |
| `results/` | Mask figures, benchmark plots, raw benchmark JSON. |
| `WRITEUP.md` | Analysis. |

---

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/prajeet0004/sparse-attention.git
cd sparse-attention

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies are `torch`, `matplotlib`, `pytest`, `numpy`.

---

## Running it

### Correctness harness

```bash
python -m pytest tests/ -v
```

40 cases. Prints PASS/FAIL per case.

The 36 cases in `test_correctness.py` cover three patterns × sequence lengths
{128, 512, 1024} × head dimensions {32, 64} × batch sizes {1, 4}. Each
compares `attention()` against an independent reference implementation using
the same mask.

For a standalone run without pytest:

```bash
python tests/test_correctness.py
```

### Dense attention sanity check

```bash
python check_dense.py
```

Compares the manual implementation against
`F.scaled_dot_product_attention` on random inputs. Prints max absolute
difference; expect under `1e-5` in float32.

### Mask figures

```bash
python sparse_attention/masks.py
```

Prints the sparsity ratio and minimum row count for each pattern at
seq_len 1024, and writes three figures to `results/`.

### Benchmark

```bash
python benchmark.py          # uses CUDA if available
python benchmark.py --cpu    # force CPU
```

Reports the hardware it ran on, then wall-clock and peak memory for each
pattern at sequence lengths 512 → 8192. Writes two plots and a JSON dump to
`results/`.

Numbers in `WRITEUP.md` were collected on a free Colab T4. Relative numbers
only — absolute timings depend entirely on the hardware.

---

## Deliverable status

| | Deliverable | Status |
|---|---|---|
| 1.1 | Manual dense attention | Done |
| 1.2 | Two sparsity patterns | Done — sliding window and block-sparse, plus causal |
| 1.3 | Correctness harness | Done — 36 cases |
| 1.4 | NaN handling | Done — trigger identified, fixed, tested |
| 1.5 | Benchmark | Done — 512 → 8192, both plots |
| 1.6 | Quality evaluation | **Not attempted.** See `WRITEUP.md` section 7. |
| 1.7 | Writeup | Done |

---

## Notes

Sparsity here is implemented purely by masking. That is a real limitation, not
an oversight, and it is discussed in the writeup: it bounds the speedup any
implementation of this shape can demonstrate.
