# Writeup — Sparse Attention from Scratch

Postman AI/ML recruitment task, Task 1.

---

## 1. What I built

Scaled dot-product attention written by hand, three mask patterns (causal,
sliding-window, block-sparse), a correctness harness comparing each pattern
against an independent dense reference, a deliberate reproduction and fix of
the fully-masked-row NaN, and a forward-pass benchmark from sequence length
512 to 8192 on a T4.

The core design decision, which the task points at and which turned out to be
the most useful thing I understood: **the attention function never changes.**
Every pattern is a different boolean `(seq_len, seq_len)` mask passed to the
same five lines. That makes the patterns trivially comparable — and, as
section 5 explains, it also means none of them are actually faster.

I did not attempt deliverable 1.6. Section 7 says what I would have expected
and why I am not claiming it as a result.

---

## 2. What each pattern loses

**Sliding window.** Position `i` may attend to `[i - w, i]`. Information
travels at most one window-width per layer, so a 2-layer model has an
effective receptive field of roughly `2 × w`. Token 10 physically cannot
influence token 5000 — not "weakly influences", cannot. The connectivity
graph is a path, and its diameter grows linearly with sequence length.

**Block-sparse (BigBird-style).** Three components ORed together: local
neighbouring blocks, a handful of global tokens, and random block-to-block
links. The random links behave like expander-graph edges — a few of them
shorten paths between arbitrary nodes dramatically. The global tokens act as
a hub, discussed separately below.

Measured sparsity at seq_len 1024:

| pattern | positions allowed |
|---|---|
| causal | 50.0% |
| sliding window, w=128 | 11.8% |
| block-sparse, block=64, n_global=8, n_random=3 | 17.8% |

Causal is 50.0% by construction — `(N+1)/2N`. Sliding window checks out by
hand: row `i` allows `min(i, 128) + 1` positions, which integrates to 11.8%
over 1024 rows. I found it useful that these are predictable; a sparsity
ratio that disagrees with the arithmetic is a mask bug.

Mask figures are in `results/`. Building the visualiser early was the single
most useful thing I did in this phase — a mask bug is invisible in a tensor
and obvious in an image.

---

## 3. Why global tokens matter disproportionately

With global tokens, **any two positions are two hops apart**: A attends to a
global token, the global token attends to B. This holds regardless of the
distance between A and B. Without them, the shortest path between distant
tokens is O(N/w) hops, so the graph diameter grows with sequence length and a
fixed-depth model cannot cover it.

Eight tokens out of 1024 — 0.8% of the sequence — collapse the diameter from
O(N/w) to 2. That is the disproportion.

**An observation from my own implementation.** Global tokens are set in two
directions: the first `n_global` tokens attend to everyone (rows), and
everyone attends to them (columns). Under causal masking, **the rows do
almost nothing.** Token 3 can only look backwards, and there is nothing
behind it. The columns — everyone attending to the first few tokens — carry
the entire effect.

This falls directly out of the ordering in `block_sparse_mask`: the row term
`idx.unsqueeze(1) < n_global` is set, then largely cancelled by the causal AND
applied afterwards. I did not expect that when I wrote it.

Relatedly, the causal AND has to come **last**, after all three components are
ORed. The global rows set entries above the diagonal to True; if causal were
applied first, those would survive and a query could attend to its own future.
Causal is a hard constraint on the other three rules, not a fourth rule
alongside them.

---

## 4. The NaN case

**The mechanism.** If every entry in a row of the score matrix is `-inf`,
softmax computes `exp(-inf) / sum(exp(-inf))` = `0 / 0` = NaN. One NaN
propagates through every subsequent operation.

**The exact configuration that triggered it:** `seq_len=128`,
`window_size=16`, with a strictly-backward causal window — allowed set
`i - w <= j < i`, excluding the query's own position. **Row 0** is the empty
row: causal requires `j <= i`, the window requires `j >= i - 16`, and the
strict inequality requires `j < i`. For `i = 0` the intersection is empty.

**Honest detail: my three shipped patterns never produce this.** Running
`masks.py` prints `min row 1` for all of them, because the local component
always includes a block's own block and the causal AND never removes the
diagonal. Every query can at minimum attend to itself. I had to construct the
strictly-backward variant deliberately to find the boundary.

I think that is worth stating rather than dressing up. The task describes this
as happening "for real at block boundaries", and in the specific patterns I
built, it does not. What I can say is where it does arise and why: any rule
that excludes self-attention, or defines local blocks without the diagonal
block, produces empty rows at the start of the sequence.

**The fix.** After softmax, zero the probability rows for queries with no
allowed positions:

```python
probs = probs.masked_fill(~mask.any(-1, keepdim=True), 0.0)
```

`mask.any(-1)` asks whether a query row has any allowed position at all;
`keepdim=True` lets it broadcast across the key axis. A query with nothing to
attend to then produces a zero vector, which is the right semantics — it
contributes nothing rather than poisoning the layer.

**Evidence the fix is surgical.** The 36-case correctness harness passes
unchanged before and after. None of those masks contain empty rows, so
`mask.any(-1)` is True everywhere and the extra `masked_fill` is a no-op. I
also assert that non-empty rows still produce non-zero output, because a
broken version of this check that zeroed everything would pass a NaN test.

The commit history reflects the order: the failing test is committed before
the fix.

---

## 5. Benchmark results

**Hardware.** Tesla T4, 14.56 GB, CUDA 12.8, PyTorch 2.11.0+cu128, Python
3.13.15, Linux. Batch 1, 8 heads, head_dim 64, forward pass only. Five warmup
iterations, median of ten timed runs, `torch.cuda.synchronize()` around each.

Plots: `results/benchmark_time.png`, `results/benchmark_memory.png`. Raw data
in `results/benchmark.json`.

### The result

| seq_len | time (ms) | peak memory (MB) |
|---|---|---|
| 512 | 0.84 | 35.9 |
| 1024 | 2.52–2.67 | 113.1 |
| 2048 | 9.30–9.74 | 416.1 |
| 4096 | 36.1 | 1616.1 |
| 8192 | 148.2–148.8 | 6392.1 |

These are the numbers for **all three patterns**. Peak memory is
byte-for-byte identical; wall-clock differs only within run-to-run noise.

There is no crossover point. Sparse is not faster than dense at any sequence
length I tested, and it does not become faster at larger ones.

### Why

Sparsity here is implemented **by masking**. The full `(N, N)` score matrix is
computed, then disallowed entries are overwritten with `-inf` and discarded by
the softmax. Nothing is skipped. The masked positions cost exactly as much to
compute and store as the unmasked ones.

Saving anything requires not materialising those entries in the first place —
gathering only the needed blocks, or fusing the whole operation in a custom
kernel so the score matrix never reaches HBM. That is a different
implementation, and it is what task 4 of this set is about. The masking
approach gives correct sparse *semantics* and zero sparse *performance*.

This bounds what any benchmark of this implementation can show, and I would
rather state it than present the flat lines as a puzzling result.

### Measured scaling

From 4096 to 8192:

- memory: 1616.1 → 6392.1 = **3.96×**, exponent **1.98**
- time: 36.13 → 148.77 = **4.12×**, exponent **2.04**

Both ≈ 2, computed from my own measurements rather than asserted from theory.

It is not quadratic at the small end: 512 → 1024 gives only 3.15× for memory.
At short sequence lengths the linear terms — q, k, v, the output, the mask
itself — still contribute meaningfully. The N² term only dominates from about
2048 upward.

### Peak memory decomposes exactly

At seq_len 8192, one score matrix is `8 heads × 8192² × 4 bytes` = **2048 MB**.
Measured peak is **6392 MB**, which is `3 × 2048 + ~250 MB`.

Three, because three full-size tensors are alive simultaneously: `scores` from
the matmul, the new tensor returned by `masked_fill`, and the softmax output.
None of these operations are in-place.

That is a concrete, fixable inefficiency. Using `masked_fill_` and
`softmax(..., out=)` would cut peak memory by roughly a third — still O(N²),
but with a constant of 1 rather than 3. I did not implement this; I found it
by checking whether the measured number matched the theoretical one, and it
did not until I accounted for the intermediates.

### One deviation from the brief

The task anticipated dense might run out of memory at 8192 on a T4. It did
not — 6.4 GB against 14.56 GB available. I note this rather than omitting it.

---

## 6. Quality evaluation

Not attempted. See section 7.

---

## 7. What I did not get to, and why

**Deliverable 1.6 — quality evaluation.** A 2-layer character-level GPT on
TinyShakespeare, trained three times with identical hyperparameters and seed,
reporting comparative loss. Cut for time.

What I would have expected, stated as reasoning rather than a result: **sparse
worse than dense.** TinyShakespeare training uses contexts of a few hundred
tokens, which sits entirely inside a single 128-wide window plus its
neighbours. At that length, sparsity discards real information while the
O(N²) cost it is meant to save is not yet significant — section 5 shows the
quadratic term only dominates above roughly 2048. Sparse attention is a
long-context technique; evaluating it at short context measures the cost
without the benefit.

I want to be clear that this is an argument, not a measurement. I did not run
it and I am not claiming the number.

**Stretch goals.** No third pattern, no per-head pattern mixing.

**The limitation underneath all of it.** Everything here is masking, not true
sparsity. The correctness harness proves the masks are constructed correctly
and that the attention computed under them matches an independent reference.
It cannot prove a speedup exists, because there isn't one. Making this fast
means gathering blocks or writing a fused kernel, and that is the piece I did
not build.

---

## 8. What I learned

Three things I did not know when I started and would now defend:

**The scale factor is not cosmetic.** `1/sqrt(d)` exists because the dot
product of two `d`-dimensional vectors with unit-variance entries has variance
`d`. At `d=64`, unscaled scores swing over roughly ±8, softmax saturates to
near one-hot, and the gradient with respect to the scores goes to zero. The
model then cannot learn what to attend to. It is a variance-control step, not
a normalisation convention.

**Masking has to happen before softmax, not after.** Zeroing a probability
after softmax leaves the remaining weights not summing to 1, which silently
scales the output down. Setting the score to `-inf` removes the term from the
numerator and the denominator simultaneously, so the survivors renormalise
correctly among themselves. This is also exactly why the fully-masked row
produces `0/0`.

**The interesting question about a sparse pattern is its graph diameter, not
its sparsity ratio.** Sliding window at 11.8% and block-sparse at 17.8% are
not 6 percentage points apart in any way that matters. One has a diameter
growing linearly with sequence length and the other has a diameter of 2. That
is the whole difference, and the sparsity ratio does not show it.
