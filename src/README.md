# `src/`

The library half of the project: pure functions, no CLI flags, no GPU
orchestration, no file I/O beyond the cache itself (that lives in `cache.py`
by design - everything else here is easy to unit-test and import from a
notebook). If you're trying to understand *how* the method works rather than
*how to run* it, start here; for the entry points that actually call this
code, see [`../scripts/`](../scripts/README.md); for what it all produced,
see [`../RESULTS.md`](../RESULTS.md).

| module | what's in it |
|---|---|
| `filters.py` | The low-pass mask + the per-image energy-adaptive cutoff search (`torch.fft`, single image). |
| `analysis.py` | **The profile maths** - confidence, agreement, and stability (`C_k`/`A_k`/`S_k`), softmax fusion. One implementation; everything downstream calls into it rather than recomputing. |
| `metrics.py` | Turns a profile + a `FusionConfig` into numbers: accuracy, ECE, the oracle bound, negative/positive transfer. |
| `splits.py` | The four corruption families and the deterministic tune/test image split. |
| `cache.py` | Everything that touches the GPU or disk - building views, running the models, reading/writing the `.npz` cache. |
| `models.py` | Loads the two frozen classifiers. |
| `imagenet_labels.py` | Image path -> true ImageNet-1k class index. |

## Two things worth knowing before you dive in

**`fused_probs()` is split out of `score()`** in `metrics.py` so anything that
needs the fused *predictions* - not just the accuracy number - can get them
without re-deriving the reliability/softmax steps by hand
(`scripts/bootstrap_ci.py` is why this exists).

**`ladder_len` is the trick that makes the combined profile work.** Stability
(`S_k`) means "how much does this view agree with its neighbours in the
filter ladder" - which only makes sense for the *ordered* spectral views, not
for the spatial-TTA crop/flip views sitting next to them in a combined
profile. Pass `ladder_len` (to `analysis.stability_from_d`, `profile_signals`,
or `metrics.precompute`) and everything at or past that index gets `S_k = 0`
instead, falling back to confidence/agreement only. Leave it at the default
`None` and you get the original single-ladder behaviour - nothing about the
spectral-only or spatial-only caches changed when this was added.
