# Source

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
| `metrics.py` | Turns a profile + a `FusionConfig` into numbers: accuracy, ECE, the oracle bound, negative/positive transfer (conditional rates: broken / originally right, fixed / originally wrong - multiply by the original accuracy / error rate to compare them). |
| `splits.py` | The four corruption families, the deterministic tune/test image split (clean images mapped to their ImageNet-C id via `data/clean_name_map.json`), and `EXTRA_FAMILY` - the 4 held-out extra corruptions, kept out of `CORRUPTION_FAMILY` so no tuner ever sees them. |
| `cache.py` | Everything that touches the GPU or disk - image geometry, building spectral and spatial views (corner crops: 196 px resized to 224 by default, or the reflect-padded `mode="pad"` control), running the models, reading/writing the `.npz` cache, and the virtual combined cache. |
| `models.py` | Loads the three frozen classifiers (ResNet-18, DeiT-Tiny, ConvNeXt-Tiny). |
| `evaluation.py` | GPU scoring of many fusion configs from one pairwise-JSD pass per condition; same definitions as `metrics.py` (cross-checked in `test/test_evaluation.py`). |
| `ce_fusion.py` | Cross-entropy fitting of the fusion coefficients (and optional per-view bias) with L-BFGS-B, NumPy and torch backends. |
| `bootstrap.py` | Paired cluster bootstrap over images for mean-over-conditions accuracy deltas. |
| `constants.py` | Shared numerical constants (probability floor). |
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
or `metrics.precompute`) and everything at or past that index gets the
image's mean ladder `S`, so after the per-image z-score it sits at exactly 0
and is weighted by confidence/agreement only. Leave it at the default `None`
for a single ladder (the spectral-only view set); `ladder_len=1` (only the
original on the "ladder", as for the spatial-only view set) gives `S = 0`
for every view.
