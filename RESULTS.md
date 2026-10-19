# Results

The full story behind the numbers in the main README - every baseline, the
diagnosis that led there, and the analysis that closes it out. See the
[main README](README.md) for the project overview and quick start. The
tables below are also plotted: run `python scripts/make_report_figures.py`
for a local `figures/` folder (not tracked in git - regenerate it whenever
you need it).

100-class ImageNet-C, held-out image half, leave-one-family-out configs, mean
over 15 corruptions x 5 severities throughout. All deltas below have
paired-bootstrap 95% CIs (`scripts/bootstrap_ci.py`, 2000 resamples within
each of the 75 conditions) - not just point estimates.

## Headline: combining view families beats either alone

The single biggest lever we found: fuse across **both** a spectral low-pass
ladder *and* a spatial-TTA ensemble (crop + flip) in one 15-view profile,
instead of either alone. Both source profiles were already cached, so this
cost zero additional GPU time - `scripts/build_combined_cache.py` just
concatenates them (`cache_combined/`), and the stability signal is made
ladder-aware (`analysis.stability_from_d(..., ladder_len=8)`) so it only
looks for "neighbouring filter strength" within the ordered spectral block,
leaving the unordered spatial views to be weighted by confidence/agreement
only.

| metric | ResNet-18 spectral-only | ResNet-18 spatial-only | **ResNet-18 combined** | DeiT-Tiny spectral-only | DeiT-Tiny spatial-only | **DeiT-Tiny combined** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| original view | 32.6% | 32.6% | 32.6% | 44.0% | 44.0% | 44.0% |
| fused / TTA | 33.3% | 34.4% | **34.9%** | 44.2% | 45.6% | **45.9%** |
| vs. original | +0.7 pts | +1.9 pts | **+2.4 pts** | +0.2 pts | +1.6 pts | **+2.0 pts** |
| vs. spatial-only (95% CI) | - | - | **+0.47 [+0.39,+0.56]** | - | - | **+0.39 [+0.31,+0.48]** |
| clean (orig -> fused) | 68.9->67.8 | 68.9->70.4 | **68.9->70.1** | 70.1->69.2 | 70.1->71.4 | **70.1->71.0** |
| per-image oracle | 41.0% | - | 48.3% | 52.4% | - | 58.9% |
| inference overhead | 8.8x | 8.8x | **16.0x** | 8.3x | 8.3x | **15.2x** |

Every "vs." delta's 95% CI excludes 0 (`P(delta <= 0) = 0.000` for all four in
`results/bootstrap_ci_*.json`) - these aren't point-estimate noise. The
combined profile beats *both* single-family baselines on both architectures,
and is the only variant that improves clean accuracy over the plain
classifier rather than costing some (spatial views add clean-image diversity
for free; the low-pass views alone always cost clean accuracy since they
throw away real detail). The honest cost: overhead roughly doubles (15 forward
passes instead of 8), and ResNet-18's calibration gets worse under the
combined profile (ECE 0.084 -> 0.117, vs. 0.084 -> 0.098 spectral-only) -
DeiT's still improves (0.114 -> 0.106). Build/reproduce:

```bash
python scripts/build_combined_cache.py                              # merge cache_fixed + cache_spatial, no GPU
python scripts/tune_fusion.py --cache-dir cache_combined --ladder-len 8
python scripts/evaluate.py    --cache-dir cache_combined --ladder-len 8 --tuning auto
python scripts/bootstrap_ci.py --cache-dir cache_combined --tag cache_combined --ladder-len 8
```

### Is the 16x overhead necessary? A cheaper variant, tried and found wanting

The combined profile's honest cost is real - 15 forward passes per image.
Obvious question: does most of the gain come from just *having* a spatial
view at all, or does it need the full 4-crop diversity? Tested a "lite"
9-view variant (spectral ladder + hflip only, dropping the 4 corner crops -
`scripts/build_combined_cache.py --spatial-views 1 --out-dir cache_combined_lite`,
still zero new GPU time):

| | ResNet-18 spectral-only | ResNet-18 lite (9v) | ResNet-18 spatial-only | **ResNet-18 combined (15v)** | DeiT-Tiny spectral-only | DeiT-Tiny lite (9v) | DeiT-Tiny spatial-only | **DeiT-Tiny combined (15v)** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| fused / TTA | 33.3% | 34.1% | 34.4% | **34.9%** | 44.2% | 44.6% | 45.6% | **45.9%** |
| vs. original | +0.7 pts | +1.5 pts | +1.9 pts | **+2.4 pts** | +0.2 pts | +0.6 pts | +1.6 pts | **+2.0 pts** |
| vs. spatial-TTA (95% CI) | -1.16 [-1.28,-1.03] | -0.38 [-0.49,-0.27] | - | **+0.47 [+0.39,+0.56]** | -1.38 [-1.50,-1.26] | -0.95 [-1.06,-0.84] | - | **+0.39 [+0.31,+0.48]** |
| clean (orig -> fused) | 68.9 -> 67.8 | 68.9 -> 68.8 | 68.9 -> 70.4 | **68.9 -> 70.1** | 70.1 -> 69.2 | 70.1 -> 69.6 | 70.1 -> 71.4 | **70.1 -> 71.0** |
| inference overhead | 8.8x | 10.1x | 8.8x | 16.1x |8.2x | 9.3x | 8.2x | 15.1x |

The honest answer: **no, one extra view isn't enough - the lite variant is a
real improvement over spectral-only (+1.5/+0.6 pts, both p<0.001) but still
loses to plain spatial TTA alone** (-0.38/-0.95 pts, both 95% CIs entirely
negative, p<0.001). It also doesn't get the "free" clean-accuracy bonus the
full combined profile has (68.8%/69.6% vs. the 68.9%/70.1% original - flat to
slightly worse, not better). Crop diversity, not merely "some spatial
information," is what the full combined profile is actually using - a single
flipped view gives the reliability signals almost nothing to arbitrate
between. If overhead is a hard constraint, plain spatial TTA alone (8.8x/8.2x,
no spectral views at all) is a better choice than this cheaper fusion
variant; the full 15-view combined profile is the only one that both beats
spatial TTA and improves clean accuracy. Reproduce:

```bash
python scripts/build_combined_cache.py --spatial-views 1 --out-dir cache_combined_lite
python scripts/tune_fusion.py --cache-dir cache_combined_lite --ladder-len 8
python scripts/evaluate.py    --cache-dir cache_combined_lite --ladder-len 8 --tuning auto
python scripts/bootstrap_ci.py --cache-dir cache_combined_lite --tag cache_combined_lite --ladder-len 8
```

## How we got here: the spectral-only story

Two view families are compared: **energy** (the original design, cutoffs
solved per image to keep a target fraction of that image's spectral energy)
and **fixed** (same absolute cutoffs for every image). The energy ladder's
"mild" cutoffs are only mild for high-frequency corruptions - a blur/fog
image's spectrum is already concentrated near DC, so "keep 99% of energy"
is satisfied by an aggressive, narrow mask, solved cutoff 0.12 for
defocus_blur vs 0.66 for gaussian_noise at the *same* nominal target. Fixed
cutoffs (`FIXED_CUTOFFS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12)` in
`src/cache.py`) use the same absolute filter regardless of the image's own
spectral shape:

| metric | ResNet-18 energy | ResNet-18 fixed | DeiT-Tiny energy | DeiT-Tiny fixed |
|---|:--:|:--:|:--:|:--:|
| original view | 32.6% | 32.6% | 44.0% | 44.0% |
| fused (tuned) | 31.6% | **33.3%** | 40.8% | **44.2%** |
| fused - original | -0.9 pts | **+0.7 pts** | -3.2 pts | **+0.2 pts** |
| per-image oracle | 39.3% | 41.0% | 50.6% | 52.4% |
| clean (orig / fused) | 68.9 / 64.7 | 68.9 / 67.8 | 70.1 / 66.1 | 70.1 / 69.2 |
| mean negative transfer | 13.7% | 8.2% | 14.5% | 6.2% |

- **The energy ladder does not work; fixed cutoffs do.** Under `energy`, tuning
  collapses to confidence-only (beta = gamma = 0, near-hard view selection) on
  both models - agreement and stability carry no signal. Under `fixed`, every
  fold uses stability too (and DeiT uses agreement), fusion actually beats the
  baseline for ResNet-18, and negative transfer is roughly halved on both
  models. Confirmed this was the real bottleneck, not the fusion coefficients:
  tuning *per corruption* (maximum possible specialisation) gave the same
  numbers as family-level tuning - no amount of re-weighting fixes a bad view
  family.
- **Corruption-dependent, as predicted.** Under `fixed`, fusion clearly wins on
  the three noise types and pixelate (+6 to +9 points), and is close to flat on
  most blur corruptions; weather and contrast/jpeg/elastic remain small net
  losses.
- **Architecture-dependent, as predicted.** ResNet-18 gets a real net win from
  low-pass views; DeiT-Tiny gets a small one. Consistent with DeiT's
  self-attention already acting like a low-pass filter - it starts more robust
  (44.0% vs 32.6% mean corrupted) and has less to gain from an external one.
- **The proposal's pre-registered agreement ablation is a real, if modest, win
  for DeiT.** `A_k` as originally defined averages JS-divergence to *every*
  other view, so a cluster of heavily-filtered views that agree with each
  other (but are all wrong) can look reliable. Anchoring agreement to view 0
  only (`analysis.agreement_to_origin`, searched in `tune_fusion.py` alongside
  `m`) correlates better with per-view correctness than the original
  definition (e.g. +0.355 vs +0.281 for DeiT/weather) and is selected in most
  LOFO folds once it's an option. Held-out effect: DeiT's fused accuracy goes
  from flat (+0.0 pts, beta tuned to 0 - agreement was inert) to +0.2 pts with
  beta up to 1.0, and negative transfer drops from 7.4% to 6.2%. ResNet-18 is
  roughly flat (+0.8 -> +0.7 pts) - it was already using stability more than
  agreement, so this signal had less to fix there.

Build and evaluate the fixed ladder yourself:
```bash
python scripts/cache_profiles.py --ladder fixed --cache-dir cache_fixed
python scripts/tune_fusion.py --cache-dir cache_fixed
python scripts/evaluate.py --cache-dir cache_fixed --tuning auto
```

## Baselines and overhead (spectral-only, fixed ladder)

All the baselines the proposal committed to, held-out half, mean over 15
corruptions x 5 severities:

| metric | ResNet-18 | DeiT-Tiny |
|---|:--:|:--:|
| original view | 32.6% | 44.0% |
| single fixed mildly-filtered view (no fusion) | 32.8% | 43.7% |
| equal-weight average of the K+1 views | 33.3% | 43.6% |
| confidence-only weighting | 33.3% | 43.5% |
| agreement-only weighting (per-corruption best definition) | 32.7% | 43.9% |
| **spatial TTA, same view count (crop+flip, equal avg)** | **34.4%** | **45.6%** |
| **fused (all three signals, tuned)** | **33.3%** | **44.2%** |
| per-image oracle | 41.0% | 52.4% |
| inference overhead (K+1 views vs. 1) | 8.8x | 8.3x |

The honest reading, updated after adding the agreement-to-origin ablation
(above): for **DeiT-Tiny**, fusion now clearly beats every same-family
baseline (equal, confidence-only, agreement-only, the single mild view). For
**ResNet-18**, fusion is statistically tied with equal-averaging and
confidence-only (33.3% all three) - the reliability signals aren't adding
resolving power beyond "average the views and lean on confidence" for this
architecture, even though the *fused* config still gets meaningfully lower
negative transfer than a naive equal-weight scheme would (see the main table).
On both architectures, a **plain spatial-TTA ensemble (flip + 4 corner crops,
same 8-view budget, `scripts/cache_spatial_profiles.py` +
`scripts/evaluate_tta.py`) still beats spectral-only fusion on raw
mean-corrupted accuracy** - the spectral profile's advantage over spatial TTA
isn't "any 8-view ensemble beats 1 view" (true of spatial TTA too), it's the
*lower negative-transfer rate* and the per-corruption story (fusion clearly
wins on noise/pixelate; spatial TTA is flatter across the board, see the
per-view accuracy curve below for why no amount of low-pass filtering can beat
the original on blur/weather/digital). This gap is exactly what motivated the
combined-profile experiment at the top of this document, which lets the same
fusion mechanism see both view families and beats spatial TTA outright - so
read this table as the diagnostic step, not the final word.

Per-family, per-view accuracy makes the mechanism concrete (ResNet-18, fixed
ladder, held-out half; view k's cutoff shown in parens; see
`figures/fig_per_view.png` for both architectures plotted):

| family | v0 (orig) | v1 (.9) | v2 (.75) | v3 (.6) | v4 (.45) | v5 (.3) | v6 (.2) | v7 (.12) |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| noise   | 22.9 | 24.4 | 24.9 | 25.6 | 26.3 | **27.0** | 26.1 | 22.0 |
| blur    | **28.3** | 28.3 | 28.2 | 28.1 | 27.8 | 26.8 | 24.7 | 18.2 |
| weather | **36.1** | 35.5 | 35.2 | 34.6 | 33.3 | 29.9 | 24.8 | 15.1 |
| digital | 40.5 | **41.0** | 41.0 | 40.9 | 40.7 | 39.0 | 34.8 | 23.9 |

Noise is the only family where filtering ever beats the original, and its
peak (view 5) is one rung short of the ladder's deepest cutoff - view 7 is a
net loss for *every* family, including noise. Blur/weather/digital decline
from view 0 onward with no exception: structurally, low-pass filtering cannot
recover detail a blur/weather corruption already removed, so the best any
per-image weighting scheme can do for those three families is converge to
"trust view 0" - which is a ceiling equal to the plain classifier, not an
improvement. The oracle's 41.0% (vs. fused 33.3%) headroom is real but lives
almost entirely in the noise/pixelate corner of this table.

## Is the 8-point oracle gap a tuning problem or an information problem?

`tune_fusion.py` searches a coarse grid over one specific parametric family
(linear combination + softmax). `scripts/signal_ceiling.py` asks a sharper
question: given exactly C_k/A_k/S_k, what's the best *any* scorer could do -
linear or not? It fits a per-fold (same LOFO discipline) classifier on
"is view k correct" from the three signals, then fuses using its predicted
reliability instead of the hand-tuned linear rule.

| | ResNet-18 | DeiT-Tiny |
|---|:--:|:--:|
| grid-tuned fused (spectral-only) | 33.3% | 44.2% |
| logistic-regression ceiling (linear) | 33.3% | 43.9% |
| gradient-boosted-trees ceiling (nonlinear, checked on a subset) | ties logistic | - |
| per-image oracle | 41.0% | 52.4% |

Both ceiling estimates land within 0.3 pts of the grid-tuned result - for
ResNet-18 they're identical. Gradient-boosted trees (checked on the severity-3
subset, `--no-gbm` skips it for the full run since it changed nothing)
matched logistic regression exactly there too. **Conclusion: the 8-point gap
to the oracle is not a tuning-procedure problem** - a properly-optimized
linear rule and a nonlinear one both land in the same place as the coarse
grid search. Closing it needs different information, not a better combination
rule - which is exactly what the combined-profile result above provides (a
second, structurally different view family), rather than a smarter fusion of
the same three signals over the same views. Reproduce:
`python scripts/signal_ceiling.py --no-gbm --agreement origin`.

## Where to push next

- a milder or differently-spaced `FIXED_CUTOFFS`, calibrated per family;
- **band-pass** views instead of low-pass, so mid frequencies survive;
- more views, or views chosen per image - though a cheaper *spatial* subset
  specifically was tried and didn't work (hflip-only still loses to plain
  spatial TTA, see above) - a middle ground like 2-3 crops might be worth a
  second look;
- ResNet-18's calibration gets worse under the combined profile (ECE 0.084 ->
  0.117) even though accuracy improves - worth understanding before claiming
  a calibration win the way the spectral-only result could;
- weather and contrast/jpeg/elastic remain small net losses even under the
  fixed ladder;
- a third architecture, or ImageNet-C's unseen "extra" corruptions, for a
  stronger generalisation claim.

Everything after the caching scripts runs in seconds to minutes, so all of
the above is cheap to explore.
