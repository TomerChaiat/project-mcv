# Results

## Setup

- **Data:** 100-class subset of ImageNet-C, 15 corruptions x 5 severities,
  plus the matching clean images. Images are split in two halves: one for
  tuning, one for testing. All numbers below are on the **test half**.
- **Models:** ResNet-18, DeiT-Tiny, ConvNeXt-Tiny (frozen, ImageNet-1k).
- **View sets:**
  - **spectral (8 views):** original + 7 low-pass views, cutoffs
    `(0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12)`;
  - **spatial TTA (8 views, baseline):** original, horizontal flip, 4 corner
    crops (196 px, resized to 224), 2 flipped corners; used both as a
    *plain average* and *tuned* exactly like the spectral views (same
    fusion methods, tuning and selection);
  - **combined (15 views):** spectral + the 7 crop/flip views.
- **Tuning:** leave-one-corruption-family-out on the tuning half. Each
  corruption is scored with a configuration tuned without its family. The
  fusion method per model is chosen on the tuning half.
- **Metric:** mean accuracy over the 75 corruption x severity conditions.
  Brackets are 95% confidence intervals from a paired bootstrap over images.
- Outputs: `results/`; figures and LaTeX tables: `figures/`.

## Summary

1. With both view sets tuned the same way (8 vs 8), low-pass views beat
   crop/flip views for the noise-sensitive ResNet-18 and tie with them for
   DeiT and ConvNeXt. Combining both kinds of views (15) is best for all
   three models.
2. Low-pass views help on noise and on pixelation/JPEG artefacts, not on
   blur or weather. No single spectral measure of the corruption explains
   every case.
3. How much they help depends on the model's sensitivity to noise, not on
   CNN vs. Transformer.
4. A fixed weight per view does about as well as per-image trust signals,
   except for ResNet-18 (+0.46 / +0.49 pts over fixed weights, spectral /
   combined).
5. On 4 unseen corruption types, the benefit holds for noise only.
6. The alternatives tried - band-pass views, notch views, a per-image
   energy-adaptive ladder, a fourth trust signal - do not beat the method's
   choices overall; notch views do better than low-pass on digital
   corruptions only.
7. Four views (the original, a mid low-pass view, a flip and one more) give
   80-98% of the 15-view gain; fitted weights only clearly help with all 15.
8. Of the three per-image signals, stability alone gives ResNet-18 the full
   per-image gain over fixed weights (+0.51 pts); for DeiT and ConvNeXt no
   combination of signals beats fixed weights.

## 1. Main comparison

| | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|:--:|:--:|:--:|
| original | 33.88 | 45.75 | 59.37 |
| spatial TTA, plain average (8 views) | 35.11 | 46.65 | 59.84 |
| spatial TTA, tuned (8 views) | 35.49 | 47.17 | 60.20 |
| spectral, tuned (8 views) | 35.86 | 47.01 | 60.10 |
| **combined fused (15 views)** | **36.60** | **47.63** | **60.57** |
| **spectral − spatial TTA, both tuned (8 vs 8)** | **+0.37** [+0.15, +0.59] | −0.16 [−0.35, +0.05] | −0.10 [−0.25, +0.05] |
| spectral (tuned) − spatial TTA (plain average) | +0.75 [+0.50, +1.01] | +0.36 [+0.10, +0.62] | +0.26 [+0.05, +0.46] |
| spectral − original | +1.98 [+1.78, +2.18] | +1.26 [+1.07, +1.47] | +0.73 [+0.61, +0.86] |
| combined − original | +2.73 [+2.48, +2.96] | +1.88 [+1.66, +2.11] | +1.20 [+1.04, +1.35] |
| **combined − spatial TTA, tuned** | **+1.11** [+0.94, +1.29] | **+0.46** [+0.33, +0.60] | **+0.36** [+0.28, +0.45] |
| combined − spatial TTA, plain average | +1.49 [+1.31, +1.67] | +0.98 [+0.80, +1.16] | +0.72 [+0.58, +0.85] |
| combined − plain average of the same 15 views | +0.29 [+0.21, +0.37] | +0.20 [+0.10, +0.31] | +0.18 [+0.08, +0.28] |
| clean accuracy: original -> combined | 68.6 -> 69.0 | 70.3 -> 70.3 | 79.8 -> 79.9 |
| clean accuracy: original -> spectral | 68.6 -> 67.7 | 70.3 -> 70.0 | 79.8 -> 79.7 |
| worst corruption: original -> combined | 18.95 -> 25.26 | 30.25 -> 32.16 | 36.89 -> 37.95 |
| images fixed / broken, % of all (combined)¹ | 5.30 / 2.57 | 4.37 / 2.49 | 2.85 / 1.65 |
| inference cost (combined / spectral) | 16.3x / 8.8x | 15.4x / 8.3x | 15.2x / 8.0x |

¹ Fixed = wrong before, right after fusion; broken = the reverse; both as a
share of all test images, so fixed − broken is the accuracy gain.

**Conclusion:** the fair comparison is both 8-view sets tuned the same way.
There, low-pass views beat crop/flip views for ResNet-18, the
noise-sensitive model, and tie with them for DeiT and ConvNeXt (both CIs
include 0). Against a plain-average crop/flip baseline the low-pass views win
on all three models, but part of that margin is the tuning, not the views.
The combined set is best for every model: it beats the tuned crop/flip views
and a plain average of its own 15 views. Gains shrink as the model gets more
robust (ResNet-18 > DeiT > ConvNeXt). Clean accuracy is unchanged with the
combined set.

## 2. Baselines

Gain over the original, combined view set
(`figures/baselines_oracle_combined.pdf`):

| | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|:--:|:--:|:--:|
| single mild low-pass view (0.90) | +0.96 [+0.84, +1.09] | −0.03 [−0.20, +0.13] | +0.26 [+0.11, +0.41] |
| spatial TTA, plain average (8 views) | +1.23 [+1.02, +1.44] | +0.90 [+0.64, +1.17] | +0.48 [+0.27, +0.68] |
| spatial TTA, tuned (8 views) | +1.62 [+1.45, +1.78] | +1.42 [+1.24, +1.59] | +0.84 [+0.70, +0.97] |
| plain average, 15 views | +2.43 [+2.21, +2.65] | +1.68 [+1.39, +1.96] | +1.02 [+0.81, +1.23] |
| confidence-only weighting | +2.66 [+2.41, +2.91] | +1.14 [+0.83, +1.46] | +0.39 [+0.16, +0.62] |
| agreement-only weighting | +1.80 [+1.61, +1.99] | +1.19 [+0.95, +1.43] | +0.96 [+0.77, +1.15] |
| **fused (selected method)** | **+2.73** [+2.48, +2.96] | **+1.88** [+1.66, +2.11] | **+1.20** [+1.04, +1.35] |
| oracle: any of the 15 views correct | +16.73 | +16.01 | +12.57 |

**Conclusion:** the fused predictor is the best realizable option on every
model. The oracle shows how often *some* view is right; it is an upper bound,
not an achievable target.

## 3. Where low-pass views help

Best single low-pass view minus the original, by corruption family
(`figures/per_view_accuracy.pdf`):

| family | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|:--:|:--:|:--:|
| noise | **+7.3** | **+2.8** | **+0.9** |
| digital | +1.2 | +1.1 | **+2.2** |
| blur | +0.3 | −0.8 | −0.6 |
| weather | −0.4 | −1.4 | −0.7 |

Gain of the combined fused predictor over the original, by family:

| family | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|:--:|:--:|:--:|
| noise | +7.08 | +4.32 | +1.33 |
| blur | +1.50 | +0.54 | +0.46 |
| weather | +0.57 | −0.28 | +0.37 |
| digital | +2.84 | +3.56 | +2.67 |

**Conclusion:** low-pass views help on noise and on digital corruptions
(pixelate, JPEG). On blur and weather no low-pass view beats the original;
the combined set's gains there come from the crop/flip views.

### Does the corruption's spectrum predict the gain?

For each corruption x severity, corrupted test images were paired with
their clean originals, and three quantities were measured in the band each
view filters out (`scripts/spectral_energy.py`, 500 images per condition):

- **corruption energy removed:** the share of the corruption's energy
  (corrupted minus clean) that the view filters out;
- **image energy change:** the change in the image's total energy in that
  band (> 0: more energy than the clean image, < 0: less);
- **energy added:** the energy the corruption added in that band, counted
  frequency by frequency (only where the corrupted image has more energy
  than the clean one), relative to the clean image's energy there.

Spearman correlation with the LP 0.75 view's accuracy gain: across all 75
conditions of the 15 corruptions, within each corruption family (mean over
the four families), and across the 20 conditions of the 4 unseen corruptions
of section 7 (`figures/table_spectral_energy.tex`, `figures/spectral_energy_gain.pdf`;
other cutoffs in `results/spectral_energy*.json`):

| model | quantity | 15 corruptions | within family | 4 unseen |
|---|---|:--:|:--:|:--:|
| ResNet-18 | corruption energy removed | **+0.73** | +0.12 | +0.38 |
| | image energy change | +0.53 | +0.18 | +0.67 |
| | energy added | +0.60 | **+0.30** | **+0.71** |
| DeiT-Tiny | corruption energy removed | **+0.71** | −0.04 | +0.68 |
| | image energy change | +0.47 | +0.48 | **+0.80** |
| | energy added | +0.53 | **+0.55** | +0.65 |
| ConvNeXt-Tiny | corruption energy removed | **+0.38** | −0.06 | +0.10 |
| | image energy change | +0.17 | +0.15 | −0.08 |
| | energy added | +0.21 | **+0.21** | −0.04 |

Per corruption at LP 0.75 (energy removed / energy added; gain of the LP
0.75 view, ResNet-18 / DeiT / ConvNeXt):

| corruption | energy removed | energy added | gain |
|---|:--:|:--:|:--:|
| Gaussian / shot / impulse noise | 0.53-0.54 | 1.73-1.86 | +3.3 to +6.7 / +1.8 to +2.7 / +0.1 to +0.5 |
| pixelate | 0.46 | 0.19 | **+4.7 / +6.0 / +7.9** |
| JPEG | 0.38 | 0.10 | +1.1 / +0.9 / −0.02 |
| blur (4 corruptions) | 0.11-0.23 | 0.01-0.03 | −2.6 to +0.95 |
| snow / frost | 0.02 | 0.59 / 0.27 | −2.8 to +0.1 |

**Conclusion:** no single spectral measure explains every corruption.
"Corruption energy removed" ranks the 15 corruptions best (rho ~ +0.72 for
ResNet-18 and DeiT) and captures pixelate, but it mostly separates the
families (within a family it is about 0) and over-credits blur, whose
"corruption energy" is detail that was lost, not added. "Energy added"
captures the noise corruptions and holds within families, but misses
pixelate (0.19, below snow and frost, where the views do not help). For
DeiT, "image energy change" also holds within families (+0.48) and on the
unseen corruptions (+0.80). Taken together: low-pass views help on noise and
on pixelation/JPEG artefacts, not on blur or weather. "Energy added" was
defined after these results were seen, so it is exploratory. For ConvNeXt no
measure predicts much: it is robust to noise to begin with (64.7% accuracy
on the noise family, vs. 23.6% for ResNet-18), so there is little gain to
predict.

## 4. Model differences

How far each view's prediction drifts from the original's (mean JSD), test
half (`scripts/profile_shape.py`, `figures/profile_drift.pdf`):

| family | model | LP 0.9 | LP 0.6 | LP 0.3 | LP 0.12 |
|---|---|:--:|:--:|:--:|:--:|
| noise | ResNet-18 | 0.124 | 0.241 | 0.398 | 0.549 |
| | DeiT-Tiny | 0.038 | 0.085 | 0.167 | 0.338 |
| | ConvNeXt-Tiny | 0.037 | 0.077 | 0.164 | 0.332 |
| clean | ResNet-18 | 0.009 | 0.030 | 0.122 | 0.454 |
| | DeiT-Tiny | 0.007 | 0.021 | 0.081 | 0.327 |
| | ConvNeXt-Tiny | 0.008 | 0.026 | 0.088 | 0.363 |

**Conclusion:** on noisy images, ResNet-18's prediction changes 3x more than
the others' under mild filtering: its decisions depend on the noise band.
ConvNeXt (a CNN) and DeiT (a Transformer) behave almost identically. The
split is noise-sensitive vs. noise-robust, not CNN vs. Transformer. A likely
reason is the training recipe rather than the architecture: DeiT-Tiny and
ConvNeXt-Tiny were trained with strong data augmentation (RandAugment, Mixup,
CutMix, random erasing), ResNet-18 (torchvision's original weights) only with
random crops and flips.

## 5. Per-image weighting vs. fixed weighting

Mean corrupted accuracy of each fusion method on the same views (**bold** =
selected on the tuning half). *Bias only* = one fixed weight per view, no
per-image signal.

| model | views | plain avg. | grid | CE | CE + bias | CE unconstr. | bias only | selected − plain avg. | selected − bias only |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet-18 | spectral | 35.86 | 35.73 | 35.83 | **35.86** | 35.69 | 35.40 | +0.00 [−0.09, +0.09] | +0.46 [+0.38, +0.54] |
| | combined | 36.31 | 36.44 | 36.57 | 36.43 | **36.60** | 36.12 | +0.29 [+0.21, +0.37] | +0.49 [+0.38, +0.60] |
| DeiT-Tiny | spectral | 46.01 | 46.64 | 46.43 | 46.89 | 46.41 | **47.01** | +1.00 [+0.83, +1.16] | (selected) |
| | combined | 47.42 | 47.53 | 47.46 | 47.54 | 47.44 | **47.63** | +0.20 [+0.10, +0.31] | (selected) |
| ConvNeXt-Tiny | spectral | 59.56 | 59.94 | 59.90 | 60.05 | 59.90 | **60.10** | +0.54 [+0.39, +0.69] | (selected) |
| | combined | 60.39 | 60.10 | 60.31 | 60.29 | 60.26 | **60.57** | +0.18 [+0.08, +0.28] | (selected) |

A learned per-view correctness classifier on the three signals
(`scripts/signal_ceiling.py`) reaches 35.80 / 36.29 (ResNet-18, spectral /
combined), 46.32 / 47.52 (DeiT) and 59.90 / 60.48 (ConvNeXt), never above the
selected method.

Selected configuration (global value, range over the 4 folds; full values in
`figures/table_final_config.tex`):

| model | views | method | α (confidence) | β (agreement) | γ (stability) |
|---|---|---|:--:|:--:|:--:|
| ResNet-18 | spectral | CE + bias | 0.21 (0.18-0.32) | 0.27 (0.02-0.51) | 0.69 (0.18-1.05) |
| | combined | CE unconstrained | 0.22 (0.15-0.37) | −0.35 (−0.49 to −0.29) | 0.30 (0.19-0.40) |
| DeiT-Tiny | both | bias only | 0 | 0 | 0 |
| ConvNeXt-Tiny | both | bias only | 0 | 0 | 0 |

**Conclusion:** a fixed weight per view is selected in 4 of 6 cases; for
DeiT and ConvNeXt the gain comes entirely from which views are used and their
fixed weights. The per-image signals help only ResNet-18, by +0.46 / +0.49
pts (spectral / combined) over fixed weights. That gain comes from
adjusting the weights image by image, not from favouring some views
overall: averaged over all images, each view's weight stays between 0.057
and 0.077 (uniform = 0.067). In ResNet-18's combined fit the agreement
weight is negative (β = −0.35): the fit favours views that disagree with
the others, in line with section 10.3, where the learned weights track
how much a view adds to the ensemble.

## 6. Crop geometry check

The baseline's corner crops are 196 px, resized to 224, which slightly
smooths them. As a check, the (plain-average) spatial TTA was rebuilt with
no resizing: 224 windows of the image with a 16 px mirrored border
(`--corner-mode pad`, `scripts/spatial_crop_control.py`).

| | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|:--:|:--:|:--:|
| spatial TTA, resized crops (baseline), plain average | 35.11 | 46.65 | 59.84 |
| spatial TTA, padded crops, plain average | 33.87 | 45.16 | 58.24 |
| padded − resized | −1.24 [−1.44, −1.04] | −1.48 [−1.71, −1.24] | −1.60 [−1.82, −1.38] |
| spectral − spatial TTA, resized | +0.75 [+0.50, +1.01] | +0.36 [+0.10, +0.62] | +0.26 [+0.05, +0.46] |
| spectral − spatial TTA, padded | +1.99 [+1.72, +2.26] | +1.85 [+1.59, +2.10] | +1.86 [+1.64, +2.09] |

Single corner view minus the original: about −1.7 to −2.5 pts (resized) vs.
−2.8 to −4.3 pts (padded), on every corruption family and on clean images.
Spatial-TTA gain over the original on noise, resized / padded: +2.20 / −0.44
(ResNet-18), +1.48 / −0.23 (DeiT), +0.19 / −1.05 (ConvNeXt).

**Conclusion:** spectral views beat the plain-average spatial TTA with
either crop type. The
resized crops are the stronger baseline: the padded border is an artefact
the models never saw, and the crops lose the slight zoom. Part of the resized
baseline's gain on noise comes from the smoothing itself, so comparing
against it is conservative.

## 7. Unseen corruption types

ImageNet-C's 4 extra corruptions - speckle noise, Gaussian blur, spatter,
saturate - were never used in tuning. The frozen configurations were applied
unchanged, each corruption scored with the configuration of its family
(`scripts/evaluate_extra.py`, `figures/table_extra_corruptions.tex`).
Gains in points, ResNet-18 / DeiT-Tiny / ConvNeXt-Tiny:

| corruption | spectral − original | combined − original | spectral − spatial TTA (plain avg.) |
|---|---|---|---|
| speckle noise | **+6.88** / **+1.46** / **+0.42** | **+6.50** / **+2.28** / **+0.62** | **+5.03** / **+1.48** / **+1.11** |
| Gaussian blur | *−1.36* / *−1.00* / *−0.95* | *−0.79* / *−0.50* / −0.23 | *−0.76* / −0.32 / −0.35 |
| spatter | +0.00 / *−0.94* / *−0.58* | **+1.23** / +0.08 / −0.20 | *−1.31* / *−1.27* / −0.02 |
| saturate | *−0.71* / *−1.22* / *−0.45* | +0.07 / −0.20 / −0.10 | *−1.74* / *−0.70* / −0.03 |
| mean of 4 | **+1.20 [+0.91, +1.49]** / *−0.43 [−0.66, −0.17]* / *−0.39 [−0.55, −0.22]* | **+1.76 [+1.41, +2.09]** / **+0.41 [+0.10, +0.70]** / +0.02 [−0.18, +0.23] | +0.31 [−0.05, +0.67] / −0.20 [−0.57, +0.16] / +0.18 [−0.10, +0.45] |

**Bold** = confidence interval above 0 (a significant gain); *italic* =
confidence interval below 0 (a significant loss). For reference, over the 4:
spatial TTA (plain average) − original = +0.89 / −0.22 / −0.57.

On these 4 corruptions (Spearman at LP 0.75, ResNet-18 / DeiT), "energy
added" from section 3 gives +0.71 / +0.65 and "corruption energy removed"
+0.38 / +0.68: the latter counts the detail that Gaussian blur removed as if
the view could remove it again. With only 4 corruption types (20
conditions), these correlations are indicative only.

**Conclusion:** on unseen corruption types, low-pass views help clearly on
noise (speckle) and not on the others. The spectral set alone loses about
0.4 pts on average for DeiT and ConvNeXt, while the combined set stays at or
above the original for all three models. As on the standard corruptions,
low-pass views help on noise and not on blur.

## 8. Calibration (combined views)

| | NLL original | NLL original, temperature-scaled | NLL fused | ECE original | ECE original, temperature-scaled | ECE fused |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| ResNet-18 | 3.679 | 3.574 | **3.342** | 0.093 | **0.061** | **0.061** |
| DeiT-Tiny | 2.838 | 2.751 | **2.609** | 0.122 | **0.084** | 0.096 |
| ConvNeXt-Tiny | 2.119 | 2.072 | **1.961** | 0.100 | 0.091 | **0.091** |

Temperature-scaled = the original classifier with one temperature fitted on
the tuning half.  
**Conclusion:** fused predictions have lower NLL on all
three models; ECE is equal or better except for DeiT.

## 9. Ladder spacing

Geometric cutoffs (same endpoints and number of views) vs. the fixed ladder,
spectral views: fixed − geometric = +0.04 [−0.00, +0.09] (ResNet-18),
−0.10 [−0.14, −0.06] (DeiT-Tiny).  
**Conclusion:** the spacing makes no
practical difference.

## 10. Other view families and signals

Alternatives to the method's design choices, each compared with it
(`scripts/view_family_comparison.py`, `scripts/decorrelation_diagnostic.py`,
`scripts/experiment_ambiguity.py`).

### 10.1 Low-pass vs. band-pass vs. notch views

Single views (no fusion), all 15 corruptions at severity 3 plus clean
images. **Band-pass** keeps only a ring of frequencies (plus the image mean);
**notch** removes only a ring. For each family and image group the best
setting is chosen on the tuning half; the table gives its test-half gain over
the original image (`figures/table_view_families.tex`):

| images | view family | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|---|:--:|:--:|:--:|
| noise | low-pass | **+10.52** [+9.14, +11.84] | **+1.32** [+0.35, +2.29] | −0.31 [−0.93, +0.28] |
| | notch | +5.41 [+4.39, +6.45] | +0.38 [−0.54, +1.26] | −2.60 [−3.26, −2.00] |
| | band-pass | −4.61 [−5.92, −3.30] | −7.80 [−9.02, −6.60] | −9.29 [−10.39, −8.20] |
| digital | low-pass | +1.30 [+0.59, +2.00] | +0.59 [+0.02, +1.17] | +0.66 [+0.25, +1.03] |
| | notch | **+2.22** [+1.58, +2.91] | **+1.11** [+0.61, +1.62] | **+1.17** [+0.63, +1.67] |
| | band-pass | −6.29 [−7.23, −5.34] | −5.25 [−6.08, −4.44] | −5.17 [−5.91, −4.40] |
| all corruptions | low-pass | **+1.50** [+1.15, +1.84] | −0.26 [−0.55, +0.00] | −0.28 [−0.51, −0.04] |
| | notch | +1.12 [+0.79, +1.44] | −0.49 [−0.81, −0.19] | −0.66 [−0.89, −0.42] |
| | band-pass | −8.39 [−8.98, −7.80] | −7.94 [−8.51, −7.35] | −7.52 [−8.05, −7.00] |
| clean | low-pass | −0.08 [−0.77, +0.69] | −0.77 [−1.42, −0.12] | −0.28 [−0.89, +0.36] |
| | notch | −0.32 [−1.14, +0.45] | −1.14 [−1.99, −0.32] | −0.04 [−0.69, +0.61] |
| | band-pass | −6.77 [−8.11, −5.47] | −4.14 [−5.39, −2.84] | −3.69 [−4.70, −2.64] |

Blur and weather rows are in the LaTeX table; there no family gains more
than +0.7 pts. These numbers are for severity 3 only, with the setting chosen
on the tuning half, so they differ from section 3, which averages all five
severities (e.g. noise, ResNet-18: +10.52 here vs. +7.3 there).

**Conclusion:** band-pass views hurt on every image group, clean included:
removing the low frequencies, where most of an image lies, takes the input
too far from what the models were trained on. Notch views beat low-pass on
digital corruptions (narrow-band block artefacts) for all three models, but
help much less on noise, so low-pass is the best single family overall.
Fusing notch views with the low-pass ladder was not tested.

### 10.2 Fixed vs. energy-adaptive ladder

The method uses the same cutoffs for every image. The alternative chooses
cutoffs per image so that each view keeps a fixed share of the image's
spectral energy (99% down to 93%). Same protocol as 10.1
(`figures/table_ladders.tex`):

| images | ResNet-18 fixed / adaptive | DeiT-Tiny fixed / adaptive | ConvNeXt-Tiny fixed / adaptive |
|---|:--:|:--:|:--:|
| noise | **+10.52** / +7.42 | +1.32 / +1.27 | −0.31 / +0.01 |
| blur | **+0.17** / −3.69 | **−0.78** / −4.99 | **−0.45** / −8.63 |
| weather | **−0.74** / −14.59 | **−1.87** / −19.88 | **−1.02** / −23.74 |
| digital | **+1.30** / −8.48 | **+0.59** / −11.87 | **+0.66** / −13.28 |
| all corruptions | **+1.50** / −6.24 | **−0.26** / −9.55 | **−0.28** / −12.17 |
| clean | **−0.08** / −2.88 | **−0.77** / −4.66 | **−0.28** / −2.31 |

(Gains over the original in points; CIs in `results/ladder_comparison.json`.)

**Conclusion:** the adaptive ladder fails on blur and weather. Those images
have little high-frequency energy, so keeping 99% of their energy already
needs a strong filter: its *mildest* view loses 4-9 pts on blur and 15-24 pts
on weather. Fixed cutoffs avoid this and are better on every image group;
on noise the two are equal for DeiT and ConvNeXt.

### 10.3 What the learned per-view weights track

Correlation (Pearson, over the 15 views, tuning half) between each view's
learned weight and (a) that view's own accuracy, (b) how much the plain
average of the views loses when that view is left out
(`figures/table_view_weight_correlation.tex`):

| fit | correlation with | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|---|:--:|:--:|:--:|
| bias only | view's own accuracy | +0.08 | +0.44 | +0.54 |
| | contribution to the ensemble | **+0.89** | **+0.92** | **+0.88** |
| CE + bias | view's own accuracy | −0.55 | +0.21 | +0.19 |
| | contribution to the ensemble | **+0.87** | **+0.88** | **+0.94** |

**Conclusion:** the learned weights favour views that add information the
other views do not have, not views that are accurate on their own. This is
why a heavily filtered view can keep a sizeable weight.

### 10.4 A fourth trust signal

Added a fourth per-view signal, how much the view disagrees with the average
of all views, to the CE + bias fit (same settings otherwise, combined views;
`figures/table_fourth_signal.tex`):

| | 3 signals + bias | 4 signals + bias | difference |
|---|:--:|:--:|:--:|
| ResNet-18 | 36.54 | 36.59 | +0.05 [+0.01, +0.09] |
| DeiT-Tiny | 47.50 | 47.48 | −0.02 [−0.05, +0.01] |
| ConvNeXt-Tiny | 60.37 | 60.37 | −0.00 [−0.00, +0.00] |

**Conclusion:** the extra signal adds essentially nothing (ConvNeXt's fit
gives it zero weight), in line with section 5: the per-image signals carry
little information about which view is right.

## 11. Accuracy vs. number of views

The combined method at budgets of 2, 4, 8 and 15 views (= forward passes),
each including the original image (`scripts/view_budget_ablation.py`,
`results/view_budget/`, `figures/view_budget.pdf`; error bars there are the
95% CIs of the gain over the original).

- **Subsets:** greedy forward selection, for each leave-one-family-out fold,
  on the tuning half of the families that fold is tuned on: start from the
  original and repeatedly add the view that gives the highest plain-average
  accuracy (mean over those corruptions x 5 severities); ties go to the
  lower view index. A budget of *k* views uses the first *k* views of that
  order, so the subsets are nested. 15 views is the full combined set. This
  is a greedy rule, not a search over all subsets of each size.
- **Fusion:** each model keeps its selected method (ResNet-18: CE with signed
  coefficients; DeiT, ConvNeXt: per-view bias only), re-fit per fold on the
  subset with the same fit, scale and temperature search, tuning half only.
  15 views is the frozen combined configuration of section 1.
- **Uniform:** plain average of the same views.
- Test half, each corruption scored with the subset and fit of the fold that
  held its family out; clean images not scored.

Accuracy, and gain over the original
(original: 33.88 / 45.75 / 59.37):

| views | | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|:--:|---|:--:|:--:|:--:|
| 2 | uniform | 34.71 (+0.83 [+0.67, +1.00]) | 45.89 (+0.14 [−0.03, +0.32]) | 59.49 (+0.12 [−0.01, +0.24]) |
| | fused | 34.49 (+0.61 [+0.44, +0.79]) | 46.01 (+0.26 [+0.11, +0.42]) | 59.57 (+0.20 [+0.09, +0.31]) |
| 4 | uniform | 36.10 (+2.22 [+1.99, +2.43]) | 47.53 (+1.79 [+1.58, +1.99]) | 60.54 (+1.17 [+1.02, +1.31]) |
| | fused | 36.07 (+2.19 [+1.95, +2.42]) | 47.37 (+1.63 [+1.44, +1.82]) | 60.50 (+1.13 [+1.00, +1.26]) |
| 8 | uniform | 36.32 (+2.44 [+2.20, +2.68]) | 47.60 (+1.85 [+1.60, +2.11]) | 60.41 (+1.05 [+0.87, +1.21]) |
| | fused | 36.43 (+2.55 [+2.30, +2.79]) | 47.61 (+1.87 [+1.64, +2.09]) | 60.48 (+1.11 [+0.95, +1.26]) |
| 15 | uniform | 36.31 (+2.43 [+2.21, +2.64]) | 47.42 (+1.68 [+1.40, +1.95]) | 60.39 (+1.02 [+0.81, +1.22]) |
| | **fused** | **36.60** (+2.73 [+2.48, +2.96]) | **47.63** (+1.88 [+1.66, +2.11]) | **60.57** (+1.20 [+1.04, +1.35]) |

Difference from the full 15-view fused method:

| views | | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|:--:|---|:--:|:--:|:--:|
| 2 | fused | −2.11 [−2.30, −1.93] | −1.62 [−1.77, −1.47] | −1.00 [−1.11, −0.89] |
| 4 | uniform | −0.51 [−0.64, −0.37] | −0.09 [−0.18, −0.00] | −0.03 [−0.09, +0.04] |
| | fused | −0.54 [−0.67, −0.39] | −0.25 [−0.34, −0.16] | −0.06 [−0.13, +0.00] |
| 8 | uniform | −0.28 [−0.39, −0.17] | −0.03 [−0.11, +0.06] | −0.15 [−0.23, −0.07] |
| | fused | −0.18 [−0.28, −0.07] | −0.01 [−0.06, +0.04] | −0.09 [−0.14, −0.04] |
| 15 | uniform | −0.29 [−0.37, −0.21] | −0.20 [−0.31, −0.10] | −0.18 [−0.28, −0.08] |

Fused minus uniform at the same budget: 2 views −0.22 / +0.12 / +0.08,
4 views −0.03 / −0.16 / −0.04, 8 views +0.10 / +0.01 / +0.06, 15 views
+0.29 / +0.20 / +0.18 (ResNet-18 / DeiT / ConvNeXt; CIs in the JSON).

Views picked first: in 11 of the 12 folds (3 models x 4 folds) the first
two additions are a mid low-pass view (cutoff 0.30 or 0.45) and a flip
(plain or of a corner crop), in either order; in ResNet-18's weather fold
they are two low-pass views (0.30, 0.45), with the flip third. Full orders:
`results/view_budget/view_budget.json`.

**Conclusion:** 4 views already give 80-98% of the full 15-view gain. At 4
views, the better of uniform and fused is 0.5 pts below the 15-view method
for ResNet-18, 0.1 for DeiT, and statistically tied for ConvNeXt (−0.03 [−0.09, +0.04]).
Past 4 views, plain averaging adds at most +0.2 pts (ResNet-18) and nothing
for DeiT and ConvNeXt. Fitted weights only clearly help with all 15 views
(+0.2-0.3 over uniform); at 2-8 views they are within ±0.2 of uniform, and
can be worse (ResNet-18 at 2 views, DeiT at 4). The full 15-view fused
method stays the most accurate configuration overall; 8 fused views tie with
it for DeiT.

## 12. What each trust signal contributes

The three per-image signals, tested one at a time on the spectral views
(8 views; `scripts/signal_ablation.py`, `results/signal_ablation/`, pipeline
phase `signals`). Signals: confidence *C*, agreement with all other views
*A*, agreement with the original view only (*A*, origin), and stability
*S* (neighbouring ladder rungs, m = 1).

### 12.1 How informative is each signal on its own

No fitting. Test half. For each low-pass view *k*, the AUROC of the raw
signal for two targets:

- **(a)** view *k* is correct (score *s_k*);
- **(b)** view *k* is correct *and the original is wrong* (score
  *s_k − s_0*), i.e. the view should be trusted over the original. Only
  2.5-6.3% of image-view pairs are such cases.

The AUROC is computed within each corruption x severity condition, averaged
over conditions and over the 7 low-pass views (0.5 = no information).
95% CIs from a bootstrap over images (`figures/table_signal_auroc.tex`,
`figures/signal_auroc.pdf` for each family):

| target | signal | ResNet-18 | DeiT-Tiny | ConvNeXt-Tiny |
|---|---|:--:|:--:|:--:|
| (a) view correct | C | **0.805** [0.798, 0.812] | **0.810** [0.802, 0.818] | **0.808** [0.800, 0.816] |
| | A | 0.644 [0.635, 0.654] | 0.687 [0.677, 0.696] | 0.733 [0.724, 0.742] |
| | A (origin) | 0.702 [0.694, 0.709] | 0.710 [0.703, 0.717] | 0.763 [0.756, 0.770] |
| | S | 0.653 [0.645, 0.661] | 0.679 [0.671, 0.687] | 0.721 [0.713, 0.728] |
| (b) view correct, original wrong | C | 0.592 [0.585, 0.599] | 0.598 [0.589, 0.606] | 0.607 [0.599, 0.616] |
| | A | **0.630** [0.623, 0.637] | **0.620** [0.612, 0.628] | **0.649** [0.641, 0.657] |
| | A (origin) | 0.347 [0.339, 0.355] | 0.369 [0.359, 0.379] | 0.307 [0.298, 0.316] |
| | S | 0.587 [0.580, 0.594] | 0.583 [0.576, 0.591] | 0.625 [0.618, 0.633] |

By family: for target (b), agreement with all views is the most informative
signal and agreement with the original the least in every family and model;
for target (a), confidence is the most informative everywhere except noise
on ConvNeXt (agreement with the original, 0.836 vs. 0.835). Stability
is most informative on noise and least on blur for all three models (target
(b): 0.64 / 0.62 / 0.66 on noise vs. 0.55 / 0.56 / 0.61 on blur).

### 12.2 Fusion with signal subsets

The CE + bias fit (the method selected for ResNet-18's spectral views)
refit with only some signals; the coefficients of the others are held at 0
(`tune_fusion_ce.py --signals`). Same fit, warm start, scale and temperature
search, leave-one-family-out on the tuning half; agreement = all views,
except in the last row. "Bias only" = fixed weight per view, no per-image
signal. The {C, A, S} and bias-only rows reproduce section 5 exactly.
Mean corrupted accuracy (test half) and paired differences
(`figures/table_signal_subsets.tex` also has the difference to {C, A, S}):

| signals fitted | ResNet-18 | − bias only | DeiT-Tiny | − bias only | ConvNeXt-Tiny | − bias only |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| none (bias only) | 35.40 | | **47.01** | | 60.10 | |
| C | 35.58 | +0.18 [+0.13, +0.24] | 46.91 | −0.09 [−0.18, −0.01] | 60.03 | −0.07 [−0.14, −0.01] |
| A | 35.64 | +0.24 [+0.18, +0.31] | 46.94 | −0.06 [−0.09, −0.04] | 60.14 | +0.04 [+0.00, +0.08] |
| S | **35.91** | +0.51 [+0.42, +0.59] | 46.70 | −0.31 [−0.38, −0.25] | 60.14 | +0.04 [+0.00, +0.08] |
| C, A | 35.82 | +0.42 [+0.34, +0.49] | 46.89 | −0.12 [−0.20, −0.03] | 60.13 | +0.03 [−0.03, +0.08] |
| C, S | 35.80 | +0.40 [+0.32, +0.48] | 46.89 | −0.12 [−0.21, −0.03] | 60.09 | −0.01 [−0.07, +0.05] |
| A, S | 35.88 | +0.48 [+0.40, +0.57] | 46.69 | −0.31 [−0.38, −0.25] | **60.15** | +0.04 [+0.01, +0.08] |
| C, A, S | 35.86 | +0.46 [+0.38, +0.54] | 46.89 | −0.12 [−0.21, −0.03] | 60.09 | −0.01 [−0.07, +0.05] |
| C, A (origin), S | 35.81 | +0.41 [+0.32, +0.49] | 46.89 | −0.12 [−0.21, −0.03] | 60.05 | −0.05 [−0.11, +0.01] |

Differences to {C, A, S} for ResNet-18: *S* alone +0.05 [+0.00, +0.09],
{A, S} +0.02 [−0.03, +0.07], {C, A} −0.04 [−0.09, −0.00], {C, S} −0.06
[−0.10, −0.02], *A* alone −0.22, *C* alone −0.28. For DeiT and ConvNeXt
every subset is within 0.2 pts of {C, A, S}. When fitted together, the
agreement weight drops to 0 for DeiT (global fit: α = 1.55, β = 0,
γ = 0.58), so {C, S}, {C, A, S} and {C, A (origin), S} give the same
accuracy.

**Conclusion:** Confidence says best whether a view is right (AUROC 0.81),
but it drops to 0.59-0.61 when the question is whether a view beats the
original, so much of it reflects how hard the image is, which all views of
the image share. For the question that matters for fusion - is this view
right where the original is wrong - all three signals carry a weak signal
(AUROC 0.58-0.65), agreement with the other views the most. Agreement with
the original points the wrong way (0.31-0.37), partly by construction: when
the original is wrong, a view that gives the same class is wrong too, and
such a view agrees closely with the original. That agreement ranks first
here while stability helps most in the fusion below is not a contradiction:
the AUROC ranks one view across images, whereas the fusion chooses between
the views of the same image, after per-image normalization of the signals
and on top of the fixed per-view weights - and the fusion result is what
decides whether a signal is useful. In the fusion, stability does most
of the per-image work for ResNet-18: on its own it gives the whole gain over
fixed per-view weights (+0.51 pts vs. +0.46 for all three signals), while
confidence alone gives +0.18 and agreement alone +0.24. For DeiT and
ConvNeXt no signal subset beats fixed per-view weights by more than 0.05
pts, and stability on its own costs DeiT 0.31 pts. So stability is useful
only for the noise-sensitive model; for the other two the per-image signals
add nothing beyond which views are used and their fixed weights.
