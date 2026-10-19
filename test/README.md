# Tests

Unit tests for the library code in [`src/`](../src/README.md) and the tuning
script. They use small synthetic inputs, so they need **no GPU and no
dataset**, and run in a few seconds.

```bash
python -m pytest -q test        # from the repository root
```

| file | what it checks |
|---|---|
| `test_analysis.py` | Jensen-Shannon divergence, the confidence / agreement / stability signals, softmax fusion weights |
| `test_metrics.py` | accuracy and calibration scoring; output temperature keeps predictions; baselines are independent of the fusion settings |
| `test_evaluation.py` | the GPU evaluation path gives exactly the same numbers as the NumPy reference |
| `test_tune_fusion.py` | the GPU grid search computes the same signals and accuracies as the reference definitions; the `--signals` subset helpers of `tune_fusion_ce.py` |
| `test_ce_fusion.py` | cross-entropy fitting: gradients match finite differences, constraints hold, NumPy and torch agree, signals held out with `free_signals` stay at 0 |
| `test_bootstrap.py` | paired cluster bootstrap: point estimate, alignment by image, wider intervals for correlated conditions; image-weighted per-condition AUROC |
| `test_cache_geometry.py` | image geometry, spatial crop/flip views, the combined 15-view cache |
| `test_splits.py` | tune/test split keys for clean images; held-out extra corruptions stay out of tuning |

`test_models.py` is not a unit test but a smoke check that the three
pretrained classifiers download and load. It needs network access on the
first run:

```bash
PYTHONPATH=. python test/test_models.py
```
