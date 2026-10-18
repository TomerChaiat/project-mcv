"""Low-pass vs. band-pass vs. notch views: which frequency-domain view family
helps a frozen classifier on corrupted images?

Single views, no fusion. For every condition (15 corruptions at one severity,
plus clean) each image is filtered with

  --set families (default):
    low-pass   the 7 cutoffs of the fixed ladder (src/cache.FIXED_CUTOFFS);
    band-pass  keep only a ring of frequencies (plus DC), src/filters.py;
    notch      remove only a ring of frequencies, src/filters.py;
  --set ladders:
    low-pass   the fixed ladder, as above;
    energy     per-image cutoffs keeping 99% ... 93% of the image's spectral
               energy (src/cache.ENERGIES, src/cache.spectral_views);

and every view goes through each frozen model. Per family and corruption
family, the best configuration is chosen on the TUNE half and reported on the
TEST half as a gain over the original view, with a paired cluster bootstrap
over images. Per-condition correctness is saved as it goes
(<out-dir>/<model>_<corruption>_<sev>.npz), so an interrupted run resumes.

    python scripts/view_family_comparison.py --out-dir results/view_family
    python scripts/view_family_comparison.py --set ladders --out-dir results/ladder_compare \
        --out results/ladder_comparison.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import cache, filters
from src.bootstrap import cluster_bootstrap, image_keys
from src.splits import CORRUPTION_FAMILY, FAMILIES, tune_test_masks

BANDPASS = [(0.25, 0.15), (0.45, 0.15), (0.65, 0.15), (0.85, 0.15), (0.45, 0.30), (0.65, 0.30)]  # (center, width)
NOTCH = [(0.25, 0.10, 1.0), (0.45, 0.10, 1.0), (0.65, 0.10, 1.0), (0.9, 0.15, 1.0),
         (0.45, 0.10, 0.5), (0.65, 0.10, 0.5), (0.3, 0.25, 1.0)]                            # (center, width, depth)
VIEWS = (["original"] + [f"lowpass {c}" for c in cache.FIXED_CUTOFFS]
         + [f"bandpass c={c} w={w}" for c, w in BANDPASS]
         + [f"notch c={c} w={w} d={d}" for c, w, d in NOTCH])
FAMILY_OF_VIEW = (["original"] + ["lowpass"] * len(cache.FIXED_CUTOFFS)
                  + ["bandpass"] * len(BANDPASS) + ["notch"] * len(NOTCH))
VIEW_FAMILIES = ["lowpass", "bandpass", "notch"]


@torch.no_grad()
def build_views(batch):
    """(B, 3, H, W) in [0, 1] -> (B, V, 3, H, W), in VIEWS order."""
    views = [cache.spectral_views_fixed(batch)]                     # original + 7 low-pass
    views += [filters.apply_bandpass(batch, c, w)[:, None] for c, w in BANDPASS]
    views += [filters.apply_notch(batch, c, w, depth=d)[:, None] for c, w, d in NOTCH]
    return torch.cat(views, dim=1)


@torch.no_grad()
def build_ladder_views(batch):
    """original + fixed ladder + energy-adaptive ladder."""
    energy_views, _ = cache.spectral_views(batch, cache.ENERGIES)   # original + 7
    return torch.cat([cache.spectral_views_fixed(batch), energy_views[:, 1:]], dim=1)


def use_view_set(name):
    """Switch the module-level view set ('families' or 'ladders')."""
    global VIEWS, FAMILY_OF_VIEW, VIEW_FAMILIES, build_views
    if name == "ladders":
        VIEWS = (["original"] + [f"lowpass {c}" for c in cache.FIXED_CUTOFFS]
                 + [f"energy {e}" for e in cache.ENERGIES])
        FAMILY_OF_VIEW = ["original"] + ["lowpass"] * 7 + ["energy"] * len(cache.ENERGIES)
        VIEW_FAMILIES = ["lowpass", "energy"]
        build_views = build_ladder_views


def run_condition(models, corruption, sev, wnid_to_index, out_dir, device, batch_size):
    paths, labels = cache.list_condition_images(corruption, sev, wnid_to_index)
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    correct = {m: np.zeros((len(paths), len(VIEWS)), dtype=bool) for m in models}
    for i, count, batch_cpu in cache._prefetched_image_batches(paths, cache.load_image, batch_size):
        views = build_views(batch_cpu.to(device))
        y = labels_t[i:i + count]
        for m, model in models.items():
            pred = cache.view_probs(model, views).argmax(-1)
            correct[m][i:i + count] = pred.eq(y[:, None]).cpu().numpy()
    for m in models:
        np.savez_compressed(out_dir / f"{m}_{corruption}_{sev}.npz",
                            correct=correct[m], paths=np.asarray(paths))


def summarize(model_names, conditions, out_dir, n_boot, seed):
    """Best configuration per (view family, corruption family) chosen on the tune
    half; its test-half gain over the original with a paired cluster bootstrap."""
    report = {"views": VIEWS, "models": {}}
    groups = {f: [c for c in conditions if CORRUPTION_FAMILY.get(c[0]) == f] for f in FAMILIES}
    groups["all corruptions"] = [c for c in conditions if c[0] != "clean"]
    groups["clean"] = [c for c in conditions if c[0] == "clean"]
    for m in model_names:
        data = {}
        for corruption, sev in conditions:
            d = np.load(out_dir / f"{m}_{corruption}_{sev}.npz")
            tune, test = tune_test_masks(d["paths"])
            data[(corruption, sev)] = (d["correct"], tune, test, image_keys(d["paths"]))
        rec = {}
        for g, conds in groups.items():
            if not conds:
                continue
            tune_acc = np.mean([data[c][0][data[c][1]].mean(0) for c in conds], 0)
            test_acc = np.mean([data[c][0][data[c][2]].mean(0) for c in conds], 0)
            row = {"original_test_acc": float(100 * test_acc[0]),
                   "per_view_test_minus_original": {v: float(100 * (test_acc[k] - test_acc[0]))
                                                    for k, v in enumerate(VIEWS)}}
            for fam in VIEW_FAMILIES:
                idx = [k for k, f in enumerate(FAMILY_OF_VIEW) if f == fam]
                best = idx[int(np.argmax(tune_acc[idx]))]
                bs = cluster_bootstrap(
                    [{"keys": data[c][3][data[c][2]], "a": data[c][0][data[c][2], best],
                      "b": data[c][0][data[c][2], 0]} for c in conds], n_boot=n_boot, seed=seed)
                row[fam] = {"selected_on_tune_half": VIEWS[best], "test_gain": bs}
            rec[g] = row
        report["models"][m] = rec
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet18", "deit", "convnext"])
    ap.add_argument("--set", choices=["families", "ladders"], default="families")
    ap.add_argument("--severity", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results/view_family")
    ap.add_argument("--out", default="results/view_family_comparison.json")
    args = ap.parse_args()
    use_view_set(args.set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wnid_to_index = json.load(open("data/subset_classes.json"))["wnid_to_index"]
    conditions = [(c, args.severity) for c in CORRUPTION_FAMILY] + [("clean", 0)]

    todo = [c for c in conditions
            if not all((out_dir / f"{m}_{c[0]}_{c[1]}.npz").exists() for m in args.models)]
    if todo:
        models = {m: cache.LOADERS[m](device)[0] for m in args.models}
        for corruption, sev in todo:
            run_condition(models, corruption, sev, wnid_to_index, out_dir, device, args.batch_size)
            print(f"  done {corruption}/{sev}", flush=True)
        del models

    report = summarize(args.models, conditions, out_dir, args.n_boot, args.seed)
    report["severity"] = args.severity
    Path(args.out).write_text(json.dumps(report, indent=2))
    for m, rec in report["models"].items():
        print(f"\n=== {m}: test-half gain over the original, best config chosen on the tune half")
        for g, row in rec.items():
            cells = []
            for fam in VIEW_FAMILIES:
                bs = row[fam]["test_gain"]
                cells.append(f"{fam} {100 * bs['point']:+.2f} [{100 * bs['ci95'][0]:+.2f},{100 * bs['ci95'][1]:+.2f}]")
            print(f"  {g:16s} orig {row['original_test_acc']:.2f}  " + "  ".join(cells))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
