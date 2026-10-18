#!/usr/bin/env python3
"""Create and validate a global-only CE+bias tuning JSON.

The regular tuner also saves leave-one-family-out configurations.  This final
experiment deliberately discards those entries so evaluation must use one
global CE+bias configuration fitted on the full tune/train half across all
corruptions and severities.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_number(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    rec = json.loads(src.read_text())
    if rec.get("method") != "ce_bias":
        raise SystemExit(f"{src} has method={rec.get('method')!r}; expected 'ce_bias'")
    if not rec.get("global") or not rec["global"].get("config"):
        raise SystemExit(f"{src} does not contain global.config")

    cfg = rec["global"]["config"]
    bias = cfg.get("view_bias")
    if not isinstance(bias, list) or len(bias) != 15:
        raise SystemExit(
            f"{src} global CE+bias config has view_bias length "
            f"{len(bias) if isinstance(bias, list) else 'invalid'}; expected 15"
        )
    for name in ("alpha", "beta", "gamma", "tau", "output_temperature"):
        if not finite_number(cfg.get(name)):
            raise SystemExit(f"{src} global config has invalid {name}={cfg.get(name)!r}")
    if float(cfg["tau"]) <= 0 or float(cfg["output_temperature"]) <= 0:
        raise SystemExit(f"{src} requires tau>0 and output_temperature>0")
    if not all(finite_number(v) for v in bias):
        raise SystemExit(f"{src} contains non-finite view_bias values")

    out = dict(rec)
    out["held_out"] = {}
    out["final_scope"] = {
        "label": args.label,
        "selection": "single global CE+bias configuration",
        "tuning_data": "all 15 corruption types x all 5 severities, tune half only",
        "evaluation_data": "held-out test half only",
        "source_tuning": str(src),
        "validated_method": "ce_bias",
        "validated_view_bias_length": 15,
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(dst)


if __name__ == "__main__":
    main()
