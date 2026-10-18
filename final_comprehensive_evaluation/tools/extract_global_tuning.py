#!/usr/bin/env python3
"""Create a global-only tuning JSON from a normal project tuning record.

The regular tuning scripts also save leave-one-family-out configurations.
For the final global experiment we deliberately discard those entries so
`scripts/evaluate.py` must use the single configuration tuned on the full
training/tuning split across all corruptions and severities.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    rec = json.loads(src.read_text())
    if not rec.get("global") or not rec["global"].get("config"):
        raise SystemExit(f"{src} does not contain global.config")

    out = dict(rec)
    out["held_out"] = {}
    out["final_scope"] = {
        "label": args.label,
        "selection": "single global configuration",
        "tuning_data": "all corruption types x all requested severities, tune half only",
        "evaluation_data": "held-out test half only",
        "source_tuning": str(src),
    }

    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(dst)


if __name__ == "__main__":
    main()
