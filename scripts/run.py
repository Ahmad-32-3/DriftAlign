"""Source train, Sinkhorn onto target, score target only.

CLI: python scripts/run.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transport import const
from transport.eval import ablate, evaluate, print_report, write_data_ts, write_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=const.N_SOURCE)
    ap.add_argument("--eps", type=float, default=const.EPS)
    args = ap.parse_args()
    n = min(args.max_samples, const.MAX_SAMPLES)
    if n > const.MAX_SAMPLES:
        sys.exit(f"HARD FAIL: samples {n} over cap {const.MAX_SAMPLES}")

    m = evaluate(n_source=n, n_target=n, eps=args.eps)
    m["ablations"] = ablate(n_source=n, n_target=n, seed=const.SEED)
    if m["n_source"] > const.MAX_SAMPLES or m["n_target"] > const.MAX_SAMPLES:
        sys.exit("HARD FAIL: cap")

    if not m.get("illustrative") and m["success_pct"] < const.FLOOR_PCT:
        sys.exit(
            f"HARD FAIL: success_pct {m['success_pct']:.1f} < {const.FLOOR_PCT} "
            f"(no-OT {m['no_ot_pct']:.1f})"
        )
    if m["success_pct"] <= m["no_ot_pct"]:
        sys.exit(
            f"HARD FAIL: OT {m['success_pct']:.1f} did not beat no-OT {m['no_ot_pct']:.1f}"
        )

    write_metrics(m)
    print_report(m)
    write_data_ts(m)


if __name__ == "__main__":
    main()
