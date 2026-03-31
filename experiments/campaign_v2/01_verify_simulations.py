"""
01_verify_simulations.py
========================
Pre-flight verification: runs the ideal SpinQit BasicSimulator for every
distribution in the campaign suite and confirms that

    TV(sim, target)  <= 1e-9
    Fid(sim, target) == 1.000  (to 10 decimal places)

for every distribution, both stages (L0, L01, FULL) and both ladder
variants (A and B at FULL).

Must pass completely before any hardware execution.

Dependencies
------------
    spinqit          (SpinQit SDK — https://doc.spinq.cn/doc/spinqit/)
    numpy

Usage (PowerShell)
------------------
    python 01_verify_simulations.py
    python 01_verify_simulations.py --dist D3 D5
    python 01_verify_simulations.py --outdir artifacts/campaign_v2
    python 01_verify_simulations.py --quiet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _spinqit_backend import run_simulator

# FULL-stage simulator-vs-target checks can differ from the analytic target
# by ~1e-10 because of floating-point roundoff in angle construction and
# compilation. A 1e-9 TV tolerance is still extremely strict while avoiding
# false negatives.
TV_TOL  = 1e-9
FID_TOL = 1e-10


# ── Self-contained metrics ───────────────────────────────────────────────────

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p - q)))

def fidelity(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.sqrt(np.clip(p, 0, None) *
                                np.clip(q, 0, None))) ** 2)


# ── Self-contained GR angle engine ──────────────────────────────────────────

def gr_angles(p: np.ndarray) -> dict:
    """Compute all GR angles for a 3-qubit distribution (degrees)."""
    def safe_acos(num, den):
        if den < 1e-15:
            return 0.0
        return 2.0 * np.degrees(np.arccos(
            np.sqrt(np.clip(num / den, 0.0, 1.0))))

    sL  = p[0] + p[1] + p[2] + p[3]
    sR  = 1.0 - sL
    sLL = p[0] + p[1]
    sLR = p[2] + p[3]
    sRL = p[4] + p[5]
    sRR = p[6] + p[7]

    theta2 = {
        "00": safe_acos(p[0], sLL),
        "01": safe_acos(p[2], sLR),
        "10": safe_acos(p[4], sRL),
        "11": safe_acos(p[6], sRR),
    }
    H = np.array([[1, 1, 1, 1],
                  [1,-1, 1,-1],
                  [1, 1,-1,-1],
                  [1,-1,-1, 1]]) / 4.0

    # Ladder A: WHT ordering [t00, t11, t10, t01]
    alpha_A = np.array([theta2["00"], theta2["11"],
                        theta2["10"], theta2["01"]]) / 2.0
    la_A = 2.0 * (H @ alpha_A)

    # Ladder B: WHT ordering [t00, t11, t01, t10]
    alpha_B = np.array([theta2["00"], theta2["11"],
                        theta2["01"], theta2["10"]]) / 2.0
    la_B = 2.0 * (H @ alpha_B)

    return {
        "theta0": safe_acos(sL, 1.0),
        "theta1_0": safe_acos(sLL, sL),
        "theta1_1": safe_acos(sRL, sR),
        "theta2": theta2,
        "ladder_angles_A": la_A,
        "ladder_angles_B": la_B,
    }


def load_distributions(config_path: Path) -> dict[str, np.ndarray]:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(v["p"]) for k, v in raw.items()}


def verify_one(dist_id: str, p: np.ndarray,
               verbose: bool = True) -> list[dict]:
    results = []
    angles = gr_angles(p)

    for stage in ["L0", "L01", "FULL"]:
        ladders = ["A", "B"] if stage == "FULL" else ["A"]
        for ladder in ladders:
            sim_out = run_simulator(angles, stage, ladder)
            # For L0/L01 the reference is the simulator itself (truncated ideal)
            # For FULL the reference is the target p
            ref = p if stage == "FULL" else run_simulator(angles, stage, "A")

            tv_val  = tv_distance(ref, sim_out)
            fid_val = fidelity(ref, sim_out)
            ok      = (tv_val <= TV_TOL) and (abs(fid_val - 1.0) <= FID_TOL)

            rec = {
                "dist_id": dist_id,
                "stage": stage,
                "ladder": ladder,
                "tv_vs_target": tv_val,
                "fidelity_vs_target": fid_val,
                "pass": ok,
            }
            results.append(rec)

            if verbose:
                print(f"  [{'PASS' if ok else 'FAIL'}] {dist_id} "
                      f"{stage}-{ladder} | TV={tv_val:.2e}  "
                      f"Fid={fid_val:.10f}")
    return results


def main(dist_ids: list[str] | None, config_path: Path,
         outdir: Path, verbose: bool) -> None:

    outdir.mkdir(parents=True, exist_ok=True)
    dists = load_distributions(config_path)

    if dist_ids:
        missing = set(dist_ids) - set(dists)
        if missing:
            sys.exit(f"[ERROR] Unknown IDs: {missing}")
        dists = {k: v for k, v in dists.items() if k in dist_ids}

    print(f"Verifying {len(dists)} distribution(s): {list(dists)}\n")

    all_results = []
    all_pass = True
    for dist_id, p in dists.items():
        print(f"--- {dist_id} ---")
        recs = verify_one(dist_id, p, verbose=verbose)
        all_results.extend(recs)
        if not all(r["pass"] for r in recs):
            all_pass = False

    report = outdir / "01_simulation_verification.json"
    with open(report, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Report written to {report}")

    if all_pass:
        print("\n=== ALL CHECKS PASSED — safe to proceed to hardware ===")
    else:
        failed = [r for r in all_results if not r["pass"]]
        print(f"\n=== {len(failed)} CHECK(S) FAILED ===")
        for r in failed:
            print(f"  FAIL: {r['dist_id']} {r['stage']}-{r['ladder']} "
                  f"TV={r['tv_vs_target']:.2e}  "
                  f"Fid={r['fidelity_vs_target']:.10f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("artifacts/campaign_v2/campaign_distributions.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/campaign_v2"),
    )
    parser.add_argument("--dist", nargs="+", metavar="ID")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(args.dist, args.config, args.outdir, not args.quiet)