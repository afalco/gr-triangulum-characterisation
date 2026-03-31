"""
01_verify_simulations.py
========================
Pre-flight verification: runs the ideal Grover-Rudolph simulator for
all seven distributions in the campaign suite and confirms that

    TV(sim, target)  <= 1e-10
    Fid(sim, target) == 1.000  (to 10 decimal places)

for every distribution and both ladder variants (A and B) at the
FULL stage, and for L0 and L01 against their respective truncated
target distributions.

Must be run and pass before any hardware execution.

Usage (PowerShell)
------------------
    python 01_verify_simulations.py
    python 01_verify_simulations.py --outdir artifacts/campaign_v2
    python 01_verify_simulations.py --dist D3 D5   # subset only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repository imports  -------------------------------------------------------
# Adjust sys.path so this script can be run from campaigns/v2/ directly
# while the gr package lives two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gr.angles import angles_3q_asin_child1          # GR angle engine
from gr.circuit import build_gr_circuit_3q           # circuit builder
from gr.backends import run_simulator                # ideal simulator
from gr.metrics import tv_distance, fidelity         # metric functions
# ---------------------------------------------------------------------------


TV_TOL  = 1e-10
FID_TOL = 1e-10


def load_distributions(config_path: Path) -> dict[str, np.ndarray]:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(v["p"]) for k, v in raw.items()}


def truncated_target(p: np.ndarray, stage: str) -> np.ndarray:
    """
    Return the ideal (simulator) distribution for a truncated stage.
    For L0 and L01 the simulator concentrates mass on partial support;
    for FULL it equals p exactly.
    """
    if stage == "FULL":
        return p.copy()

    # Recompute partial support via the angle engine at that depth
    angles = angles_3q_asin_child1(p)
    circ   = build_gr_circuit_3q(angles, depth=stage, ladder="A")
    return run_simulator(circ)


def verify_one(dist_id: str, p: np.ndarray,
               verbose: bool = True) -> list[dict]:
    """Run verification for all stages and ladders for one distribution."""
    results = []
    angles = angles_3q_asin_child1(p)

    for stage in ["L0", "L01", "FULL"]:
        ladders = ["A", "B"] if stage == "FULL" else ["A"]
        for ladder in ladders:
            circ    = build_gr_circuit_3q(angles, depth=stage, ladder=ladder)
            sim_out = run_simulator(circ)
            ref     = truncated_target(p, stage)

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
                tag = "PASS" if ok else "FAIL"
                print(
                    f"  [{tag}] {dist_id} {stage}-{ladder} | "
                    f"TV={tv_val:.2e}  Fid={fid_val:.10f}"
                )
    return results


def main(dist_ids: list[str] | None,
         config_path: Path,
         outdir: Path,
         verbose: bool) -> None:

    outdir.mkdir(parents=True, exist_ok=True)
    dists = load_distributions(config_path)

    if dist_ids:
        missing = set(dist_ids) - set(dists)
        if missing:
            sys.exit(f"[ERROR] Unknown distribution IDs: {missing}")
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

    # Write report
    report_path = outdir / "01_simulation_verification.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Verification report written to {report_path}")

    if all_pass:
        print("\n=== ALL CHECKS PASSED — safe to proceed to hardware ===")
    else:
        failed = [r for r in all_results if not r["pass"]]
        print(f"\n=== {len(failed)} CHECK(S) FAILED ===")
        for r in failed:
            print(f"  FAIL: {r['dist_id']} {r['stage']}-{r['ladder']} "
                  f"TV={r['tv_vs_target']:.2e} Fid={r['fidelity_vs_target']:.10f}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-flight simulator verification for campaign v2.")
    parser.add_argument("--outdir", type=Path,
                        default=Path("artifacts/campaign_v2"))
    parser.add_argument("--config", type=Path,
                        default=Path("artifacts/campaign_v2/campaign_distributions.json"),
                        help="Path to campaign_distributions.json")
    parser.add_argument("--dist", nargs="+", metavar="ID",
                        help="Verify only these distribution IDs (e.g. D3 D5)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-check output")
    args = parser.parse_args()

    main(
        dist_ids=args.dist,
        config_path=args.config,
        outdir=args.outdir,
        verbose=not args.quiet,
    )
