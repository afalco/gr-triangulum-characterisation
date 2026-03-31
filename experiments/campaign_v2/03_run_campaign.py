"""
03_run_campaign.py
==================
Main campaign orchestrator for the extended Grover-Rudolph
characterisation campaign (v2).

Executes 1050 hardware runs in a drift-mitigated interleaved order:
  - 7 distributions (D0-D6)
  - 3 stages per distribution (L0, L01, FULL)
  - 2 ladders at FULL (A, B); 1 ladder (A) at L0 and L01
  - 25 runs per (distribution, stage, ladder) group

All results are appended to a JSONL log in real time so that the
campaign can be resumed after interruption.

Usage (PowerShell)
------------------
    $env:SPINQ_IP       = "192.168.1.25"
    $env:SPINQ_PORT     = "55444"
    $env:SPINQ_ACCOUNT  = "my_user"
    $env:SPINQ_PASSWORD = "my_secret_password"
    $env:SPINQ_BITORDER = "MSB->LSB"

    # Full campaign
    python 03_run_campaign.py

    # Resume from a previous partial log
    python 03_run_campaign.py --resume

    # Only specific distributions
    python 03_run_campaign.py --dist D3 D5 D6

    # Dry-run (simulator only, no hardware)
    python 03_run_campaign.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gr.angles import angles_3q_asin_child1
from gr.circuit import build_gr_circuit_3q
from gr.backends import run_simulator, run_nmr_probs_robust
from gr.metrics import tv_distance, l2_distance, fidelity
from gr.utils import per_qubit_marginals
# ---------------------------------------------------------------------------

# Campaign constants
N_RUNS_PER_GROUP  = 25
STAGES            = ["L0", "L01", "FULL"]
LADDERS           = {"L0": ["A"], "L01": ["A"], "FULL": ["A", "B"]}
BARE_CHECK_EVERY  = 100    # runs between periodic bare-state checks
BARE_THRESHOLD    = 0.97


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_distributions(config_path: Path) -> dict[str, np.ndarray]:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(v["p"]) for k, v in raw.items()}


def get_connection_params() -> dict:
    params = {
        "ip":       os.environ.get("SPINQ_IP", ""),
        "port":     int(os.environ.get("SPINQ_PORT", "55444")),
        "account":  os.environ.get("SPINQ_ACCOUNT", ""),
        "password": os.environ.get("SPINQ_PASSWORD", ""),
        "bitorder": os.environ.get("SPINQ_BITORDER", "MSB->LSB"),
    }
    return params


def load_completed_runs(logfile: Path) -> set[str]:
    """Return set of run_name values already in the log (for resume)."""
    if not logfile.exists():
        return set()
    completed = set()
    with open(logfile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if rec.get("status") == "completed":
                        completed.add(rec["run_name"])
                except json.JSONDecodeError:
                    pass
    return completed


def shannon_entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-np.sum(q * np.log2(q)))


def contrast(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(nz.max() / nz.min())


def log_run(record: dict, logfile: Path) -> None:
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_run_schedule(dists: dict[str, np.ndarray]) -> list[tuple]:
    """
    Build the interleaved run schedule.

    Returns a list of (dist_id, stage, ladder, repeat) tuples ordered
    so that each cycle visits all (dist, stage, ladder) groups once
    (5 runs per group per cycle, 5 cycles = 25 runs per group total).

    This interleaving ensures hardware drift affects all groups equally.
    """
    groups = []
    for dist_id in dists:
        for stage in STAGES:
            for ladder in LADDERS[stage]:
                groups.append((dist_id, stage, ladder))

    schedule = []
    runs_per_cycle = N_RUNS_PER_GROUP // 5   # 5 runs per group per cycle
    for cycle in range(5):
        for dist_id, stage, ladder in groups:
            for r in range(runs_per_cycle):
                repeat = cycle * runs_per_cycle + r + 1
                schedule.append((dist_id, stage, ladder, repeat))
    return schedule


def execute_run(
    dist_id: str,
    p: np.ndarray,
    stage: str,
    ladder: str,
    repeat: int,
    run_index: int,
    angles: dict,
    conn: dict,
    dry_run: bool,
    shots: int,
) -> dict:
    """Execute one hardware run and return the log record."""

    run_name = f"{dist_id}_{stage}_{ladder}_{repeat:03d}"
    ts_start = datetime.now(timezone.utc).isoformat()

    circ    = build_gr_circuit_3q(angles, depth=stage, ladder=ladder)
    sim_out = run_simulator(circ)

    # Gate metadata from circuit object
    n_gates = getattr(circ, "n_gates", None)
    n_ry    = getattr(circ, "n_ry",    None)
    n_x     = getattr(circ, "n_x",     None)
    n_cnot  = getattr(circ, "n_cnot",  None)

    status = "completed"
    exp_probs = None

    if dry_run:
        # Simulate hardware with a small amount of noise for testing
        noise = np.random.default_rng(run_index).dirichlet(
            np.full(8, 50.0))   # very concentrated -> near-ideal
        exp_probs = 0.95 * sim_out + 0.05 * noise
        exp_probs /= exp_probs.sum()
        platform_run_id = -1
    else:
        try:
            exp_probs = run_nmr_probs_robust(
                circuit=circ,
                shots=shots,
                conn=conn,
            )
            platform_run_id = int(time.time() * 1000) % 10_000_000
        except Exception as exc:
            status = "failed"
            exp_probs = np.full(8, np.nan)
            platform_run_id = -1
            print(f"    [HARDWARE ERROR] {run_name}: {exc}")

    ts_end = datetime.now(timezone.utc).isoformat()

    # Metrics (skip if failed)
    if status == "completed":
        tv_sim  = tv_distance(sim_out, exp_probs)
        l2_sim  = l2_distance(sim_out, exp_probs)
        fid_sim = fidelity(sim_out, exp_probs)
        tv_tgt  = tv_distance(p, exp_probs)
        l2_tgt  = l2_distance(p, exp_probs)
        fid_tgt = fidelity(p, exp_probs)
        sim_marg = per_qubit_marginals(sim_out)
        exp_marg = per_qubit_marginals(exp_probs)
    else:
        nan = float("nan")
        tv_sim = l2_sim = fid_sim = nan
        tv_tgt = l2_tgt = fid_tgt = nan
        sim_marg = exp_marg = {}

    STATES = ["000", "001", "010", "011", "100", "101", "110", "111"]

    record = {
        # Identity
        "run_name":        run_name,
        "run_index":       run_index,
        "dist_id":         dist_id,
        "stage":           stage,
        "ladder":          ladder,
        "repeat":          repeat,
        # Hardware
        "platform_run_id": platform_run_id,
        "status":          status,
        "created":         ts_start,
        "end":             ts_end,
        # Distribution metadata
        "p_target":        p.tolist(),
        "shannon_entropy": shannon_entropy(p),
        "contrast":        contrast(p),
        # Circuit metadata
        "n_qubits":        3,
        "n_gates":         n_gates,
        "n_ry":            n_ry,
        "n_x":             n_x,
        "n_cnot":          n_cnot,
        # Probabilities
        "sim_probs":  {s: float(v) for s, v in zip(STATES, sim_out)},
        "exp_probs":  {s: float(v) for s, v in zip(STATES, exp_probs)},
        # Metrics
        "tv_vs_sim":         tv_sim,
        "l2_vs_sim":         l2_sim,
        "fidelity_vs_sim":   fid_sim,
        "tv_vs_target":      tv_tgt,
        "l2_vs_target":      l2_tgt,
        "fidelity_vs_target": fid_tgt,
        # Marginals
        "sim_marginals":  sim_marg,
        "exp_marginals":  exp_marg,
    }
    return record


# ---------------------------------------------------------------------------
# Bare-state health check (periodic)
# ---------------------------------------------------------------------------

def bare_state_ok(conn: dict, shots: int = 2048) -> bool:
    """Quick bare-state check; returns True if P(|000>) >= BARE_THRESHOLD."""
    from gr.circuit import build_bare_circuit
    circ  = build_bare_circuit(n_qubits=3)
    try:
        probs = run_nmr_probs_robust(circuit=circ, shots=shots, conn=conn)
        p000  = float(probs[0])
        ok    = p000 >= BARE_THRESHOLD
        print(f"  [Bare-state check] P(|000>)={p000:.4f} "
              f"{'OK' if ok else 'WARN'}")
        return ok
    except Exception as exc:
        print(f"  [Bare-state check] Hardware error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    dist_ids:   list[str] | None,
    config:     Path,
    outdir:     Path,
    logfile:    Path,
    resume:     bool,
    dry_run:    bool,
    shots:      int,
    start_from: int,
) -> None:

    outdir.mkdir(parents=True, exist_ok=True)
    dists = load_distributions(config)

    if dist_ids:
        missing = set(dist_ids) - set(dists)
        if missing:
            sys.exit(f"[ERROR] Unknown IDs: {missing}")
        dists = {k: v for k, v in dists.items() if k in dist_ids}

    conn = get_connection_params()
    if not dry_run and not conn["ip"]:
        sys.exit("[ERROR] SPINQ_IP not set. Use --dry-run or set env variables.")

    # Pre-compute angles for all distributions
    print("Pre-computing GR angles ...")
    all_angles = {dist_id: angles_3q_asin_child1(p)
                  for dist_id, p in dists.items()}
    print(f"  Done. {len(all_angles)} distributions loaded.\n")

    # Build schedule
    schedule = build_run_schedule(dists)
    total    = len(schedule)
    print(f"Campaign schedule: {total} runs total")
    print(f"  Distributions : {list(dists.keys())}")
    print(f"  Stages        : {STAGES}")
    print(f"  Ladder groups : L0->A  L01->A  FULL->A,B")
    print(f"  Runs/group    : {N_RUNS_PER_GROUP}")
    print(f"  Mode          : {'DRY-RUN (simulator)' if dry_run else 'HARDWARE'}\n")

    # Resume: skip already-completed runs
    completed = load_completed_runs(logfile) if resume else set()
    if completed:
        print(f"[Resume] {len(completed)} runs already in log — skipping.\n")

    n_done    = 0
    n_failed  = 0
    rng_health = 0    # counter for periodic bare-state checks

    for run_index, (dist_id, stage, ladder, repeat) in enumerate(schedule, 1):
        if run_index < start_from:
            continue

        run_name = f"{dist_id}_{stage}_{ladder}_{repeat:03d}"
        if run_name in completed:
            continue

        # Periodic bare-state health check
        if not dry_run and rng_health > 0 and rng_health % BARE_CHECK_EVERY == 0:
            if not bare_state_ok(conn):
                print("[WARN] Bare-state check failed. "
                      "Pausing 5 min for equilibration ...")
                time.sleep(300)
                if not bare_state_ok(conn):
                    sys.exit("[ERROR] Device not ready after wait. Aborting.")

        p = dists[dist_id]
        angles = all_angles[dist_id]

        print(f"[{run_index:04d}/{total}] {run_name} ...", end=" ", flush=True)
        t0 = time.monotonic()

        record = execute_run(
            dist_id=dist_id, p=p, stage=stage, ladder=ladder,
            repeat=repeat, run_index=run_index,
            angles=angles, conn=conn,
            dry_run=dry_run, shots=shots,
        )
        record["duration_s"] = round(time.monotonic() - t0, 3)
        log_run(record, logfile)

        status = record["status"]
        fid    = record.get("fidelity_vs_target", float("nan"))
        tv     = record.get("tv_vs_target",       float("nan"))
        print(f"{status.upper()}  Fid={fid:.4f}  TV={tv:.4f}  "
              f"({record['duration_s']:.1f}s)")

        if status == "completed":
            n_done += 1
        else:
            n_failed += 1

        rng_health += 1

    print(f"\n{'='*60}")
    print(f"Campaign complete.")
    print(f"  Completed : {n_done}")
    print(f"  Failed    : {n_failed}")
    print(f"  Log       : {logfile}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Campaign v2 hardware orchestrator.")
    parser.add_argument("--config", type=Path,
        default=Path("artifacts/campaign_v2/campaign_distributions.json"))
    parser.add_argument("--outdir", type=Path,
        default=Path("artifacts/campaign_v2"))
    parser.add_argument("--logfile", type=Path,
        default=Path("artifacts/campaign_v2/campaign_v2_runs.jsonl"))
    parser.add_argument("--dist", nargs="+", metavar="ID",
        help="Run only these distribution IDs (e.g. D3 D5)")
    parser.add_argument("--resume", action="store_true",
        help="Skip runs already present in the log file")
    parser.add_argument("--dry-run", action="store_true",
        help="Use simulator only (no hardware)")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--start-from", type=int, default=1, metavar="N",
        help="Skip the first N-1 scheduled runs (useful for partial resume)")
    args = parser.parse_args()

    main(
        dist_ids=args.dist,
        config=args.config,
        outdir=args.outdir,
        logfile=args.logfile,
        resume=args.resume,
        dry_run=args.dry_run,
        shots=args.shots,
        start_from=args.start_from,
    )
