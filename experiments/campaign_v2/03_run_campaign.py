"""
03_run_campaign.py
==================
Main campaign orchestrator for the extended Grover-Rudolph characterisation
campaign (v2). Executes 1050 hardware runs via the SpinQit NMR backend in
a drift-mitigated interleaved order.

Dependencies
------------
    spinqit   (https://doc.spinq.cn/doc/spinqit/)
    numpy, pandas

Usage (PowerShell)
------------------
    $env:SPINQ_IP       = "192.168.1.25"
    $env:SPINQ_PORT     = "55444"
    $env:SPINQ_ACCOUNT  = "my_user"
    $env:SPINQ_PASSWORD = "my_secret_password"

    python 03_run_campaign.py
    python 03_run_campaign.py --dry-run
    python 03_run_campaign.py --resume
    python 03_run_campaign.py --dist D3 D5 D6
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

from _spinqit_backend import run_simulator, run_hardware, run_bare_hardware

# ── Campaign constants ────────────────────────────────────────────────────────
N_RUNS_PER_GROUP = 25
STAGES   = ["L0", "L01", "FULL"]
LADDERS  = {"L0": ["A"], "L01": ["A"], "FULL": ["A", "B"]}
BARE_CHECK_EVERY = 100
BARE_THRESHOLD   = 0.90
STATES = ["000","001","010","011","100","101","110","111"]


# ── Self-contained helpers ────────────────────────────────────────────────────

def tv_distance(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))

def l2_distance(p, q):
    return float(np.sqrt(np.sum((p - q)**2)))

def fidelity(p, q):
    return float(np.sum(np.sqrt(np.clip(p,0,None)*np.clip(q,0,None)))**2)

def shannon_entropy(p):
    q = p[p > 0]
    return float(-np.sum(q * np.log2(q)))

def contrast(p):
    nz = p[p > 0]
    return float(nz.max() / nz.min())

def per_qubit_marginals(probs: np.ndarray) -> dict:
    out = {}
    for qi in range(3):
        p0 = sum(probs[k] for k in range(8) if not (k >> (2-qi) & 1))
        out[f"q{qi}"] = {"p0": float(p0), "p1": float(1.0 - p0)}
    return out

def gr_angles(p: np.ndarray) -> dict:
    def safe_acos(num, den):
        if den < 1e-15:
            return 0.0
        return 2.0 * np.degrees(np.arccos(
            np.sqrt(np.clip(num / den, 0.0, 1.0))))
    sL  = p[0]+p[1]+p[2]+p[3]; sR = 1.0 - sL
    sLL = p[0]+p[1]; sLR = p[2]+p[3]
    sRL = p[4]+p[5]; sRR = p[6]+p[7]
    theta2 = {
        "00": safe_acos(p[0], sLL), "01": safe_acos(p[2], sLR),
        "10": safe_acos(p[4], sRL), "11": safe_acos(p[6], sRR),
    }
    H = np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]) / 4.0
    alpha_A = np.array([theta2["00"], theta2["11"],
                        theta2["10"], theta2["01"]]) / 2.0
    la_A = 2.0 * (H @ alpha_A)
    alpha_B = np.array([theta2["00"], theta2["11"],
                        theta2["01"], theta2["10"]]) / 2.0
    la_B = 2.0 * (H @ alpha_B)
    return {
        "theta0": safe_acos(sL, 1.0), "theta1_0": safe_acos(sLL, sL),
        "theta1_1": safe_acos(sRL, sR), "theta2": theta2,
        "ladder_angles_A": la_A, "ladder_angles_B": la_B,
    }


# ── IO helpers ────────────────────────────────────────────────────────────────

def load_distributions(config_path: Path) -> dict[str, np.ndarray]:
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(v["p"]) for k, v in raw.items()}

def load_completed(logfile: Path) -> set[str]:
    if not logfile.exists():
        return set()
    completed = set()
    with open(logfile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("status") == "completed":
                        completed.add(r["run_name"])
                except json.JSONDecodeError:
                    pass
    return completed

def log_run(record: dict, logfile: Path) -> None:
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_conn() -> dict:
    params = {
        "ip":       os.environ.get("SPINQ_IP", "10.30.227.5"),
        "port":     os.environ.get("SPINQ_PORT", "55444"),
        "account":  os.environ.get("SPINQ_ACCOUNT", "user1"),
        "password": os.environ.get("SPINQ_PASSWORD", "lacasito99"),
    }
    return params

def build_schedule(dists: dict) -> list[tuple]:
    groups = [(d, s, l)
              for d in dists
              for s in STAGES
              for l in LADDERS[s]]
    runs_per_cycle = N_RUNS_PER_GROUP // 5
    schedule = []
    for cycle in range(5):
        for dist_id, stage, ladder in groups:
            for r in range(runs_per_cycle):
                repeat = cycle * runs_per_cycle + r + 1
                schedule.append((dist_id, stage, ladder, repeat))
    return schedule


# ── Single run ────────────────────────────────────────────────────────────────

def execute_run(dist_id, p, stage, ladder, repeat, run_index,
                angles, conn, dry_run, shots) -> dict:

    run_name = f"{dist_id}_{stage}_{ladder}_{repeat:03d}"
    ts_start = datetime.now(timezone.utc).isoformat()
    sim_out  = run_simulator(angles, stage, ladder)
    status   = "completed"
    exp_probs = None

    if dry_run:
        rng = np.random.default_rng(run_index)
        noise = rng.dirichlet(np.full(8, 50.0))
        exp_probs = 0.95 * sim_out + 0.05 * noise
        exp_probs /= exp_probs.sum()
        platform_run_id = -1
    else:
        try:
            task_conn = {**conn,
                         "task_name": f"gr_{dist_id}_{stage}_{ladder}_{repeat:03d}",
                         "task_desc": f"Campaign v2 run {run_index}"}
            exp_probs = run_hardware(angles, stage, ladder, task_conn, shots)
            platform_run_id = int(time.time() * 1000) % 10_000_000
        except Exception as exc:
            status = "failed"
            exp_probs = np.full(8, float("nan"))
            platform_run_id = -1
            print(f"    [HW ERROR] {run_name}: {exc}")

    ts_end = datetime.now(timezone.utc).isoformat()

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

    return {
        "run_name": run_name, "run_index": run_index,
        "dist_id": dist_id, "stage": stage, "ladder": ladder,
        "repeat": repeat, "platform_run_id": platform_run_id,
        "status": status, "created": ts_start, "end": ts_end,
        "p_target": p.tolist(),
        "shannon_entropy": shannon_entropy(p), "contrast": contrast(p),
        "n_qubits": 3,
        "sim_probs":  {s: float(v) for s,v in zip(STATES, sim_out)},
        "exp_probs":  {s: float(v) for s,v in zip(STATES, exp_probs)},
        "tv_vs_sim": tv_sim,  "l2_vs_sim": l2_sim,
        "fidelity_vs_sim": fid_sim,
        "tv_vs_target": tv_tgt, "l2_vs_target": l2_tgt,
        "fidelity_vs_target": fid_tgt,
        "sim_marginals": sim_marg, "exp_marginals": exp_marg,
    }


# ── Bare-state health check ───────────────────────────────────────────────────

def bare_ok(conn: dict, shots: int = 2048) -> bool:
    try:
        probs = run_bare_hardware(conn, shots=shots)
        p000 = float(probs[0])
        print(f"  [Bare check] P(|000>)={p000:.4f} "
              f"{'OK' if p000 >= BARE_THRESHOLD else 'WARN'}")
        return p000 >= BARE_THRESHOLD
    except Exception as exc:
        print(f"  [Bare check] HW error: {exc}")
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main(dist_ids, config, outdir, logfile, resume,
         dry_run, shots, start_from):

    outdir.mkdir(parents=True, exist_ok=True)
    dists = load_distributions(config)
    if dist_ids:
        missing = set(dist_ids) - set(dists)
        if missing:
            sys.exit(f"[ERROR] Unknown IDs: {missing}")
        dists = {k: v for k, v in dists.items() if k in dist_ids}

    conn = get_conn()
    if not dry_run and not conn["ip"]:
        sys.exit("[ERROR] SPINQ_IP not set. Use --dry-run or set env vars.")

    print("Pre-computing GR angles ...")
    all_angles = {d: gr_angles(p) for d, p in dists.items()}

    schedule = build_schedule(dists)
    total    = len(schedule)
    completed = load_completed(logfile) if resume else set()
    if completed:
        print(f"[Resume] {len(completed)} completed runs skipped.\n")

    print(f"Campaign: {total} runs  |  "
          f"Mode: {'DRY-RUN' if dry_run else 'HARDWARE'}\n")

    n_done = n_failed = health_ctr = 0

    for run_index, (dist_id, stage, ladder, repeat) in \
            enumerate(schedule, 1):
        if run_index < start_from:
            continue
        run_name = f"{dist_id}_{stage}_{ladder}_{repeat:03d}"
        if run_name in completed:
            continue

        if (not dry_run and health_ctr > 0
                and health_ctr % BARE_CHECK_EVERY == 0):
            if not bare_ok(conn):
                print("[WARN] Bare check failed. Pausing 5 min ...")
                time.sleep(300)
                if not bare_ok(conn):
                    sys.exit("[ERROR] Device not ready. Aborting.")

        p      = dists[dist_id]
        angles = all_angles[dist_id]
        print(f"[{run_index:04d}/{total}] {run_name} ...",
              end=" ", flush=True)
        t0 = time.monotonic()

        record = execute_run(dist_id, p, stage, ladder, repeat,
                             run_index, angles, conn, dry_run, shots)
        record["duration_s"] = round(time.monotonic() - t0, 3)
        log_run(record, logfile)

        fid = record.get("fidelity_vs_target", float("nan"))
        tv  = record.get("tv_vs_target",       float("nan"))
        print(f"{record['status'].upper()}  "
              f"Fid={fid:.4f}  TV={tv:.4f}  "
              f"({record['duration_s']:.1f}s)")

        if record["status"] == "completed":
            n_done += 1
        else:
            n_failed += 1
        health_ctr += 1

    print(f"\n{'='*55}")
    print(f"Done. Completed: {n_done}  Failed: {n_failed}")
    print(f"Log: {logfile}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=Path,
        default=Path("artifacts/campaign_v2/campaign_distributions.json"))
    parser.add_argument("--outdir",  type=Path,
        default=Path("artifacts/campaign_v2"))
    parser.add_argument("--logfile", type=Path,
        default=Path("artifacts/campaign_v2/campaign_v2_runs.jsonl"))
    parser.add_argument("--dist",    nargs="+", metavar="ID")
    parser.add_argument("--resume",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shots",   type=int, default=4096)
    parser.add_argument("--start-from", type=int, default=1, metavar="N")
    args = parser.parse_args()
    main(args.dist, args.config, args.outdir, args.logfile,
         args.resume, args.dry_run, args.shots, args.start_from)
