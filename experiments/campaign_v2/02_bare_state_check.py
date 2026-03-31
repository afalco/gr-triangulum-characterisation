"""
02_bare_state_check.py
======================
Hardware pre-session check: executes a bare measurement (no gates) on the
SpinQ Triangulum via the SpinQit NMR backend and verifies P(|000>) >= 0.98.

Dependencies
------------
    spinqit   (https://doc.spinq.cn/doc/spinqit/)
    numpy

Usage (PowerShell)
------------------
    $env:SPINQ_IP       = "192.168.1.25"
    $env:SPINQ_PORT     = "55444"
    $env:SPINQ_ACCOUNT  = "my_user"
    $env:SPINQ_PASSWORD = "my_secret_password"

    python 02_bare_state_check.py
    python 02_bare_state_check.py --threshold 0.97 --shots 4096
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

from _spinqit_backend import run_bare_hardware

IDEAL_BARE = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def tv_distance(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))

def fidelity(p, q):
    return float(np.sum(np.sqrt(np.clip(p,0,None)*np.clip(q,0,None)))**2)


def get_conn() -> dict:
    params = {
        "ip":       os.environ.get("SPINQ_IP", ""),
        "port":     os.environ.get("SPINQ_PORT", "55444"),
        "account":  os.environ.get("SPINQ_ACCOUNT", ""),
        "password": os.environ.get("SPINQ_PASSWORD", ""),
        "task_name": "bare_check",
        "task_desc": "Pre-session bare-state check",
    }
    missing = [k for k in ("ip","account","password") if not params[k]]
    if missing:
        sys.exit("[ERROR] Missing env vars: "
                 + ", ".join(f"SPINQ_{k.upper()}" for k in missing))
    return params


def main(threshold: float, shots: int, outdir: Path,
         attempts: int, wait_s: float) -> None:

    outdir.mkdir(parents=True, exist_ok=True)
    conn = get_conn()

    print(f"Bare-state check  {conn['ip']}:{conn['port']}")
    print(f"Threshold P(|000>) >= {threshold}   Shots: {shots}\n")

    for attempt in range(1, attempts + 1):
        ts = datetime.now(timezone.utc).isoformat()
        print(f"Attempt {attempt}/{attempts} ...", end=" ", flush=True)
        try:
            probs = run_bare_hardware(conn, shots=shots)
        except Exception as exc:
            print(f"HARDWARE ERROR: {exc}")
            if attempt < attempts:
                print(f"  Waiting {wait_s}s ...")
                time.sleep(wait_s)
            continue

        p000   = float(probs[0])
        status = "PASS" if p000 >= threshold else "FAIL"
        print(f"{status}  P(|000>)={p000:.4f}  "
              f"TV={tv_distance(IDEAL_BARE,probs):.4f}  "
              f"Fid={fidelity(IDEAL_BARE,probs):.4f}")

        record = {"timestamp": ts, "attempt": attempt, "p_000": p000,
                  "threshold": threshold, "shots": shots,
                  "pass": status == "PASS", "device_ip": conn["ip"]}
        out = outdir / "02_bare_state_check.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if status == "PASS":
            print(f"\n[OK] Device ready. Report: {out}")
            sys.exit(0)
        elif attempt < attempts:
            print(f"  Waiting {wait_s}s for equilibration ...")
            time.sleep(wait_s)

    print(f"\n[FAIL] Below threshold after {attempts} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.98)
    parser.add_argument("--shots",     type=int,   default=4096)
    parser.add_argument("--outdir",    type=Path,
        default=Path("artifacts/campaign_v2"))
    parser.add_argument("--attempts",  type=int,   default=3)
    parser.add_argument("--wait",      type=float, default=120.0)
    args = parser.parse_args()
    main(args.threshold, args.shots, args.outdir, args.attempts, args.wait)
