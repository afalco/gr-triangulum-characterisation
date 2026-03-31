"""
02_bare_state_check.py
======================
Hardware pre-session check: executes a bare measurement (no gates)
on the SpinQ Triangulum and verifies that P(|000>) >= 0.98.

If the threshold is not met the script exits with a non-zero code
so that the PowerShell orchestrator can halt before submitting any
circuit.

Usage (PowerShell)
------------------
    $env:SPINQ_IP       = "192.168.1.25"
    $env:SPINQ_PORT     = "55444"
    $env:SPINQ_ACCOUNT  = "my_user"
    $env:SPINQ_PASSWORD = "my_secret_password"
    $env:SPINQ_BITORDER = "MSB->LSB"

    python 02_bare_state_check.py
    python 02_bare_state_check.py --threshold 0.97 --shots 4096
"""

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

from gr.backends import run_nmr_probs_robust    # hardware runner with retries
from gr.circuit import build_bare_circuit       # zero-gate identity circuit
from gr.metrics import tv_distance, fidelity
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.98
IDEAL_BARE = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # |000>


def get_connection_params() -> dict:
    params = {
        "ip":       os.environ.get("SPINQ_IP", ""),
        "port":     int(os.environ.get("SPINQ_PORT", "55444")),
        "account":  os.environ.get("SPINQ_ACCOUNT", ""),
        "password": os.environ.get("SPINQ_PASSWORD", ""),
        "bitorder": os.environ.get("SPINQ_BITORDER", "MSB->LSB"),
    }
    missing = [k for k in ("ip", "account", "password") if not params[k]]
    if missing:
        sys.exit(
            f"[ERROR] Missing environment variables: "
            + ", ".join(f"SPINQ_{k.upper()}" for k in missing)
            + "\nSet them in PowerShell before running this script."
        )
    return params


def main(threshold: float, shots: int, outdir: Path,
         n_attempts: int, wait_s: float) -> None:

    outdir.mkdir(parents=True, exist_ok=True)
    conn = get_connection_params()

    print(f"Bare-state check on {conn['ip']}:{conn['port']}")
    print(f"Threshold: P(|000>) >= {threshold}   Shots: {shots}")
    print()

    circ = build_bare_circuit(n_qubits=3)

    for attempt in range(1, n_attempts + 1):
        ts_start = datetime.now(timezone.utc).isoformat()
        print(f"Attempt {attempt}/{n_attempts} ...", end=" ", flush=True)

        try:
            probs = run_nmr_probs_robust(
                circuit=circ,
                shots=shots,
                conn=conn,
            )
        except Exception as exc:
            print(f"HARDWARE ERROR: {exc}")
            if attempt < n_attempts:
                print(f"  Waiting {wait_s}s before retry ...")
                time.sleep(wait_s)
            continue

        ts_end = datetime.now(timezone.utc).isoformat()
        p000  = float(probs[0])
        tv_v  = tv_distance(IDEAL_BARE, probs)
        fid_v = fidelity(IDEAL_BARE, probs)

        status = "PASS" if p000 >= threshold else "FAIL"
        print(f"{status}  P(|000>)={p000:.4f}  TV={tv_v:.4f}  Fid={fid_v:.4f}")

        record = {
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "attempt": attempt,
            "p_000": p000,
            "tv_vs_ideal": tv_v,
            "fidelity_vs_ideal": fid_v,
            "threshold": threshold,
            "shots": shots,
            "pass": status == "PASS",
            "device_ip": conn["ip"],
        }

        report_path = outdir / "02_bare_state_check.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        if status == "PASS":
            print(f"\n[OK] Device is ready. Report: {report_path}")
            sys.exit(0)
        else:
            print(f"  P(|000>) = {p000:.4f} < {threshold}")
            if attempt < n_attempts:
                print(f"  Waiting {wait_s}s for thermal equilibration ...")
                time.sleep(wait_s)

    print(
        f"\n[FAIL] Bare-state fidelity below threshold after {n_attempts} attempts.\n"
        "Allow the device to equilibrate (>= 30 min) and retry."
    )
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hardware pre-session bare-state check.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--outdir", type=Path,
                        default=Path("artifacts/campaign_v2"))
    parser.add_argument("--attempts", type=int, default=3,
                        help="Max hardware attempts (default 3)")
    parser.add_argument("--wait", type=float, default=120.0,
                        help="Seconds to wait between failed attempts (default 120)")
    args = parser.parse_args()

    main(
        threshold=args.threshold,
        shots=args.shots,
        outdir=args.outdir,
        n_attempts=args.attempts,
        wait_s=args.wait,
    )
