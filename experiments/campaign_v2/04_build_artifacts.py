"""
04_build_artifacts.py
=====================
Reads the raw JSONL campaign log produced by 03_run_campaign.py and
builds five clean tabular artifacts:

  runs_flat_v2.csv              -- one row per completed run (all fields)
  summary_by_dist_stage.csv     -- means/stds by (dist_id, stage, ladder)
  summary_by_dist_FULL.csv      -- FULL-stage summary by dist_id only
  circuits_flat_v2.csv          -- one row per (dist_id, stage, ladder) group
  campaign_v2_runs_clean.jsonl  -- re-serialised clean records

Usage (PowerShell)
------------------
    python 04_build_artifacts.py
    python 04_build_artifacts.py --logfile path/to/campaign_v2_runs.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STATES  = ["000", "001", "010", "011", "100", "101", "110", "111"]
METRICS = ["tv_vs_sim", "l2_vs_sim", "fidelity_vs_sim",
           "tv_vs_target", "l2_vs_target", "fidelity_vs_target"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    completed = [r for r in records if r.get("status") == "completed"]
    print(f"[Load] {len(records)} total records, "
          f"{len(completed)} completed, "
          f"{len(records)-len(completed)} failed/skipped.")
    return completed


# ---------------------------------------------------------------------------
# runs_flat_v2.csv
# ---------------------------------------------------------------------------

def build_runs_flat(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "run_name":        r["run_name"],
            "run_index":       r.get("run_index"),
            "dist_id":         r["dist_id"],
            "stage":           r["stage"],
            "ladder":          r["ladder"],
            "repeat":          r["repeat"],
            "platform_run_id": r.get("platform_run_id"),
            "status":          r["status"],
            "created":         r.get("created"),
            "end":             r.get("end"),
            "duration_s":      r.get("duration_s"),
            "shannon_entropy": r.get("shannon_entropy"),
            "contrast":        r.get("contrast"),
            "n_qubits":        r.get("n_qubits", 3),
            "n_gates":         r.get("n_gates"),
            "n_ry":            r.get("n_ry"),
            "n_x":             r.get("n_x"),
            "n_cnot":          r.get("n_cnot"),
        }
        # Metrics
        for m in METRICS:
            row[m] = r.get(m)
        # Probabilities
        sim_p = r.get("sim_probs", {})
        exp_p = r.get("exp_probs", {})
        for s in STATES:
            row[f"sim_p_{s}"] = sim_p.get(s)
            row[f"exp_p_{s}"] = exp_p.get(s)
        # Marginals
        sim_m = r.get("sim_marginals", {})
        exp_m = r.get("exp_marginals", {})
        for qi in range(3):
            row[f"sim_q{qi}_p0"] = sim_m.get(f"q{qi}", {}).get("p0")
            row[f"sim_q{qi}_p1"] = sim_m.get(f"q{qi}", {}).get("p1")
            row[f"exp_q{qi}_p0"] = exp_m.get(f"q{qi}", {}).get("p0")
            row[f"exp_q{qi}_p1"] = exp_m.get(f"q{qi}", {}).get("p1")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# summary tables
# ---------------------------------------------------------------------------

def build_summary(df: pd.DataFrame,
                  group_cols: list[str]) -> pd.DataFrame:
    agg_metrics = {m: ["mean", "std"] for m in METRICS}
    agg_extra   = {"duration_s": ["mean", "std"], "run_name": "count"}
    agg_probs   = {f"exp_p_{s}": "mean" for s in STATES}
    agg_probs.update({f"sim_p_{s}": "mean" for s in STATES})

    all_agg = {**agg_metrics, **agg_extra, **agg_probs}
    summary = df.groupby(group_cols).agg(all_agg)

    # Flatten MultiIndex columns
    summary.columns = [
        f"{col}_{stat}" if stat not in ("count",) else f"n_runs"
        for col, stat in summary.columns
    ]
    summary = summary.rename(columns={"run_name_count": "n_runs"})
    return summary.reset_index()


# ---------------------------------------------------------------------------
# circuits_flat_v2.csv
# ---------------------------------------------------------------------------

def build_circuits_flat(df: pd.DataFrame) -> pd.DataFrame:
    gate_cols = ["n_gates", "n_ry", "n_x", "n_cnot"]
    return (
        df.groupby(["dist_id", "stage", "ladder"])[gate_cols]
        .first()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(logfile: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(logfile)

    if not records:
        print("[WARN] No completed records found. Artifacts will be empty.")

    # 1. Flat runs table
    df = build_runs_flat(records)
    out1 = outdir / "runs_flat_v2.csv"
    df.to_csv(out1, index=False)
    print(f"[OK] {out1}  ({len(df)} rows)")

    # 2. Summary by (dist_id, stage, ladder)
    summary_full = build_summary(df, ["dist_id", "stage", "ladder"])
    out2 = outdir / "summary_by_dist_stage.csv"
    summary_full.to_csv(out2, index=False)
    print(f"[OK] {out2}  ({len(summary_full)} rows)")

    # 3. Summary at FULL stage only (dist_id level, both ladders)
    df_full = df[df["stage"] == "FULL"].copy()
    if not df_full.empty:
        summary_full_stage = build_summary(df_full, ["dist_id", "ladder"])
        out3 = outdir / "summary_by_dist_FULL.csv"
        summary_full_stage.to_csv(out3, index=False)
        print(f"[OK] {out3}  ({len(summary_full_stage)} rows)")
    else:
        print("[WARN] No FULL-stage runs found; skipping summary_by_dist_FULL.csv")

    # 4. Circuits flat
    df_circ = build_circuits_flat(df)
    out4 = outdir / "circuits_flat_v2.csv"
    df_circ.to_csv(out4, index=False)
    print(f"[OK] {out4}  ({len(df_circ)} rows)")

    # 5. Clean JSONL (re-serialise completed records)
    out5 = outdir / "campaign_v2_runs_clean.jsonl"
    with open(out5, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {out5}  ({len(records)} records)")

    # Quick sanity summary
    if not df.empty:
        print("\n--- Stage-wise fidelity summary (all distributions) ---")
        tbl = (
            df.groupby(["stage"])["fidelity_vs_target"]
            .agg(mean="mean", std="std", n="count")
            .round(4)
        )
        print(tbl.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build campaign v2 data artifacts from JSONL log.")
    parser.add_argument("--logfile", type=Path,
        default=Path("artifacts/campaign_v2/campaign_v2_runs.jsonl"))
    parser.add_argument("--outdir", type=Path,
        default=Path("artifacts/campaign_v2"))
    args = parser.parse_args()
    main(args.logfile, args.outdir)
