"""
05_characterisation_analysis.py
================================
Cross-distribution characterisation analysis for campaign v2.

Reads the summary artifacts produced by 04_build_artifacts.py and
produces three outputs:

  A) Performance map table: mean FULL-stage fidelity vs. Shannon entropy
     and contrast, for each distribution and each ladder.

  B) Mann-Whitney A vs B comparison table: one row per (dist_id, metric).

  C) Linear regression: Fidelity_FULL ~ H_S + log(Contrast) + max_UCRy_dev
     with R², coefficients, and 95% confidence intervals.

All outputs are written to CSV and printed to the console.

Usage (PowerShell)
------------------
    python 05_characterisation_analysis.py
    python 05_characterisation_analysis.py \\
        --summary artifacts/campaign_v2/summary_by_dist_FULL.csv \\
        --flat    artifacts/campaign_v2/runs_flat_v2.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, t as t_dist


# ---------------------------------------------------------------------------
# GR angle helper (needed for UCRy deviation feature)
# ---------------------------------------------------------------------------

def gr_angles_simple(p: np.ndarray) -> dict:
    """Minimal angle computation (no repo import required)."""
    def safe_acos(num, den):
        if den < 1e-15:
            return 0.0
        return 2 * np.degrees(np.arccos(
            np.sqrt(np.clip(num / den, 0.0, 1.0))))

    sL  = p[0]+p[1]+p[2]+p[3]
    sR  = 1.0 - sL
    sLL = p[0]+p[1]; sLR = p[2]+p[3]
    sRL = p[4]+p[5]; sRR = p[6]+p[7]

    theta2 = {
        "00": safe_acos(p[0], sLL),
        "01": safe_acos(p[2], sLR),
        "10": safe_acos(p[4], sRL),
        "11": safe_acos(p[6], sRR),
    }
    max_ucry_dev = max(abs(v - 90.0) for v in theta2.values())
    return {"theta2": theta2, "max_ucry_dev": max_ucry_dev}


# ---------------------------------------------------------------------------
# A) Performance map
# ---------------------------------------------------------------------------

def performance_map(df_full: pd.DataFrame,
                    config_path: Path) -> pd.DataFrame:
    """
    Build a per-distribution performance table at the FULL stage.
    Returns one row per (dist_id, ladder).
    """
    # Load distribution metadata
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for _, row in df_full.iterrows():
        dist_id = row["dist_id"]
        meta    = raw.get(dist_id, {})
        p       = np.array(meta.get("p", [np.nan]*8))
        angles  = gr_angles_simple(p) if not np.any(np.isnan(p)) else {}

        rows.append({
            "dist_id":           dist_id,
            "ladder":            row.get("ladder"),
            "shannon_entropy":   meta.get("shannon_entropy_bits", np.nan),
            "contrast":          meta.get("contrast", np.nan),
            "log_contrast":      np.log(meta.get("contrast", np.nan))
                                 if meta.get("contrast", 0) > 0 else np.nan,
            "max_ucry_dev":      angles.get("max_ucry_dev", np.nan),
            "n_runs":            row.get("n_runs"),
            "fidelity_mean":     row.get("fidelity_vs_target_mean"),
            "fidelity_std":      row.get("fidelity_vs_target_std"),
            "tv_mean":           row.get("tv_vs_target_mean"),
            "tv_std":            row.get("tv_vs_target_std"),
            "l2_mean":           row.get("l2_vs_target_mean"),
            "l2_std":            row.get("l2_vs_target_std"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B) Mann-Whitney A vs B
# ---------------------------------------------------------------------------

def mann_whitney_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    For each distribution, test ladders A vs B at FULL stage
    across the three primary metrics.
    """
    df = df_runs[(df_runs["stage"] == "FULL")].copy()
    metrics = ["fidelity_vs_target", "tv_vs_target", "l2_vs_target"]
    rows = []

    for dist_id, grp in df.groupby("dist_id"):
        a = grp[grp["ladder"] == "A"]
        b = grp[grp["ladder"] == "B"]
        if len(a) < 2 or len(b) < 2:
            continue
        for metric in metrics:
            xa = a[metric].dropna().values
            xb = b[metric].dropna().values
            if len(xa) < 2 or len(xb) < 2:
                continue
            U, pval = mannwhitneyu(xa, xb, alternative="two-sided")
            r = 1 - 2 * U / (len(xa) * len(xb))  # rank-biserial correlation
            rows.append({
                "dist_id":  dist_id,
                "metric":   metric,
                "n_A":      len(xa),
                "n_B":      len(xb),
                "mean_A":   xa.mean(),
                "mean_B":   xb.mean(),
                "U":        U,
                "p_value":  pval,
                "r":        r,
                "significant_005": pval < 0.05,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C) Regression
# ---------------------------------------------------------------------------

def ols_with_ci(X: np.ndarray, y: np.ndarray,
                alpha: float = 0.05) -> dict:
    """
    OLS regression with R² and 95% confidence intervals.
    X should already include a column of ones for the intercept.
    """
    n, k = X.shape
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Standard errors via (X'X)^{-1} * s^2
    dof = n - k
    s2  = ss_res / dof if dof > 0 else np.nan
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        se  = np.sqrt(np.diag(cov))
        t_crit = t_dist.ppf(1 - alpha / 2, df=dof)
        ci_lo = beta - t_crit * se
        ci_hi = beta + t_crit * se
    except np.linalg.LinAlgError:
        se = ci_lo = ci_hi = np.full_like(beta, np.nan)

    return {"beta": beta, "se": se, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "R2": r2, "dof": dof}


def characterisation_regression(perf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit: Fidelity_FULL ~ intercept + H_S + log(Contrast) + max_UCRy_dev
    Uses mean fidelity over both ladders per distribution.
    """
    df = (
        perf.groupby("dist_id")[
            ["shannon_entropy", "log_contrast", "max_ucry_dev", "fidelity_mean"]
        ].mean().dropna()
    )

    if len(df) < 4:
        print("[WARN] Too few distributions for regression "
              f"({len(df)} available, need >= 4).")
        return pd.DataFrame()

    y = df["fidelity_mean"].values
    X = np.column_stack([
        np.ones(len(df)),
        df["shannon_entropy"].values,
        df["log_contrast"].values,
        df["max_ucry_dev"].values,
    ])
    labels = ["intercept", "H_S", "log_contrast", "max_ucry_dev"]
    res = ols_with_ci(X, y)

    rows = []
    for i, lbl in enumerate(labels):
        rows.append({
            "feature":    lbl,
            "coefficient": res["beta"][i],
            "std_error":   res["se"][i],
            "ci_lo_95":    res["ci_lo"][i],
            "ci_hi_95":    res["ci_hi"][i],
        })
    df_out = pd.DataFrame(rows)
    df_out.attrs["R2"]  = res["R2"]
    df_out.attrs["dof"] = res["dof"]
    return df_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(summary_path: Path, flat_path: Path,
         config_path: Path, outdir: Path) -> None:

    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    df_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    df_flat    = pd.read_csv(flat_path)    if flat_path.exists()    else pd.DataFrame()

    if df_summary.empty and df_flat.empty:
        print("[ERROR] No input data found. Run 04_build_artifacts.py first.")
        return

    # A) Performance map
    if not df_summary.empty:
        perf = performance_map(df_summary, config_path)
        out_a = outdir / "05a_performance_map.csv"
        perf.to_csv(out_a, index=False)
        print(f"[OK] {out_a}")
        print("\n--- Performance map (FULL stage, mean fidelity) ---")
        cols = ["dist_id", "ladder", "shannon_entropy",
                "contrast", "fidelity_mean", "fidelity_std",
                "tv_mean", "n_runs"]
        print(perf[cols].to_string(index=False, float_format="%.4f"))
    else:
        perf = pd.DataFrame()

    # B) Mann-Whitney
    if not df_flat.empty:
        mw = mann_whitney_table(df_flat)
        out_b = outdir / "05b_mann_whitney.csv"
        mw.to_csv(out_b, index=False)
        print(f"\n[OK] {out_b}")
        print("\n--- Mann-Whitney A vs B (FULL stage, fidelity) ---")
        fid_mw = mw[mw["metric"] == "fidelity_vs_target"]
        if not fid_mw.empty:
            print(fid_mw[["dist_id", "mean_A", "mean_B",
                           "U", "p_value", "significant_005"]]
                  .to_string(index=False, float_format="%.4f"))

    # C) Regression
    if not perf.empty:
        reg = characterisation_regression(perf)
        if not reg.empty:
            out_c = outdir / "05c_regression.csv"
            reg.to_csv(out_c, index=False)
            r2  = reg.attrs.get("R2", np.nan)
            dof = reg.attrs.get("dof", "?")
            print(f"\n[OK] {out_c}")
            print(f"\n--- OLS regression  R²={r2:.4f}  dof={dof} ---")
            print(reg.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-distribution characterisation analysis.")
    parser.add_argument("--summary", type=Path,
        default=Path("artifacts/campaign_v2/summary_by_dist_FULL.csv"))
    parser.add_argument("--flat", type=Path,
        default=Path("artifacts/campaign_v2/runs_flat_v2.csv"))
    parser.add_argument("--config", type=Path,
        default=Path("artifacts/campaign_v2/campaign_distributions.json"))
    parser.add_argument("--outdir", type=Path,
        default=Path("artifacts/campaign_v2"))
    args = parser.parse_args()
    main(args.summary, args.flat, args.config, args.outdir)
