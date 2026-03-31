"""
00_generate_distributions.py
============================
Generates and persists the seven target probability distributions
for the extended Grover-Rudolph characterisation campaign (v2).

Distributions D5 and D6 are drawn from Dirichlet priors with fixed
random seeds so that every participant uses identical targets.

Outputs
-------
campaign_distributions.json   -- canonical distribution file (shared)
campaign_distributions_check.txt -- SHA-256 checksums for verification

Usage (PowerShell)
------------------
    python 00_generate_distributions.py
    python 00_generate_distributions.py --outdir artifacts/campaign_v2
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Distribution definitions
# ---------------------------------------------------------------------------

def _dirichlet(alpha: float, seed: int, n: int = 8,
               clip_min: float = 1e-4) -> np.ndarray:
    """Draw a Dirichlet sample with fixed seed and clip near-zero entries."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n, alpha))
    p = np.clip(p, clip_min, None)
    p /= p.sum()
    return p


def build_distributions() -> dict[str, np.ndarray]:
    """Return the canonical distribution suite as a dict id -> ndarray."""
    return {
        "D0": np.full(8, 0.125),                                         # uniform
        "D1": np.array([0.05, 0.10, 0.15, 0.20,
                        0.20, 0.15, 0.10, 0.05]),                        # symmetric ref
        "D2": np.array([0.35, 0.25, 0.15, 0.10,
                        0.07, 0.04, 0.03, 0.01]),                        # monotone
        "D3": np.array([0.02, 0.03, 0.05, 0.40,
                        0.40, 0.05, 0.03, 0.02]),                        # bimodal
        "D4": np.array([0.02, 0.05, 0.12, 0.31,
                        0.31, 0.12, 0.05, 0.02]),                        # bell
        "D5": _dirichlet(alpha=0.5, seed=42),                            # sparse random
        "D6": _dirichlet(alpha=1.0, seed=137),                           # generic random
    }


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits."""
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def contrast(p: np.ndarray) -> float:
    """max / min over nonzero entries."""
    nz = p[p > 0]
    return float(nz.max() / nz.min())


def checksum(vec: list[float]) -> str:
    """SHA-256 of the JSON-serialised vector (15 decimal places, compact)."""
    s = json.dumps([f"{v:.15f}" for v in vec], separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dists = build_distributions()

    # Validate
    for dist_id, p in dists.items():
        assert len(p) == 8, f"{dist_id}: wrong length"
        assert np.isclose(p.sum(), 1.0, atol=1e-12), f"{dist_id}: does not sum to 1"
        assert np.all(p > 0), f"{dist_id}: zero or negative entry"

    # Build serialisable record
    record = {}
    for dist_id, p in dists.items():
        record[dist_id] = {
            "p": p.tolist(),
            "shannon_entropy_bits": shannon_entropy(p),
            "contrast": contrast(p),
            "checksum_sha256": checksum(p.tolist()),
        }

    # Write canonical JSON
    out_json = outdir / "campaign_distributions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[OK] Distributions written to {out_json}")

    # Write human-readable checksum file
    out_check = outdir / "campaign_distributions_check.txt"
    with open(out_check, "w", encoding="utf-8") as f:
        f.write("# Campaign v2 distribution checksums\n")
        f.write("# Verify with: python 00_generate_distributions.py --verify\n\n")
        f.write(f"{'ID':<4}  {'H_S (bits)':>12}  {'Contrast':>10}  SHA-256\n")
        f.write("-" * 80 + "\n")
        for dist_id, meta in record.items():
            f.write(
                f"{dist_id:<4}  {meta['shannon_entropy_bits']:>12.4f}"
                f"  {meta['contrast']:>10.2f}  {meta['checksum_sha256']}\n"
            )
    print(f"[OK] Checksums written to {out_check}")

    # Console summary
    print()
    print(f"{'ID':<4}  {'H_S (bits)':>12}  {'Contrast':>10}")
    print("-" * 32)
    for dist_id, meta in record.items():
        print(f"{dist_id:<4}  {meta['shannon_entropy_bits']:>12.4f}"
              f"  {meta['contrast']:>10.2f}")


def verify(outdir: Path) -> None:
    """Re-generate and compare checksums against the persisted file."""
    canon = outdir / "campaign_distributions.json"
    if not canon.exists():
        sys.exit(f"[ERROR] {canon} not found. Run without --verify first.")
    with open(canon, encoding="utf-8") as f:
        stored = json.load(f)
    fresh = build_distributions()
    all_ok = True
    for dist_id, p in fresh.items():
        expected = stored[dist_id]["checksum_sha256"]
        got = checksum(p.tolist())
        status = "OK" if got == expected else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"{dist_id}: {status}")
    if all_ok:
        print("\nAll checksums match. Distributions are consistent.")
    else:
        sys.exit("\nChecksum mismatch detected. Do NOT proceed to hardware.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and persist campaign v2 distributions.")
    parser.add_argument("--outdir", type=Path,
                        default=Path("artifacts/campaign_v2"),
                        help="Output directory (default: artifacts/campaign_v2)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify checksums against existing file instead of writing")
    args = parser.parse_args()

    if args.verify:
        verify(args.outdir)
    else:
        main(args.outdir)
