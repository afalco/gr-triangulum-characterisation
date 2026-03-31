"""
tests/test_distributions.py
============================
Basic sanity tests for the distribution suite used in campaign v2.
Run with:  python -m pytest tests/
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Make the campaign scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "experiments" / "campaign_v2"))

from importlib import import_module
gen = import_module("00_generate_distributions")


class TestDistributions:

    def setup_method(self):
        self.dists = gen.build_distributions()

    def test_all_ids_present(self):
        assert set(self.dists.keys()) == {"D0","D1","D2","D3","D4","D5","D6"}

    def test_all_sum_to_one(self):
        for dist_id, p in self.dists.items():
            assert np.isclose(p.sum(), 1.0, atol=1e-12), \
                f"{dist_id} does not sum to 1: {p.sum()}"

    def test_all_positive(self):
        for dist_id, p in self.dists.items():
            assert np.all(p > 0), \
                f"{dist_id} has zero or negative entry"

    def test_all_length_8(self):
        for dist_id, p in self.dists.items():
            assert len(p) == 8, f"{dist_id} has wrong length: {len(p)}"

    def test_d0_uniform(self):
        p = self.dists["D0"]
        assert np.allclose(p, 0.125), "D0 is not uniform"

    def test_d1_reference(self):
        p = self.dists["D1"]
        expected = np.array([0.05,0.10,0.15,0.20,0.20,0.15,0.10,0.05])
        assert np.allclose(p, expected), "D1 does not match reference values"

    def test_d1_symmetric(self):
        p = self.dists["D1"]
        assert np.allclose(p, p[::-1]), "D1 is not symmetric"

    def test_d5_d6_deterministic(self):
        """D5 and D6 must be identical on repeated calls (fixed seeds)."""
        dists2 = gen.build_distributions()
        assert np.allclose(self.dists["D5"], dists2["D5"]), \
            "D5 is not deterministic"
        assert np.allclose(self.dists["D6"], dists2["D6"]), \
            "D6 is not deterministic"

    def test_entropy_ordering(self):
        """D0 must have the highest entropy (uniform = max entropy)."""
        entropies = {k: gen.shannon_entropy(v)
                     for k, v in self.dists.items()}
        assert entropies["D0"] == pytest.approx(3.0, abs=1e-10), \
            "D0 entropy is not 3 bits"
        for dist_id, h in entropies.items():
            assert h <= 3.0 + 1e-10, \
                f"{dist_id} has entropy > 3 bits: {h}"

    def test_checksums_stable(self, tmp_path):
        """Checksums must be identical to those written by the generator."""
        gen.main(tmp_path)
        import json
        with open(tmp_path / "campaign_distributions.json") as f:
            stored = json.load(f)
        fresh = gen.build_distributions()
        for dist_id, p in fresh.items():
            expected = stored[dist_id]["checksum_sha256"]
            got = gen.checksum(p.tolist())
            assert got == expected, \
                f"{dist_id}: checksum mismatch after re-generation"
