# Grover–Rudolph Triangulum Characterisation Campaign (v2)

This repository contains the full experimental pipeline for the **extended
Grover–Rudolph hardware characterisation campaign** on the
**SpinQ Triangulum 3-qubit NMR quantum computer**, conducted jointly by
Universidad CEU Cardenal Herrera and SpinQ Technology.

The campaign systematically probes the state-preparation capability of the
Triangulum across a heterogeneous suite of **seven target probability
distributions**, covering the full range of Shannon entropy, symmetry,
concentration, and structural complexity that the 3-qubit Grover–Rudolph
construction can address.

---

## Companion repositories and papers

| Resource | Link |
|---|---|
| Core GR implementation (v1 campaign) | [afalco/grover-rudolph-practical-implementation](https://github.com/afalco/grover-rudolph-practical-implementation) |
| Theoretical paper | arXiv:2601.17930 |
| Experimental paper (in preparation) | *Practical Grover–Rudolph state preparation on a 3-qubit NMR quantum computer* |

The `gr/` package from the core repository is a **required dependency**
(see [Installation](#installation)).

---

## What this campaign adds

The first campaign (v1) characterised hardware performance for a single
symmetric benchmark distribution. This extended campaign answers three
new questions:

1. **How does fidelity vary with target complexity?**
   We produce a quantitative performance map: mean FULL-stage classical
   fidelity vs. Shannon entropy of the target distribution.

2. **Is hardware degradation generic or circuit-structure-specific?**
   By comparing seven qualitatively different distributions we can
   separate device-level noise from UCRy-block-specific error.

3. **Does Gray-code ladder ordering matter across different targets?**
   The A/B ladder comparison from v1 is extended to all seven distributions
   with a Mann–Whitney test per distribution.

---

## Target distribution suite

| ID | Type | Shannon entropy (bits) | Contrast |
|---|---|---|---|
| D0 | Uniform | 3.00 | 1.0 |
| D1 | Symmetric (v1 reference) | 2.77 | 4.0 |
| D2 | Monotone decreasing | 2.57 | 35.0 |
| D3 | Bimodal | 2.17 | 20.0 |
| D4 | Unimodal / bell | 2.53 | 15.5 |
| D5 | Random sparse (Dirichlet α=0.5, seed 42) | ≈ 1.8 | ≫ 1 |
| D6 | Random generic (Dirichlet α=1.0, seed 137) | ≈ 2.5 | ∼ 10 |

D1 is retained from v1 as an inter-campaign anchor point.
D5 and D6 are generated deterministically from fixed seeds
(see `experiments/campaign_v2/00_generate_distributions.py`).

---

## Campaign structure

- **Total runs:** 1 050
- **Per distribution:** 150 runs (50 per stage × 3 stages)
- **Per stage:** L0 (50), L01 (50), FULL (50 = 25×A + 25×B)
- **Execution:** fully automated, interleaved across distributions
  to mitigate hardware drift

---

## Repository layout

```
gr-triangulum-characterisation/
├── README.md
├── LICENSE
├── .gitignore
├── environment.yml             # conda environment (spinq-gr)
├── requirements.txt            # pip fallback
│
├── experiments/
│   └── campaign_v2/            # ← all campaign scripts
│       ├── 00_generate_distributions.py
│       ├── 01_verify_simulations.py
│       ├── 02_bare_state_check.py
│       ├── 03_run_campaign.py
│       ├── 04_build_artifacts.py
│       ├── 05_characterisation_analysis.py
│       ├── Run-Campaign-V2.ps1     # PowerShell orchestrator
│       └── README.md
│
├── docs/
│   ├── experimental_protocol_v2.pdf   # this campaign protocol
│   └── experimental_protocol_v2.tex   # LaTeX source
│
├── artifacts/                  # generated outputs — gitignored
│   └── campaign_v2/
│       ├── campaign_distributions.json
│       ├── campaign_v2_runs.jsonl
│       ├── runs_flat_v2.csv
│       ├── summary_by_dist_stage.csv
│       ├── summary_by_dist_FULL.csv
│       ├── circuits_flat_v2.csv
│       ├── 05a_performance_map.csv
│       ├── 05b_mann_whitney.csv
│       └── 05c_regression.csv
│
└── tests/
    └── test_distributions.py
```

---

## Installation

### 1. Clone this repository

```powershell
git clone https://github.com/afalco/gr-triangulum-characterisation.git
cd gr-triangulum-characterisation
```

### 2. Install the core GR package

The `gr/` package lives in the companion repository and must be
available on the Python path. The simplest approach is a local
editable install:

```powershell
# Clone the core repo next to this one (or wherever you prefer)
git clone https://github.com/afalco/grover-rudolph-practical-implementation.git

# Install it as an editable package
pip install -e ..\grover-rudolph-practical-implementation
```

Alternatively, set `PYTHONPATH` to point to the core repo root:

```powershell
$env:PYTHONPATH = "..\grover-rudolph-practical-implementation"
```

### 3. Create the conda environment

```powershell
conda env create -f environment.yml
conda activate spinq-gr
```

Or with pip:

```powershell
pip install -r requirements.txt
```

---

## Quick start (PowerShell)

### Set hardware credentials

```powershell
$env:SPINQ_IP       = "192.168.1.25"
$env:SPINQ_PORT     = "55444"
$env:SPINQ_ACCOUNT  = "my_user"
$env:SPINQ_PASSWORD = "my_secret_password"
$env:SPINQ_BITORDER = "MSB->LSB"
```

### Run the full campaign

```powershell
cd experiments\campaign_v2
.\Run-Campaign-V2.ps1
```

### Dry-run (simulator only, no hardware)

```powershell
.\Run-Campaign-V2.ps1 -DryRun
```

### Resume an interrupted campaign

```powershell
.\Run-Campaign-V2.ps1 -Resume
```

### Post-processing only (rebuild artifacts from existing log)

```powershell
.\Run-Campaign-V2.ps1 -PostprocessOnly
```

See [`experiments/campaign_v2/README.md`](experiments/campaign_v2/README.md)
for the full list of options and individual script usage.

---

## Hardware credentials — security note

Never commit credentials to the repository.
Always use environment variables as shown above.
The `.gitignore` excludes `*.env`, `secrets.*`, and `credentials.*`.

---

## Experimental protocol

The complete self-contained experimental protocol (v2.0) is available as
a PDF in `docs/experimental_protocol_v2.pdf`. It describes every step
from distribution specification through circuit compilation, hardware
execution, data normalization, and cross-distribution analysis, with
sufficient detail to allow independent replication on any compatible
3-qubit NMR platform.

---

## Authors

- Antonio Falcó — Universidad CEU Cardenal Herrera, Spain
- Daniela Falcó-Pomares
- Juan-Carlos Latorre
- Yuefeng Feng — SpinQ Technology

---

## License

MIT — see [LICENSE](LICENSE).
