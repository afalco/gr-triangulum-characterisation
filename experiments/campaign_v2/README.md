# Campaign v2 — Quick-start guide (PowerShell)

Extended Grover–Rudolph characterisation campaign on the SpinQ Triangulum.

## Repository location

Place the five Python scripts and the PowerShell orchestrator inside
the repo under `experiments/campaign_v2/`:

```
grover-rudolph-practical-implementation/
└── experiments/
    └── campaign_v2/
        ├── 00_generate_distributions.py
        ├── 01_verify_simulations.py
        ├── 02_bare_state_check.py
        ├── 03_run_campaign.py
        ├── 04_build_artifacts.py
        ├── 05_characterisation_analysis.py
        └── Run-Campaign-V2.ps1
```

## Environment setup

```powershell
conda activate spinq-gr
```

## Hardware credentials

Set once per PowerShell session (or add to your profile):

```powershell
$env:SPINQ_IP       = "192.168.1.25"
$env:SPINQ_PORT     = "55444"
$env:SPINQ_ACCOUNT  = "my_user"
$env:SPINQ_PASSWORD = "my_secret_password"
$env:SPINQ_BITORDER = "MSB->LSB"
```

---

## Typical workflows

### Full campaign (first time)

```powershell
cd path\to\grover-rudolph-practical-implementation\experiments\campaign_v2
.\Run-Campaign-V2.ps1
```

Pipeline steps executed automatically:
1. Generate & checksum the 7 distributions → `campaign_distributions.json`
2. Ideal simulator verification for all distributions and ladders
3. Hardware bare-state check (P(|000⟩) ≥ 0.98)
4. 1050 hardware runs, interleaved across distributions
5. Build CSV/JSONL artifacts
6. Cross-distribution analysis (performance map, Mann–Whitney, regression)

---

### Dry-run (simulator only, no hardware)

```powershell
.\Run-Campaign-V2.ps1 -DryRun
```

Runs the full pipeline using a simulated hardware backend.
Useful for testing the pipeline before connecting the Triangulum.

---

### Resume an interrupted campaign

```powershell
.\Run-Campaign-V2.ps1 -Resume
```

Reads the existing `campaign_v2_runs.jsonl` and skips any
`run_name` already present with `status = "completed"`.

---

### Run only specific distributions

```powershell
.\Run-Campaign-V2.ps1 -Distributions D3,D5,D6
```

---

### Post-processing only (no new hardware runs)

```powershell
.\Run-Campaign-V2.ps1 -PostprocessOnly
```

Reads the existing log and re-builds all artifacts and analysis.
Useful after manually adding or correcting run records.

---

### Verify simulations only

```powershell
.\Run-Campaign-V2.ps1 -VerifyOnly
```

---

### Skip the bare-state check (second session same day)

```powershell
.\Run-Campaign-V2.ps1 -SkipBareCheck
```

---

## Running individual scripts

Every script can also be run standalone:

```powershell
# Generate distributions
python 00_generate_distributions.py

# Verify checksums
python 00_generate_distributions.py --verify

# Pre-flight simulation check for D3 and D5 only
python 01_verify_simulations.py --dist D3 D5

# Bare-state hardware check
python 02_bare_state_check.py --threshold 0.90

# Run only D3 with resume
python 03_run_campaign.py --dist D3 --resume --shots 4096

# Build artifacts from existing log
python 04_build_artifacts.py --logfile artifacts\campaign_v2\campaign_v2_runs.jsonl

# Analysis only
python 05_characterisation_analysis.py
```

---

## Output artifacts

| File | Rows | Description |
|------|------|-------------|
| `campaign_distributions.json` | 7 | Canonical distribution definitions + checksums |
| `campaign_v2_runs.jsonl` | 1050 | Raw per-run log (real-time, append-only) |
| `campaign_v2_runs_clean.jsonl` | 1050 | Re-serialised completed records |
| `runs_flat_v2.csv` | 1050 | Flat table: all metrics, probs, marginals |
| `summary_by_dist_stage.csv` | 28 | Means/stds by (dist\_id, stage, ladder) |
| `summary_by_dist_FULL.csv` | 14 | FULL-stage means/stds by (dist\_id, ladder) |
| `circuits_flat_v2.csv` | 28 | Gate counts by (dist\_id, stage, ladder) |
| `05a_performance_map.csv` | 14 | Fidelity vs entropy/contrast per distribution |
| `05b_mann_whitney.csv` | ≤42 | Mann–Whitney A vs B results |
| `05c_regression.csv` | 4 | OLS regression coefficients |

---

## Notes

- The log file is **append-only**; runs are never deleted.
  Use `--resume` to continue from where you left off.
- A bare-state health check runs automatically every 100 hardware
  runs; the campaign pauses if P(|000⟩) < 0.97.
- Credentials are read from environment variables and never written
  to disk. Do not commit them to the repository.
- Distribution D1 is the first-campaign reference. After the v2 run,
  its results should agree with the v1 values within ±2σ;
  deviations indicate hardware drift between sessions.
