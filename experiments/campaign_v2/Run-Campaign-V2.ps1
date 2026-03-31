# ============================================================================
# Run-Campaign-V2.ps1
# ============================================================================
# PowerShell orchestrator for the extended Grover-Rudolph characterisation
# campaign v2 on the SpinQ Triangulum.
#
# USAGE
# -----
#   Full hardware campaign:
#       .\Run-Campaign-V2.ps1
#
#   Dry-run (simulator only, no hardware):
#       .\Run-Campaign-V2.ps1 -DryRun
#
#   Resume an interrupted campaign:
#       .\Run-Campaign-V2.ps1 -Resume
#
#   Single distribution (e.g. only D3 and D5):
#       .\Run-Campaign-V2.ps1 -Distributions D3,D5
#
#   Skip hardware pre-check (e.g. second session same day):
#       .\Run-Campaign-V2.ps1 -SkipBareCheck
#
#   Post-processing only (artifacts + analysis from existing log):
#       .\Run-Campaign-V2.ps1 -PostprocessOnly
#
# HARDWARE CREDENTIALS
# --------------------
# Set these variables before running, or let the script prompt for them:
#
#   $env:SPINQ_IP       = "192.168.1.25"
#   $env:SPINQ_PORT     = "55444"
#   $env:SPINQ_ACCOUNT  = "my_user"
#   $env:SPINQ_PASSWORD = "my_secret_password"
#   $env:SPINQ_BITORDER = "MSB->LSB"
#
# PREREQUISITES
# -------------
#   conda activate spinq-gr     (or the equivalent pip environment)
#   The campaigns/v2/ scripts must be located in the repo under
#   experiments/campaign_v2/ or whichever path you specify via -ScriptDir.
# ============================================================================

param(
    [switch]$DryRun,
    [switch]$Resume,
    [switch]$SkipBareCheck,
    [switch]$PostprocessOnly,
    [switch]$VerifyOnly,
    [string[]]$Distributions = @(),
    [string]$ScriptDir  = "$PSScriptRoot",
    [string]$ArtifactDir = "artifacts\campaign_v2",
    [string]$LogFile    = "artifacts\campaign_v2\campaign_v2_runs.jsonl",
    [int]$Shots         = 4096
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Colours ────────────────────────────────────────────────────────────────
function Write-Header  ([string]$msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Step    ([string]$msg) { Write-Host "--- $msg"       -ForegroundColor Yellow }
function Write-Ok      ([string]$msg) { Write-Host "[OK] $msg"      -ForegroundColor Green }
function Write-Warn    ([string]$msg) { Write-Host "[WARN] $msg"    -ForegroundColor DarkYellow }
function Write-Err     ([string]$msg) { Write-Host "[ERROR] $msg"   -ForegroundColor Red }

# ─── Helper: run a Python script and check exit code ────────────────────────
function Invoke-Python {
    param(
        [string]$Script,
        [string[]]$Args = @()
    )
    Write-Step "python $Script $($Args -join ' ')"
    $fullPath = Join-Path $ScriptDir $Script
    & python $fullPath @Args
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Script $Script exited with code $LASTEXITCODE — aborting."
        exit $LASTEXITCODE
    }
    Write-Ok "$Script completed."
}

# ─── Hardware credentials ────────────────────────────────────────────────────
function Set-HardwareEnv {
    if (-not $DryRun) {
        if (-not $env:SPINQ_IP) {
            $env:SPINQ_IP = Read-Host "Enter Triangulum IP address"
        }
        if (-not $env:SPINQ_PORT)     { $env:SPINQ_PORT     = "55444" }
        if (-not $env:SPINQ_ACCOUNT) {
            $env:SPINQ_ACCOUNT  = Read-Host "Enter SpinQ account name"
        }
        if (-not $env:SPINQ_PASSWORD) {
            $secPwd = Read-Host "Enter SpinQ password" -AsSecureString
            $bstr   = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secPwd)
            $env:SPINQ_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
        if (-not $env:SPINQ_BITORDER) { $env:SPINQ_BITORDER = "MSB->LSB" }

        Write-Ok ("Hardware target: {0}:{1}  account: {2}" -f `
            $env:SPINQ_IP, $env:SPINQ_PORT, $env:SPINQ_ACCOUNT)
    }
    else {
        Write-Warn "DRY-RUN mode — no hardware credentials required."
    }
}

# ─── Build python argument lists ────────────────────────────────────────────
function Get-CommonArgs {
    $a = @("--outdir", $ArtifactDir)
    if ($Distributions.Count -gt 0) {
        $a += "--dist"
        $a += $Distributions
    }
    return $a
}

function Get-RunArgs {
    $a  = Get-CommonArgs
    $a += @("--logfile", $LogFile, "--shots", "$Shots")
    if ($DryRun)  { $a += "--dry-run" }
    if ($Resume)  { $a += "--resume"  }
    return $a
}

# ─── Timestamp banner ────────────────────────────────────────────────────────
function Write-Timestamp ([string]$label) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] $label" -ForegroundColor DarkGray
}

# ============================================================================
# MAIN
# ============================================================================

Write-Header "Grover-Rudolph Campaign v2 — SpinQ Triangulum"
Write-Timestamp "Pipeline start"

$configPath = Join-Path $ArtifactDir "campaign_distributions.json"
$verifyFlag = Join-Path $ArtifactDir ".simulation_verified"

# ── 0. Environment & credentials ────────────────────────────────────────────
Set-HardwareEnv

# ── STEP 0: Generate distributions ──────────────────────────────────────────
if (-not $PostprocessOnly) {
    Write-Header "STEP 0 — Generate / verify target distributions"

    if (Test-Path $configPath) {
        Write-Step "campaign_distributions.json found — verifying checksums ..."
        Invoke-Python "00_generate_distributions.py" @("--outdir", $ArtifactDir, "--verify")
    }
    else {
        Write-Step "Generating distributions for the first time ..."
        Invoke-Python "00_generate_distributions.py" @("--outdir", $ArtifactDir)
    }
}

# ── STEP 1: Simulator verification ──────────────────────────────────────────
if (-not $PostprocessOnly) {
    Write-Header "STEP 1 — Ideal simulator pre-flight verification"

    if ((Test-Path $verifyFlag) -and -not $VerifyOnly) {
        Write-Warn "Simulation already verified (delete $verifyFlag to re-run)."
    }
    else {
        $simArgs = @("--outdir", $ArtifactDir, "--config", $configPath)
        if ($Distributions.Count -gt 0) {
            $simArgs += "--dist"
            $simArgs += $Distributions
        }
        Invoke-Python "01_verify_simulations.py" $simArgs
        # Leave a marker so subsequent runs skip this step
        New-Item -ItemType File -Path $verifyFlag -Force | Out-Null
    }

    if ($VerifyOnly) {
        Write-Ok "Verification-only run complete."
        exit 0
    }
}

# ── STEP 2: Bare-state hardware check ───────────────────────────────────────
if (-not $DryRun -and -not $SkipBareCheck -and -not $PostprocessOnly) {
    Write-Header "STEP 2 — Hardware bare-state check"
    Invoke-Python "02_bare_state_check.py" @("--outdir", $ArtifactDir, "--shots", "$Shots")
}
elseif ($DryRun) {
    Write-Warn "STEP 2 skipped (dry-run mode)."
}
elseif ($SkipBareCheck) {
    Write-Warn "STEP 2 skipped (--SkipBareCheck flag)."
}

# ── STEP 3: Run campaign ─────────────────────────────────────────────────────
if (-not $PostprocessOnly) {
    Write-Header "STEP 3 — Execute campaign (1050 runs)"
    Write-Timestamp "Hardware execution start"

    $runArgs = Get-RunArgs
    Invoke-Python "03_run_campaign.py" $runArgs

    Write-Timestamp "Hardware execution end"
}

# ── STEP 4: Build artifacts ──────────────────────────────────────────────────
Write-Header "STEP 4 — Build data artifacts"
Invoke-Python "04_build_artifacts.py" @(
    "--logfile", $LogFile,
    "--outdir",  $ArtifactDir
)

# ── STEP 5: Characterisation analysis ───────────────────────────────────────
Write-Header "STEP 5 — Cross-distribution characterisation analysis"
Invoke-Python "05_characterisation_analysis.py" @(
    "--summary", (Join-Path $ArtifactDir "summary_by_dist_FULL.csv"),
    "--flat",    (Join-Path $ArtifactDir "runs_flat_v2.csv"),
    "--config",  $configPath,
    "--outdir",  $ArtifactDir
)

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Header "Campaign v2 complete"
Write-Timestamp "Pipeline end"

Write-Host ""
Write-Host "Artifacts in: $ArtifactDir" -ForegroundColor Cyan
Write-Host ""
Write-Host "  runs_flat_v2.csv              (1050 rows)" -ForegroundColor White
Write-Host "  summary_by_dist_stage.csv     (28 rows)"  -ForegroundColor White
Write-Host "  summary_by_dist_FULL.csv      (14 rows)"  -ForegroundColor White
Write-Host "  circuits_flat_v2.csv          (28 rows)"  -ForegroundColor White
Write-Host "  campaign_v2_runs_clean.jsonl  (full log)" -ForegroundColor White
Write-Host "  05a_performance_map.csv"                  -ForegroundColor White
Write-Host "  05b_mann_whitney.csv"                     -ForegroundColor White
Write-Host "  05c_regression.csv"                       -ForegroundColor White
Write-Host ""
