# ============================================================================
# Setup-Repo.ps1
# ============================================================================
# Creates the gr-triangulum-characterisation GitHub repository from scratch,
# populates it with the campaign v2 files, and pushes it to GitHub.
#
# Run this script ONCE from the folder that CONTAINS the two local
# directories produced in the previous steps:
#
#   your-working-folder\
#       gr-triangulum-characterisation\   ← repo skeleton (this script's $RepoDir)
#       campaigns_v2\                     ← the 6 Python scripts + .ps1
#
# PREREQUISITES
# -------------
#   1. Git installed and on PATH
#         winget install --id Git.Git
#   2. GitHub CLI installed and authenticated
#         winget install --id GitHub.cli
#         gh auth login
#   3. Python environment active
#         conda activate spinq-gr
#   4. The two local directories described above exist
#
# USAGE
# -----
#   Full setup (create repo on GitHub, commit, push):
#       .\Setup-Repo.ps1
#
#   Dry-run (build local repo only, no GitHub operations):
#       .\Setup-Repo.ps1 -LocalOnly
#
#   Use a different GitHub account (e.g. for a fork):
#       .\Setup-Repo.ps1 -GitHubUser yuefeng-spinq
#
#   Set the repo to private initially:
#       .\Setup-Repo.ps1 -Private
# ============================================================================

param(
    [string]$GitHubUser  = "afalco",
    [string]$RepoName    = "gr-triangulum-characterisation",
    [string]$RepoDir     = ".\gr-triangulum-characterisation",
    [string]$ScriptSrc   = ".\campaigns_v2",
    [string]$DocsSrc     = ".",        # folder containing the protocol PDFs/TeX
    [string]$CommitMsg   = "Add campaign v2: multi-distribution GR characterisation pipeline",
    [switch]$LocalOnly,
    [switch]$Private
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ────────────────────────────────────────────────────────────
function Write-Header  ([string]$m) { Write-Host "`n=== $m ===" -ForegroundColor Cyan }
function Write-Step    ([string]$m) { Write-Host "  >> $m"      -ForegroundColor Yellow }
function Write-Ok      ([string]$m) { Write-Host "  [OK] $m"    -ForegroundColor Green }
function Write-Warn    ([string]$m) { Write-Host "  [WARN] $m"  -ForegroundColor DarkYellow }
function Write-Fail    ([string]$m) { Write-Host "  [FAIL] $m"  -ForegroundColor Red; exit 1 }

# ── Prerequisite checks ───────────────────────────────────────────────────────
function Test-Command ([string]$cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

Write-Header "Prerequisite checks"

if (-not (Test-Command "git"))  { Write-Fail "git not found. Install: winget install Git.Git" }
Write-Ok "git found: $(git --version)"

if (-not $LocalOnly) {
    if (-not (Test-Command "gh")) {
        Write-Fail "GitHub CLI (gh) not found. Install: winget install GitHub.cli"
    }
    Write-Ok "gh found: $(gh --version | Select-Object -First 1)"

    # Check gh auth
    $authStatus = gh auth status 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Not authenticated with GitHub CLI. Run: gh auth login"
    }
    Write-Ok "GitHub CLI authenticated."
}

# ── Resolve absolute paths ────────────────────────────────────────────────────
$RepoDir   = Resolve-Path $RepoDir   -ErrorAction Stop | Select-Object -ExpandProperty Path
$ScriptSrc = Resolve-Path $ScriptSrc -ErrorAction Stop | Select-Object -ExpandProperty Path

Write-Ok "Repo directory : $RepoDir"
Write-Ok "Script source  : $ScriptSrc"

# ── Step 1: Copy campaign v2 scripts ─────────────────────────────────────────
Write-Header "STEP 1 — Copy campaign v2 scripts"

$destScripts = Join-Path $RepoDir "experiments\campaign_v2"
New-Item -ItemType Directory -Path $destScripts -Force | Out-Null

$scriptFiles = @(
    "00_generate_distributions.py",
    "01_verify_simulations.py",
    "02_bare_state_check.py",
    "03_run_campaign.py",
    "04_build_artifacts.py",
    "05_characterisation_analysis.py",
    "Run-Campaign-V2.ps1",
    "README_campaign_v2.md"
)

foreach ($f in $scriptFiles) {
    $src = Join-Path $ScriptSrc $f
    if (Test-Path $src) {
        # Rename README_campaign_v2.md -> README.md inside the dest folder
        $destName = if ($f -eq "README_campaign_v2.md") { "README.md" } else { $f }
        Copy-Item $src (Join-Path $destScripts $destName) -Force
        Write-Ok "Copied $f"
    } else {
        Write-Warn "$f not found in $ScriptSrc — skipping"
    }
}

# ── Step 2: Copy protocol documents ──────────────────────────────────────────
Write-Header "STEP 2 — Copy protocol documents"

$destDocs = Join-Path $RepoDir "docs"
New-Item -ItemType Directory -Path $destDocs -Force | Out-Null

$docFiles = @(
    "experimental_protocol_v2.pdf",
    "experimental_protocol_v2.tex",
    "experimental_protocol.pdf",      # v1 protocol
    "experimental_protocol.tex"
)

foreach ($f in $docFiles) {
    $src = Join-Path $DocsSrc $f
    if (Test-Path $src) {
        # Normalise v1 name for clarity
        $destName = $f -replace "^experimental_protocol\.","experimental_protocol_v1."
        Copy-Item $src (Join-Path $destDocs $destName) -Force
        Write-Ok "Copied $f -> $destName"
    } else {
        Write-Warn "$f not found in $DocsSrc — skipping (add manually if needed)"
    }
}

# ── Step 3: Create artifacts placeholder ─────────────────────────────────────
Write-Header "STEP 3 — Create artifacts placeholder"

$artifactsDir = Join-Path $RepoDir "artifacts\campaign_v2"
New-Item -ItemType Directory -Path $artifactsDir -Force | Out-Null

# .gitkeep so the directory is tracked but contents are gitignored
New-Item -ItemType File -Path (Join-Path $artifactsDir ".gitkeep") -Force | Out-Null
Write-Ok "Created artifacts/campaign_v2/.gitkeep"

# ── Step 4: Git init and first commit ────────────────────────────────────────
Write-Header "STEP 4 — Git init and first commit"

Push-Location $RepoDir

# Initialise if not already a git repo
if (-not (Test-Path ".git")) {
    Write-Step "git init"
    git init -b main
    Write-Ok "Repository initialised."
} else {
    Write-Warn ".git already exists — skipping init."
}

# Stage everything
Write-Step "git add -A"
git add -A

# Check if there is anything to commit
$status = git status --porcelain
if ($status) {
    Write-Step "git commit"
    git commit -m $CommitMsg
    Write-Ok "Commit created."
} else {
    Write-Warn "Nothing to commit — working tree clean."
}

Pop-Location

# ── Step 5: Create GitHub repo and push ──────────────────────────────────────
if (-not $LocalOnly) {
    Write-Header "STEP 5 — Create GitHub repository and push"

    $visibility = if ($Private) { "--private" } else { "--public" }
    $fullName   = "$GitHubUser/$RepoName"

    # Check if repo already exists
    $exists = gh repo view $fullName 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "Repository $fullName already exists on GitHub — skipping creation."
    } else {
        Write-Step "Creating $fullName on GitHub ($( if ($Private) { 'private' } else { 'public' } )) ..."
        gh repo create $fullName `
            $visibility `
            --description "Extended Grover-Rudolph hardware characterisation campaign on SpinQ Triangulum — 7 distributions, 1050 runs, automated pipeline" `
            --source $RepoDir `
            --remote origin `
            --push
        Write-Ok "Repository created and pushed: https://github.com/$fullName"
    }

    # If repo existed, just add remote and push
    Push-Location $RepoDir
    $remotes = git remote
    if ($remotes -notcontains "origin") {
        Write-Step "Adding remote origin ..."
        git remote add origin "https://github.com/$fullName.git"
    }
    Write-Step "git push origin main"
    git push -u origin main
    Pop-Location

    Write-Ok "Push complete."
    Write-Host ""
    Write-Host "  Repository URL : https://github.com/$fullName" -ForegroundColor Cyan
    Write-Host "  Share with SpinQ by inviting them as collaborators:" -ForegroundColor Cyan
    Write-Host "  gh api repos/$fullName/collaborators/SPINQ_GITHUB_USER -X PUT -f permission=push" `
        -ForegroundColor DarkGray
    Write-Host ""

} else {
    Write-Header "LOCAL-ONLY mode — GitHub steps skipped"
    Write-Ok "Local repository ready at: $RepoDir"
    Write-Warn "To push later, run:"
    Write-Host "    cd $RepoDir" -ForegroundColor DarkGray
    Write-Host "    gh repo create $GitHubUser/$RepoName --public --source . --remote origin --push" `
        -ForegroundColor DarkGray
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Header "Done"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Add Yuefeng as collaborator on GitHub (Settings > Collaborators)" -ForegroundColor White
Write-Host "  2. Run the campaign:" -ForegroundColor White
Write-Host "     cd $RepoDir\experiments\campaign_v2" -ForegroundColor DarkGray
Write-Host "     .\Run-Campaign-V2.ps1 -DryRun   # test first" -ForegroundColor DarkGray
Write-Host "     .\Run-Campaign-V2.ps1            # real hardware" -ForegroundColor DarkGray
Write-Host ""
