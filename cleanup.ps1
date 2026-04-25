# cleanup.ps1
# Run once from repo root to remove scratch/duplicate files and tidy the folder.
# Usage (from repo root):  .\cleanup.ps1

$root = $PSScriptRoot

# ── Test / scratch files ───────────────────────────────────────────────────────
$scratch = @(
    "scripts\ml_models\test12.py",
    "scripts\data\test_write.py",
    "data\eda_charts.ipynb",
    "scripts\analysis\analysis_kc.ipynb"
)
foreach ($f in $scratch) {
    $p = Join-Path $root $f
    if (Test-Path $p) { Remove-Item $p -Force; Write-Host "Deleted $p" }
}

# ── scripts/analysis/CleanedData/ (duplicate data / notebooks in wrong place) ──
$cdDir = Join-Path $root "scripts\analysis\CleanedData"
if (Test-Path $cdDir) {
    Remove-Item $cdDir -Recurse -Force
    Write-Host "Deleted $cdDir"
}

# ── Root-level catboost_info/ (gitignored, remove the stale folder) ────────────
$cbDir = Join-Path $root "catboost_info"
if (Test-Path $cbDir) {
    Remove-Item $cbDir -Recurse -Force
    Write-Host "Deleted $cbDir"
}

Write-Host "`nCleanup complete."
