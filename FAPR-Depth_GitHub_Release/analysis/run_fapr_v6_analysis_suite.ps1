param(
    [ValidateSet("core", "stress-quick", "stress-paper", "all")]
    [string]$Mode = "core"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
    Write-Host ""
    Write-Host ("python " + ($Arguments -join " ")) -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Run-Core {
    Run-Python "analyze_fapr_failure_posterior.py"
    Run-Python "analyze_fapr_failure_regions.py"
    Run-Python "visualize_fapr_expert_routing.py"
    Run-Python "analyze_fapr_safe_correction.py" "--phase" "joint"
    Run-Python "analyze_fapr_risk_calibration.py" "--phase" "joint"
}

Write-Host "============================================================"
Write-Host "FAPR-Depth v6 paper analysis suite"
Write-Host "Mode: $Mode"
Write-Host "============================================================"

switch ($Mode) {
    "core" {
        Run-Core
    }
    "stress-quick" {
        Run-Python "stress_test_fapr_relative_prior.py" "--profile" "quick" "--max-shards" "64"
    }
    "stress-paper" {
        Run-Python "stress_test_fapr_relative_prior.py" "--profile" "paper" "--max-shards" "512"
    }
    "all" {
        Run-Core
        Run-Python "stress_test_fapr_relative_prior.py" "--profile" "paper" "--max-shards" "512"
    }
}
