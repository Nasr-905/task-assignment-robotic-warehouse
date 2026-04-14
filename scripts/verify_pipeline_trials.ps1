param(
    [switch]$Render,
    [int]$Episodes = 1,
    [int]$SeedBase = 100,
    [string]$LogDir = "logs/trial_runs"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = Join-Path $repoRoot $LogDir
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$logFile = Join-Path $logRoot "trial_matrix_$timestamp.log"

function Set-TrialEnv {
    param(
        [hashtable]$trial
    )

    $env:TARWARE_MAX_STEPS = [string]$trial.max_steps
    $env:TARWARE_STEPS_PER_SIMULATED_SECOND = [string]$trial.steps_per_sim_s
    $env:TARWARE_RENDER_PHYSICAL_TIME_OVERLAY = "1"
    $env:TARWARE_RENDER_HUMAN_FACTORS_OVERLAY = "1"
    $env:TARWARE_RENDER_PICKER_DIAGNOSTICS_WINDOW = if ($Render) { "1" } else { "0" }

    if ($trial.motion_model -eq "physical") {
        $env:TARWARE_USE_PHYSICAL_SPEEDS = "1"
        $env:TARWARE_GRID_CELL_SIZE_M = [string]$trial.grid_cell_m
        $env:TARWARE_AGV_NOMINAL_SPEED_M_S = [string]$trial.agv_speed_m_s
        $env:TARWARE_PICKER_NOMINAL_SPEED_M_S = [string]$trial.picker_speed_m_s
        Remove-Item Env:TARWARE_AGV_CELLS_PER_STEP -ErrorAction SilentlyContinue
        Remove-Item Env:TARWARE_PICKER_CELLS_PER_STEP -ErrorAction SilentlyContinue
    }
    else {
        $env:TARWARE_USE_PHYSICAL_SPEEDS = "0"
        $env:TARWARE_AGV_CELLS_PER_STEP = [string]$trial.agv_cells_per_step
        $env:TARWARE_PICKER_CELLS_PER_STEP = [string]$trial.picker_cells_per_step
        Remove-Item Env:TARWARE_GRID_CELL_SIZE_M -ErrorAction SilentlyContinue
        Remove-Item Env:TARWARE_AGV_NOMINAL_SPEED_M_S -ErrorAction SilentlyContinue
        Remove-Item Env:TARWARE_PICKER_NOMINAL_SPEED_M_S -ErrorAction SilentlyContinue
    }
}

$trials = @(
    @{ name = "tiny_cells_baseline";   size = "tiny";       agvs = 3; pickers = 2; obs = "partial"; motion_model = "cells";    agv_cells_per_step = 1.0; picker_cells_per_step = 1.0; steps_per_sim_s = 1.0; max_steps = 120 },
    @{ name = "tiny_cells_slow";       size = "tiny";       agvs = 3; pickers = 2; obs = "partial"; motion_model = "cells";    agv_cells_per_step = 0.25; picker_cells_per_step = 0.25; steps_per_sim_s = 4.0; max_steps = 120 },
    @{ name = "small_physical_even";   size = "small";      agvs = 4; pickers = 2; obs = "partial"; motion_model = "physical"; grid_cell_m = 1.0; agv_speed_m_s = 1.0; picker_speed_m_s = 1.0; steps_per_sim_s = 2.0; max_steps = 140 },
    @{ name = "medium_physical_mixed"; size = "medium";     agvs = 5; pickers = 3; obs = "partial"; motion_model = "physical"; grid_cell_m = 0.8; agv_speed_m_s = 1.2; picker_speed_m_s = 0.8; steps_per_sim_s = 3.0; max_steps = 160 },
    @{ name = "large_cells_mixed";     size = "large";      agvs = 6; pickers = 3; obs = "partial"; motion_model = "cells";    agv_cells_per_step = 0.5; picker_cells_per_step = 0.75; steps_per_sim_s = 5.0; max_steps = 180 },
    @{ name = "xlarge_physical_slow";  size = "extralarge"; agvs = 8; pickers = 4; obs = "partial"; motion_model = "physical"; grid_cell_m = 1.0; agv_speed_m_s = 0.9; picker_speed_m_s = 0.6; steps_per_sim_s = 4.0; max_steps = 220 }
)

"Running $($trials.Count) pipeline trials. Render=$Render" | Tee-Object -FilePath $logFile -Append

for ($i = 0; $i -lt $trials.Count; $i++) {
    $trial = $trials[$i]
    $seed = $SeedBase + $i
    Set-TrialEnv -trial $trial

    $cmdArgs = @(
        "-m", "tarware.main",
        "classical", "eval",
        "--episodes", "$Episodes",
        "--seed", "$seed",
        "--size", "$($trial.size)",
        "--agvs", "$($trial.agvs)",
        "--pickers", "$($trial.pickers)",
        "--obs-type", "$($trial.obs)"
    )
    if ($Render) {
        $cmdArgs += "--render"
    }

    "" | Tee-Object -FilePath $logFile -Append
    "=== Trial $($i + 1)/$($trials.Count): $($trial.name) ===" | Tee-Object -FilePath $logFile -Append
    "env: steps_per_sim_s=$($trial.steps_per_sim_s) model=$($trial.motion_model) max_steps=$($trial.max_steps)" | Tee-Object -FilePath $logFile -Append

    $stdoutPath = Join-Path $env:TEMP "tarware_trial_stdout_$timestamp`_$i.log"
    $stderrPath = Join-Path $env:TEMP "tarware_trial_stderr_$timestamp`_$i.log"

    $startInfo = @{
        FilePath = "python"
        ArgumentList = $cmdArgs
        NoNewWindow = $true
        Wait = $true
        PassThru = $true
        RedirectStandardOutput = $stdoutPath
        RedirectStandardError = $stderrPath
    }
    $proc = Start-Process @startInfo

    $outputLines = @()
    if (Test-Path $stdoutPath) {
        $outputLines += Get-Content -Path $stdoutPath
    }
    if (Test-Path $stderrPath) {
        $outputLines += Get-Content -Path $stderrPath
    }

    if ($outputLines.Count -gt 0) {
        $outputLines | Tee-Object -FilePath $logFile -Append
    }

    if (Test-Path $stdoutPath) {
        Remove-Item -Path $stdoutPath -Force
    }
    if (Test-Path $stderrPath) {
        Remove-Item -Path $stderrPath -Force
    }

    if ($proc.ExitCode -ne 0) {
        "FAILED trial=$($trial.name) exit_code=$($proc.ExitCode)" | Tee-Object -FilePath $logFile -Append
        throw "Trial $($trial.name) failed"
    }
}

"" | Tee-Object -FilePath $logFile -Append
"All trials completed successfully." | Tee-Object -FilePath $logFile -Append
"Log file: $logFile" | Tee-Object -FilePath $logFile -Append
