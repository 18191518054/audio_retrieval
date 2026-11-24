$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$RuntimeCache = Join-Path $ProjectRoot "runtime-cache"
$env:PIP_CACHE_DIR = Join-Path $RuntimeCache "pip"
$env:TEMP = Join-Path $RuntimeCache "temp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR, $env:TEMP | Out-Null

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda was not found. Run this script from Anaconda Prompt, or run 'conda init powershell' first." -ForegroundColor Red
    exit 1
}

$EnvironmentName = if ($env:BGM_CONDA_ENV) { $env:BGM_CONDA_ENV } else { "bgm-v2" }
$Existing = conda env list --json | ConvertFrom-Json
$EnvironmentExists = $Existing.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName }

if ($EnvironmentExists) {
    Write-Host "Updating Conda environment: $EnvironmentName" -ForegroundColor Cyan
    conda env update -n $EnvironmentName -f environment.yml --prune
} else {
    Write-Host "Creating Conda environment: $EnvironmentName" -ForegroundColor Cyan
    conda env create -n $EnvironmentName -f environment.yml
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Keep torch >= 2.6 for CVE-2025-32434. NVIDIA machines use CUDA by default;
# set BGM_TORCH_VARIANT=cpu before running this script to force the CPU build.
$UseCuda = (Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and ($env:BGM_TORCH_VARIANT -ne "cpu")
if ($UseCuda) {
    Write-Host "Installing secure CUDA PyTorch wheel (CUDA 12.4)..." -ForegroundColor Cyan
    conda run -n $EnvironmentName python -m pip install --upgrade "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
} else {
    Write-Host "Installing secure CPU PyTorch wheel..." -ForegroundColor Cyan
    conda run -n $EnvironmentName python -m pip install --upgrade "torch==2.6.0" "torchaudio==2.6.0" --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Environment is ready. Run the Conda startup script next." -ForegroundColor Green
