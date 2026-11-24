$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$ModelRoot = Join-Path $ProjectRoot "models"
$RuntimeCache = Join-Path $ProjectRoot "runtime-cache"
New-Item -ItemType Directory -Force -Path $ModelRoot, $RuntimeCache | Out-Null
$env:HF_HOME = Join-Path $RuntimeCache "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path $RuntimeCache "torch"
$env:XDG_CACHE_HOME = $RuntimeCache
$env:TEMP = Join-Path $RuntimeCache "temp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:TORCH_HOME, $env:TEMP | Out-Null

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda was not found. Run this script from Anaconda Prompt." -ForegroundColor Red
    exit 1
}

$LocalConfig = Join-Path $ProjectRoot "local-config.ps1"
if (Test-Path -LiteralPath $LocalConfig) {
    . $LocalConfig
}

$EnvironmentName = if ($env:BGM_CONDA_ENV) { $env:BGM_CONDA_ENV } else { "bgm-v2" }
$EnvironmentList = conda env list --json | ConvertFrom-Json
$EnvironmentPath = $EnvironmentList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
if (-not $EnvironmentPath) {
    Write-Host "Conda environment '$EnvironmentName' does not exist." -ForegroundColor Red
    exit 1
}
$PythonExe = Join-Path $EnvironmentPath "python.exe"
$ClapModel = if ($env:BGM_CLAP_MODEL) { $env:BGM_CLAP_MODEL } else { "laion/clap-htsat-unfused" }
$AnalysisModel = if ($env:BGM_ANALYSIS_MODEL) { $env:BGM_ANALYSIS_MODEL } else { "MIT/ast-finetuned-audioset-10-10-0.4593" }
$TextModel = if ($env:BGM_TEXT_MODEL) { $env:BGM_TEXT_MODEL } else { "BAAI/bge-small-zh-v1.5" }

Write-Host "Downloading models to the local Hugging Face cache:" -ForegroundColor Cyan
Write-Host "  Search model: $ClapModel"
Write-Host "  Analysis model: $AnalysisModel"
Write-Host "  Chinese text model: $TextModel"
Write-Host "HF_TOKEN is optional for these public models." -ForegroundColor DarkGray

& $PythonExe -m bgm_v2.download_models --clap $ClapModel --analysis $AnalysisModel --text $TextModel --model-root $ModelRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Models are ready. Run the Conda startup script next." -ForegroundColor Green
