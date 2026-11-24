$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$ModelRoot = Join-Path $ProjectRoot "models"
$RuntimeCache = Join-Path $ProjectRoot "runtime-cache"
New-Item -ItemType Directory -Force -Path $RuntimeCache | Out-Null
$env:HF_HOME = Join-Path $RuntimeCache "huggingface"
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
$env:TORCH_HOME = Join-Path $RuntimeCache "torch"
$env:XDG_CACHE_HOME = $RuntimeCache
$env:TEMP = Join-Path $RuntimeCache "temp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:HF_HUB_CACHE, $env:TORCH_HOME, $env:TEMP | Out-Null
if (Test-Path -LiteralPath (Join-Path $ModelRoot "clap")) { $env:BGM_CLAP_MODEL = Join-Path $ModelRoot "clap" }
if (Test-Path -LiteralPath (Join-Path $ModelRoot "ast")) { $env:BGM_ANALYSIS_MODEL = Join-Path $ModelRoot "ast" }
if (Test-Path -LiteralPath (Join-Path $ModelRoot "bge")) { $env:BGM_TEXT_MODEL = Join-Path $ModelRoot "bge" }
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda was not found. Run from Anaconda Prompt, or run 'conda init powershell' first." -ForegroundColor Red
    exit 1
}

# Every setting is optional. Copy local-config.example.ps1 to local-config.ps1 to customize.
function Import-Utf8Config([string]$Path) {
    # PowerShell 5.1 assumes ANSI for BOM-less files. Parse explicitly as UTF-8
    # so a Chinese comment cannot swallow the following environment assignment.
    $ConfigSource = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    & ([scriptblock]::Create($ConfigSource))
}
$LocalConfig = Join-Path $ProjectRoot "local-config.ps1"
if (Test-Path -LiteralPath $LocalConfig) {
    Import-Utf8Config $LocalConfig
} else {
    # Build the Chinese filename from Unicode code points. Windows PowerShell
    # 5.1 may otherwise decode a UTF-8-without-BOM script as the system ANSI
    # code page and silently look for a mojibake filename.
    $ChineseConfigName = [string]([char]0x672C) + [char]0x5730 + [char]0x914D + [char]0x7F6E + ".ps1"
    $ChineseLocalConfig = Join-Path $ProjectRoot $ChineseConfigName
    if (Test-Path -LiteralPath $ChineseLocalConfig) {
        Import-Utf8Config $ChineseLocalConfig
    }
}

$EnvironmentName = if ($env:BGM_CONDA_ENV) { $env:BGM_CONDA_ENV } else { "bgm-v2" }
$EnvironmentList = conda env list --json | ConvertFrom-Json
$EnvironmentPath = $EnvironmentList.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvironmentName } | Select-Object -First 1
if (-not $EnvironmentPath) {
    Write-Host "Conda environment '$EnvironmentName' does not exist. Run the environment installer first." -ForegroundColor Yellow
    exit 1
}

$PythonExe = Join-Path $EnvironmentPath "python.exe"
$CondaBin = Join-Path $EnvironmentPath "Library\bin"
$env:PATH = "$EnvironmentPath;$CondaBin;$env:PATH"
$env:CONDA_PREFIX = $EnvironmentPath
$env:CONDA_DEFAULT_ENV = $EnvironmentName
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:TRANSFORMERS_VERBOSITY = "error"
if (-not $env:BGM_FFMPEG) { $env:BGM_FFMPEG = Join-Path $CondaBin "ffmpeg.exe" }
if (-not $env:BGM_FFPROBE) { $env:BGM_FFPROBE = Join-Path $CondaBin "ffprobe.exe" }

Write-Host "Checking environment..." -ForegroundColor Cyan
& $PythonExe -m bgm_v2 doctor
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$HostAddress = if ($env:BGM_HOST) { $env:BGM_HOST } else { "127.0.0.1" }
$Port = if ($env:BGM_PORT) { $env:BGM_PORT } else { "8765" }
Write-Host "Open in browser: http://${HostAddress}:${Port}" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
& $PythonExe -m bgm_v2 serve
